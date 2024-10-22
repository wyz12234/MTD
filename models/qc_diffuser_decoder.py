import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math


import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action
from tbsim.models.diffuser_helpers import EMA
from tbsim.models.scenediffuser import SceneDiffuserModel
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates, add_scene_dim_to_agent_data, get_stationary_mask
from trajdata.utils.arr_utils import angle_wrap
from utils.diffuser_utils import cosine_beta_schedule, convert_state_to_state_and_action, extract, unicyle_forward_dynamics

import tbsim.dynamics as dynamics
from models.qc_diffuser_encoder import PositionalEncoding, gen_sineembed_for_position

class QCDiffuserDecoder(nn.Module):
    def __init__(self, d_k=128, dropout=0.0, L_enc=2, feedforward_dim=512, num_heads=16, output_dim=2, horizon=52):
        super(QCDiffuserDecoder, self).__init__()

        self.d_k = d_k
        self.dropout = dropout
        self.L_enc = L_enc
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.neighbor_inds = [0, 1, 2, 3, 4, 5]
        self.social_attn_radius = 30.0
        self.horizon = horizon

        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)
        self.pos_mlp = nn.Sequential(nn.Linear(self.d_k * 3, self.d_k), 
                                     nn.ReLU(), 
                                     nn.Linear(self.d_k, self.d_k))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.d_k),
            nn.Linear(self.d_k, self.d_k * 4),
            nn.Mish(),
            nn.Linear(self.d_k * 4, self.d_k),
        )

        self.temporal_decoder_layers= []
        self.map_agent_decoder_layers = []
        self.agent_agent_decoder_layers = []
        for _ in range(self.L_enc):
            temp_temporal_transformerlayer = nn.TransformerEncoderLayer(
                d_model=self.d_k, nhead=self.num_heads, dim_feedforward=self.feedforward_dim, dropout=self.dropout, batch_first=True)
            self.temporal_decoder_layers.append(temp_temporal_transformerlayer)
            temp_map_agent_layer = MapAgentDecoder(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.feedforward_dim)
            self.map_agent_decoder_layers.append(temp_map_agent_layer)
            temp_agent_agent_layer = AgentAgentDecoder(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.feedforward_dim)
            self.agent_agent_decoder_layers.append(temp_agent_agent_layer)

        self.temporal_decoder_layers = nn.ModuleList(self.temporal_decoder_layers)
        self.map_agent_decoder_layers = nn.ModuleList(self.map_agent_decoder_layers)
        self.agent_agent_decoder_layers = nn.ModuleList(self.agent_agent_decoder_layers)
        self.output_transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=self.d_k, nhead=self.num_heads, dim_feedforward=self.feedforward_dim, dropout=self.dropout, batch_first=True), num_layers=2)

        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.ReLU(),
            nn.Linear(self.d_k, self.d_k),
            nn.ReLU(),
            nn.Linear(self.d_k, self.output_dim),
        )

    def forward(self, auxiliaries_info, encoder, t):
        agent_emb = encoder['agent_emb']
        agent_mask = encoder['agent_mask']
        map_feat = encoder['map_feat']
        road_segs_masks = encoder['road_segs_masks'] # (BxM, S_seg)
        B, M, T_total, _ = agent_emb.shape 

        # 1, embedding time
        time_emb = self.time_mlp(t).unsqueeze(1).unsqueeze(1)
        agent_emb = agent_emb + time_emb

        for i in range(self.L_enc):
            # 2. decoding temporal agent self-attention
            agent_emb, agent_mask_all = self.temporal_fn(agent_emb, agent_mask, self.temporal_decoder_layers[i])

            # 3. decoding map agent attention
            agent_emb = self.map_agent_decoder_layers[i](map_feat, agent_emb, road_segs_masks)

            # 4. decoding agent agent attention
            agent_emb = self.agent_agent_fn(agent_emb, auxiliaries_info, self.agent_agent_decoder_layers[i])

        # 5. output transformer
        agent_emb = self.pos_encoder(agent_emb.reshape(B * M, T_total, -1))
        q = agent_emb[..., -self.horizon:, :]
        kv = agent_emb
        agent_emb = self.output_transformer(q, kv, memory_key_padding_mask=agent_mask_all, tgt_key_padding_mask=agent_mask_all[..., -self.horizon:])
        agent_emb = agent_emb.reshape(B, M, self.horizon, -1)

        # 5. output mlp
        noise = self.output_mlp(agent_emb)

        return noise

    def temporal_fn(self, agent_emb, agent_mask, layer):
        B, M, T_total, _ = agent_emb.shape
        T_obs = agent_mask.shape[-1]
        T_fut = T_total - T_obs

        agent_emb = agent_emb.reshape(B * M, T_total, -1)

        agent_mask = agent_mask.reshape(B * M, T_obs)
        agent_mask_fut = agent_mask[..., -1].unsqueeze(-1).repeat(1, T_fut)
        agent_mask = torch.cat((agent_mask, agent_mask_fut), dim=-1) # (BxM, T_total)

        # Ensure agent's don't exist NaNs
        agent_mask[:, T_obs - 1][agent_mask.sum(-1) == T_total] = False

        # (BxM, T_total, d_k)
        agent_emb = layer(self.pos_encoder(agent_emb), src_key_padding_mask=agent_mask)

        return agent_emb.reshape(B, M, T_total, -1), agent_mask
    
    def agent_agent_fn(self, agent_emb, auxiliaries_info, layer):
        # attention mask
        B, M, T_total, _ = agent_emb.shape # (B, M, T_total, d_k)
        # (M, M) -> (M, M, 1) -> (M, M, M) -> (M, M*M)
        attn_block_mask = torch.eye(M, device=agent_emb.device).unsqueeze(-1).repeat(1, 1, M).reshape(M, M * M)
        attn_block_mask = (1 - attn_block_mask).to(torch.bool)
        # (M, M*M) -> (1, M, M*M) -> (B*T_all*num_heads, M, M*M)
        attn_block_mask = attn_block_mask.unsqueeze(0).repeat(B * T_total * self.num_heads, 1, 1)
        
        # (M, M) -> (M, 1, M) -> (M, M, M) -> (M, M*M)
        attn_self_unmask = torch.eye(M, device=agent_emb.device).unsqueeze(1).repeat(1, M, 1).view(M, M * M)
        attn_self_unmask = (1 - attn_self_unmask).type(torch.bool)
        # (M, M*M) -> (B*T_all*self.num_heads, M, M*M)
        attn_self_unmask = attn_self_unmask.unsqueeze(0).repeat(B * T_total * self.num_heads, 1, 1)
        attn_self_unmask = attn_self_unmask | attn_block_mask

        # (B, M1, M2, T_all)
        neighbor_mask = (1.0 - auxiliaries_info['neighbor_feat'][..., -1]).type(torch.bool).to(agent_emb.device)
        # (B, M1, M2, T_all) -> (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
        attn_mask = neighbor_mask.permute(0, 3, 1, 2).unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B * T_total, M, M * M)
        # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * T_total * self.num_heads, M, M * M)

        neighbor_pos = auxiliaries_info['neighbor_feat'][..., (0, 1)]
        # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
        neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0, 3, 1, 2)

        neighbor_dist_mask = neighbor_dist > self.social_attn_radius
        # (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
        neighbor_dist_mask = neighbor_dist_mask.unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B * T_total, M, M * M)
        # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
        neighbor_dist_mask = neighbor_dist_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * T_total * self.num_heads, M, M * M)

        # (B*T_all*num_heads, M1, M1*M2)
        attn_mask = attn_mask | attn_block_mask | neighbor_dist_mask
        # unmask self-attention to avoid invalid values
        if_nan = torch.sum(attn_mask, dim=-1) == M * M
        attn_mask[if_nan] = attn_mask[if_nan] & attn_self_unmask[if_nan]

        neighbor_feat = auxiliaries_info['neighbor_feat'][..., self.neighbor_inds]

        # (B, M, M, T_hist, edge_attr) -> (B * T_hist, M * M, d_k)
        neighbor_feat = neighbor_feat.permute(0, 3, 1, 2, 4).reshape(B * T_total, M * M, -1)
        position_feat = gen_sineembed_for_position(neighbor_feat[..., :2], self.d_k)
        yaw_feat = gen_sineembed_for_position(torch.arctan2(neighbor_feat[..., 3], neighbor_feat[..., 2]).unsqueeze(-1), self.d_k)
        velocity_feat = gen_sineembed_for_position(neighbor_feat[..., 4:], self.d_k)
        pos_feat = self.pos_mlp(torch.cat([position_feat, yaw_feat, velocity_feat], dim=-1))

        # (B, M, T_total, d_k) -> (B, T_total, M, d_k) -> (B * T_total, M, d_k) -> (B * T_total, M, M, d_k)
        agent_emb_expand = agent_emb.permute(0, 2, 1, 3).reshape(B * T_total, M, -1).unsqueeze(1).repeat(1, M, 1, 1).reshape(B * T_total, M * M, -1)
        agent_emb_expand = agent_emb_expand + pos_feat
        agent_emb = agent_emb.permute(0, 2, 1, 3).reshape(B * T_total, M, -1)

        agent_emb = layer(agent_emb, agent_emb_expand, attn_mask)

        return agent_emb.reshape(B, T_total, M, -1).permute(0, 2, 1, 3)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MapAgentDecoder(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super(MapAgentDecoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, map_feat, agent_feat, map_mask):
        B, M, T, _ = agent_feat.shape
        agent_feat = agent_feat.reshape(B * M, T, -1)
        attn_out, _ = self.multihead_attn(query=agent_feat, key=map_feat, value=map_feat, key_padding_mask=map_mask)
        attn_out = self.norm1(attn_out + agent_feat)
        ff_out = self.feedforward(attn_out)
        return self.norm2(ff_out + attn_out).reshape(B, M, T, -1)
    
class AgentAgentDecoder(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super(AgentAgentDecoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, agent_feat, agent_feat_expand, attn_mask):
        attn_out, _ = self.multihead_attn(query=agent_feat, key=agent_feat_expand, value=agent_feat_expand, attn_mask=attn_mask)
        attn_out = self.norm1(attn_out + agent_feat)
        ff_out = self.feedforward(attn_out)
        return self.norm2(ff_out + attn_out)