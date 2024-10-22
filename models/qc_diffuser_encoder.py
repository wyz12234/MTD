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


class QCDiffuserEncoder(nn.Module):
    def __init__(self, d_k=128, map_attr=3, dropout=0.0, k_attr=7, L_enc=2, feedforward_dim=512, num_heads=16, input_dim=6):
        super(QCDiffuserEncoder, self).__init__()

        self.d_k = d_k
        self.input_dim = input_dim
        self.map_attr = map_attr
        self.dropout = dropout
        self.k_attr = k_attr
        self.L_enc = L_enc
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.neighbor_inds = [0, 1, 2, 3, 4, 5]
        self.social_attn_radius = 30.0

        self.pos_encoder = PositionalEncoding(d_model=self.d_k, dropout=self.dropout)
        self.map_encoder = MapEncoder(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.agent_hist_embedder = nn.Sequential(nn.Linear(self.k_attr, self.d_k))
        self.agent_future_embedder = nn.Sequential(nn.Linear(self.input_dim, self.d_k))
        self.pos_mlp = nn.Sequential(nn.Linear(self.d_k * 3, self.d_k), 
                                     nn.ReLU(), 
                                     nn.Linear(self.d_k, self.d_k))
        
        self.temporal_encoder_layers= []
        self.map_hist_encoder_layers = []
        self.agent_agent_encoder_layers = []
        for _ in range(self.L_enc):
            temp_temporal_transformerlayer = nn.TransformerEncoderLayer(
                d_model=self.d_k, nhead=self.num_heads, dim_feedforward=self.feedforward_dim, dropout=self.dropout, batch_first=True)
            self.temporal_encoder_layers.append(temp_temporal_transformerlayer)
            temp_map_hist_layer = MapAgentEncoder(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.feedforward_dim)
            self.map_hist_encoder_layers.append(temp_map_hist_layer)
            temp_agent_agent_layer = AgentAgentEncoder(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.feedforward_dim)
            self.agent_agent_encoder_layers.append(temp_agent_agent_layer)


        self.temporal_encoder_layers = nn.ModuleList(self.temporal_encoder_layers)
        self.map_hist_encoder_layers = nn.ModuleList(self.map_hist_encoder_layers)
        self.agent_agent_encoder_layers = nn.ModuleList(self.agent_agent_encoder_layers)


        




        
    
    def forward(self, x_noisy, auxiliaries_info):
        B, M, T_fut, _ = x_noisy.shape
        T_hist = auxiliaries_info['agents_hist'].shape[-2]
        # x_noisy [x, y, vel, yaw, acc, yawvel]
        agent_hist = auxiliaries_info['agents_hist'] # (x,y,cos,sin,v,l,w,avail)
        map_polyline = auxiliaries_info['map_feat'] # (B, M, S_seg, S_p, K_map)
        agent_hist, agent_hist_mask = agent_hist[..., :-1], (1.0 - agent_hist[..., -1]).type(torch.BoolTensor).to(agent_hist.device)
        # agent_hist_mast (B, M, T_hist)

        # 1. encoding each agent map features

        # (S_seg, B, M, d_k) -> (S_seg, B, M, d_k+1) add availability dimension
        nan_inds = torch.sum(torch.isnan(map_polyline).float(), dim=-1) > 0
        avail = torch.ones_like(map_polyline[...,0])
        avail[nan_inds] = 0
        map_polyline = torch.nan_to_num(map_polyline, nan=0.0)
        map_polyline = torch.cat([map_polyline, avail.unsqueeze(-1)], dim=-1)
        # extract map vectors
        # (B, M, S_seg, P_per_seg, map_attr) -> (S_seg, B, M, d_k)
        map_feat, road_segs_masks = self.map_encoder(map_polyline)
        # (S_seg, B, M, d_k) -> (S_seg, BxM, d_k) -> (BxM, S_seg, d_k)
        map_feat = map_feat.view(-1, B * M, self.d_k).permute(1, 0, 2)
        # (B, M, S_seg) -> (BxM, S_seg)
        road_segs_masks = road_segs_masks.view(B * M, -1)

        # 2. encoding agent history

        # 2.1 embedding agent history
        agent_hist_emb = self.agent_hist_embedder(agent_hist)

        for i in range(self.L_enc):

            # 2.2 encoding temporal agent history self attention
            agent_hist_emb = self.hist_temporal_fn(agent_hist_emb, agent_hist_mask, self.temporal_encoder_layers[i])

            # 2.3 encoding map agent history attention
            agent_hist_emb = self.map_hist_encoder_layers[i](map_feat, agent_hist_emb, road_segs_masks)

            # 2.4 encoding agent agent history attention
            agent_hist_emb = self.hist_agent_agent_fn(agent_hist_emb, auxiliaries_info, self.agent_agent_encoder_layers[i])

        # 3. encoding agent future
        agent_future_emb = self.agent_future_embedder(x_noisy)

        agent_emb = torch.cat([agent_hist_emb, agent_future_emb], dim=-2)


        return {
            'agent_emb': agent_emb,
            'agent_mask': agent_hist_mask,
            'agent_hist': agent_hist_emb,
            'agent_future': agent_future_emb,
            'map_feat': map_feat,
            'road_segs_masks': road_segs_masks
        }



    def hist_temporal_fn(self, agent_hist, agent_hist_mask, layer):
        B, M, T_hist, _ = agent_hist.shape

        agent_hist = agent_hist.reshape(B * M, T_hist, -1)
        agent_hist_mask = agent_hist_mask.reshape(B * M, T_hist)

        # Ensure agent's don't exist NaNs
        agent_hist_mask[:, -1][agent_hist_mask.sum(-1) == T_hist] = False

        # (BxM, T_hist, d_k)
        agent_hist = layer(self.pos_encoder(agent_hist), src_key_padding_mask=agent_hist_mask)

        return agent_hist.reshape(B, M, T_hist, -1)
    
    def hist_agent_agent_fn(self, agent_hist, auxiliaries_info, layer):
        # attention mask
        B, M, T_hist, _ = agent_hist.shape # (B, M, T_hist, d_k)
        # (M, M) -> (M, M, 1) -> (M, M, M) -> (M, M*M)
        attn_block_mask = torch.eye(M, device=agent_hist.device).unsqueeze(-1).repeat(1, 1, M).reshape(M, M * M)
        attn_block_mask = (1 - attn_block_mask).to(torch.bool)
        # (M, M*M) -> (1, M, M*M) -> (B*T_all*num_heads, M, M*M)
        attn_block_mask = attn_block_mask.unsqueeze(0).repeat(B * T_hist * self.num_heads, 1, 1)
        
        # (M, M) -> (M, 1, M) -> (M, M, M) -> (M, M*M)
        attn_self_unmask = torch.eye(M, device=agent_hist.device).unsqueeze(1).repeat(1, M, 1).view(M, M * M)
        attn_self_unmask = (1 - attn_self_unmask).type(torch.bool)
        # (M, M*M) -> (B*T_all*self.num_heads, M, M*M)
        attn_self_unmask = attn_self_unmask.unsqueeze(0).repeat(B * T_hist * self.num_heads, 1, 1)
        attn_self_unmask = attn_self_unmask | attn_block_mask

        # (B, M1, M2, T_all)
        neighbor_mask = (1.0 - auxiliaries_info['neighbor_hist'][..., -1]).type(torch.bool).to(agent_hist.device)
        # (B, M1, M2, T_all) -> (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
        attn_mask = neighbor_mask.permute(0, 3, 1, 2).unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B * T_hist, M, M * M)
        # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * T_hist*self.num_heads, M, M * M)

        neighbor_pos = auxiliaries_info['neighbor_hist'][..., (0, 1)]
        # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
        neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0, 3, 1, 2)

        neighbor_dist_mask = neighbor_dist > self.social_attn_radius
        # (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
        neighbor_dist_mask = neighbor_dist_mask.unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B * T_hist, M, M * M)
        # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
        neighbor_dist_mask = neighbor_dist_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * T_hist * self.num_heads, M, M * M)

        # (B*T_all*num_heads, M1, M1*M2)
        attn_mask = attn_mask | attn_block_mask | neighbor_dist_mask
        # unmask self-attention to avoid invalid values
        if_nan = torch.sum(attn_mask, dim=-1) == M * M
        attn_mask[if_nan] = attn_mask[if_nan] & attn_self_unmask[if_nan]

        neighbor_hist_feat = auxiliaries_info['neighbor_hist'][..., self.neighbor_inds]

        # (B, M, M, T_hist, edge_attr) -> (B * T_hist, M * M, d_k)
        neighbor_hist_feat = neighbor_hist_feat.permute(0, 3, 1, 2, 4).reshape(B * T_hist, M * M, -1)
        position_feat = gen_sineembed_for_position(neighbor_hist_feat[..., :2], self.d_k)
        yaw_feat = gen_sineembed_for_position(torch.arctan2(neighbor_hist_feat[..., 3], neighbor_hist_feat[..., 2]).unsqueeze(-1), self.d_k)
        velocity_feat = gen_sineembed_for_position(neighbor_hist_feat[..., 4:], self.d_k)
        pos_feat = self.pos_mlp(torch.cat([position_feat, yaw_feat, velocity_feat], dim=-1))

        # (B, M, T_hist, d_k) -> (B, T_hist, M, d_k) -> (B * T_hist, M, d_k) -> (B * T_hist, M, M, d_k)
        agent_hist_expand = agent_hist.permute(0, 2, 1, 3).reshape(B * T_hist, M, -1).unsqueeze(1).repeat(1, M, 1, 1).reshape(B * T_hist, M * M, -1)
        agent_hist_expand = agent_hist_expand + pos_feat
        agent_hist = agent_hist.permute(0, 2, 1, 3).reshape(B * T_hist, M, -1)

        agent_hist = layer(agent_hist, agent_hist_expand, attn_mask)

        return agent_hist.reshape(B, T_hist, M, -1).permute(0, 2, 1, 3)
        

class MapEncoder(nn.Module):
    '''
    This class operates on the multi-agent road lanes provided as a tensor with shape
    (B, num_agents, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoder, self).__init__()
        self.dropout = dropout
        self.d_k = d_k

        self.map_attr = map_attr

        # Seed parameters for the map
        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        self.road_pts_lin = nn.Sequential(nn.Linear(self.map_attr, self.d_k))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            nn.Linear(self.d_k, self.d_k*3), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(self.d_k*3, self.d_k),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, :, -1], dim=3) == 0
        road_pts_mask = (1.0 - roads[:, :, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[3])

        # The next lines ensure that we do not obtain NaNs during training for missing agents or for empty roads.
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[3]] = False  # for empty agents
        road_segment_mask[:, :, 0][road_segment_mask.sum(-1) == road_segment_mask.shape[2]] = False  # for empty roads
        return road_segment_mask, road_pts_mask

    def forward(self, roads):
        '''
        :param roads: (B, M, S, P, k_attr+1)  where B is batch size, M is num_agents, S is num road segments, P is
        num pts per road segment.
        :return: embedded road segments with shape (S)
        '''
        B, M, S, P, _ = roads.shape
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :, :self.map_attr]).view(B*M*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        map_seeds = self.map_seeds.repeat(1, B * M * S, 1)
        # agents_emb = agents_emb[-1].detach().unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=map_seeds, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, M, S, -1)

        return road_seg_emb.permute(2, 0, 1, 3), road_segment_mask
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
    
class MapAgentEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super(MapAgentEncoder, self).__init__()
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
    
class AgentAgentEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super(AgentAgentEncoder, self).__init__()
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

def gen_sineembed_for_position(pos_tensor, hidden_dim=128):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 128)
    if pos_tensor.size(-1) == 1:
        scale = 2 * math.pi
        pos = pos_tensor[:, :, 0] * scale
        dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)
        pos = pos[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
    else:
        half_hidden_dim = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos