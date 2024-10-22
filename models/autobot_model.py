import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import math
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace
from scipy import special
from torch.optim.optimizer import Optimizer
from tbsim.policies.common import Action
import tbsim.dynamics as dynamics
from tbsim.utils.trajdata_utils import maybe_pad_neighbor, trajdata2posyawspeed, convert_nusc_type_to_lyft_type
from trajdata.data_structures.state import StateTensor, StateArray
from tbsim.algos.algo_utils import yaw_from_pos
from tbsim.utils.geometry_utils import transform_agents_to_world, transform_points_tensor
from tbsim.models.diffuser_helpers import convert_state_to_state_and_action, unicyle_forward_dynamics

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class PositionalEncoding(nn.Module):
    '''
    Standard positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)

class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        road_segment_mask[:, 0][road_segment_mask.sum(-1) == road_segment_mask.shape[1]] = False
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask

class AutoBotEgoModel(pl.LightningModule):
    def __init__(self, d_k=128, c=10, T=52, L_enc=2, dropout=0.1, k_attr=7, map_attr=3,
                 num_heads=16, L_dec=2, tx_hidden_size=384, use_map_lanes=True):
        super(AutoBotEgoModel, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.fine_tune = False
        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = d_k
        self._M = None
        self.c = c
        self.T = T
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec= L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_lanes = use_map_lanes

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(k_attr, d_k)))

        # ============================== AutoBot-Ego ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_lanes:
            self.map_encoder = MapEncoderPts(d_k=d_k, map_attr=map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        # ============================== AutoBot-Ego DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(c, 1, d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        if self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.epoch_ade_losses = []
        self.epoch_fde_losses = []
        self.epoch_mode_probs = []
        self.dyn = dynamics.Unicycle(
                    "dynamics",
                    max_steer=0.5,
                    max_yawvel=math.pi * 2.0,
                    acce_bound=(-10, 8),
                )
        self.dt = 0.1
        self.choose = None

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).type(torch.BoolTensor).to(env_masks_orig.device)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks[:, -1][temp_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M+1))
        agents_soc_emb = agents_soc_emb.view(self._M+1, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def forward(self, ego_in, agents_in, roads):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        B = ego_in.size(0)
        self._M = agents_in.size(2)

        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        # Process through AutoBot's encoder
        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        ego_soctemp_emb = agents_emb[:, :, 0]  # take ego-agent encodings only.

        # Process map information
        if self.use_map_lanes:
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
            map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B*self.c, self.d_k)
            road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B*self.c, -1)


        # Repeat the tensors for the number of modes for efficient forward pass.
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B*self.c, self.d_k)

        # AutoBot-Ego Decoding
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B*self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        for d in range(self.L_dec):
            if self.use_map_lanes and d == 1:
                ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,
                                                       key_padding_mask=road_segs_masks)[0]
                out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        # Mode prediction
        mode_params_emb = self.P.repeat(1, B, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        if self.use_map_lanes:
            mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        # return  [c, T, B, 5], [B, c]
        return out_dists, mode_probs
    
    def training_step(self, batch, batch_idx):
        if self.fine_tune:
            agent_hist = batch["agent_hist"]
            neigh_hist = batch["neigh_hist"]
            map_polyline = batch["map_polyline"]
            batch['target_positions'] = batch['target_pos']
        else:
            agent_hist, neigh_hist, map_polyline = self.normal_data(batch)

        pred_obs, mode_probs = self(agent_hist, neigh_hist, map_polyline)

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, batch['target_positions'], mode_probs,
                                                                                   entropy_weight=40.0,
                                                                                   kl_weight=20.0,
                                                                                   use_FDEADE_aux_loss=True)
        loss = nll_loss + kl_loss + adefde_loss

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/nll_loss', nll_loss, prog_bar=True)
        self.log('train/kl_loss', kl_loss, prog_bar=True)
        self.log('train/adefde_loss', adefde_loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        if self.fine_tune:
            agent_hist = batch["agent_hist"]
            neigh_hist = batch["neigh_hist"]
            map_polyline = batch["map_polyline"]
            batch['target_positions'] = batch['target_pos']
        else:
            agent_hist, neigh_hist, map_polyline = self.normal_data(batch)
        
        pred_obs, mode_probs = self(agent_hist, neigh_hist, map_polyline)

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, batch['target_positions'], mode_probs,
                                                                                   entropy_weight=40.0,
                                                                                   kl_weight=20.0,
                                                                                   use_FDEADE_aux_loss=True)
        loss = nll_loss + kl_loss + adefde_loss

        self.log('val/loss', loss)
        self.log('val/nll_loss', nll_loss)
        self.log('val/kl_loss', kl_loss)
        self.log('val/adefde_loss', adefde_loss)

        ade_losses, fde_losses = self._compute_ego_errors(pred_obs, batch['target_positions'])
        self.epoch_ade_losses.append(ade_losses)
        self.epoch_fde_losses.append(fde_losses)
        self.epoch_mode_probs.append(mode_probs)

        return loss
    
    def normal_data(self, batch):
        overwrite_nan=True
        maybe_pad_neighbor(batch)
        fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"], nan_to_zero=overwrite_nan)
        hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"], nan_to_zero=overwrite_nan)
        curr_speed = hist_speed[..., -1]
        curr_state = batch["curr_agent_state"]
        assert isinstance(curr_state, StateTensor) or isinstance(curr_state, StateArray)
        h1, h2 = curr_state[:, -1], curr_state.heading[...,0]
        p1, p2 = curr_state[:, :2], curr_state.position
        assert torch.all(h1[~torch.isnan(h1)] == h2[~torch.isnan(h2)])
        assert torch.all(p1[~torch.isnan(p1)] == p2[~torch.isnan(p2)])
        curr_yaw = curr_state.heading[...,0]
        curr_pos = curr_state.position

        # convert nuscenes types to l5kit types
        agent_type = batch["agent_type"]
        agent_type = convert_nusc_type_to_lyft_type(agent_type)
        
        # mask out invalid extents
        agent_hist_extent = batch["agent_hist_extent"]
        agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.
        neigh_indices = batch["neigh_indices"]
        neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"], nan_to_zero=overwrite_nan)
        neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"], nan_to_zero=overwrite_nan)
        neigh_curr_speed = neigh_hist_speed[..., -1]
        neigh_types = batch["neigh_types"]
        # convert nuscenes types to l5kit types
        neigh_types = convert_nusc_type_to_lyft_type(neigh_types)
        # mask out invalid extents
        neigh_hist_extents = batch["neigh_hist_extents"]
        neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

        world_from_agents = torch.inverse(batch["agents_from_world_tf"])

        bsize = batch["agents_from_world_tf"].shape[0]

        extent_scale = 1.0

        d = dict(
            map_names=batch["map_names"],
            target_positions=fut_pos,
            target_yaws=fut_yaw,
            target_availabilities=fut_mask,
            history_positions=hist_pos,
            history_yaws=hist_yaw,
            history_speeds=hist_speed,
            history_availabilities=hist_mask,
            curr_speed=curr_speed,
            centroid=curr_pos,
            yaw=curr_yaw,
            type=agent_type,
            extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
            agent_from_world=batch["agents_from_world_tf"],
            world_from_agent=world_from_agents,
            all_other_agents_indices=neigh_indices,
            all_other_agents_history_positions=neigh_hist_pos,
            all_other_agents_history_yaws=neigh_hist_yaw,
            all_other_agents_history_speeds=neigh_hist_speed,
            all_other_agents_history_availabilities=neigh_hist_mask,
            all_other_agents_history_availability=neigh_hist_mask,
            all_other_agents_curr_speed=neigh_curr_speed,
            all_other_agents_future_positions=neigh_fut_pos,
            all_other_agents_future_yaws=neigh_fut_yaw,
            all_other_agents_future_availability=neigh_fut_mask,
            all_other_agents_types=neigh_types,
            all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
            all_other_agents_history_extents=neigh_hist_extents * extent_scale,
        )
        batch = dict(batch)
        batch.update(d)
        if overwrite_nan:
            for k,v in batch.items():
                if isinstance(v,torch.Tensor):
                    batch[k]=v.nan_to_num(0)

        agent_hist = torch.cat([batch["history_positions"], 
                                torch.cos(batch["history_yaws"]),
                                torch.sin(batch["history_yaws"]),
                                batch["history_speeds"].unsqueeze(-1).expand_as(batch["history_yaws"]), 
                                batch["extent"].unsqueeze(1).repeat(1, batch["history_yaws"].shape[1], 1)[..., :2],
                                batch["history_availabilities"].unsqueeze(-1).expand_as(batch["history_yaws"])], dim=-1)
        
        neigh_hist = torch.cat([batch["all_other_agents_history_positions"], 
                                torch.cos(batch["all_other_agents_history_yaws"]),
                                torch.sin(batch["all_other_agents_history_yaws"]),
                                batch["all_other_agents_history_speeds"].unsqueeze(-1).expand_as(batch["all_other_agents_history_yaws"]), 
                                batch["all_other_agents_history_extents"][..., :2],
                                batch["all_other_agents_history_availabilities"].unsqueeze(-1).expand_as(batch["all_other_agents_history_yaws"])],
                                dim=-1).transpose(1, 2)

        map_polyline = batch["extras"]["closest_lane_point"]
        nan_inds = torch.sum(torch.isnan(map_polyline).float(), dim=-1) > 0
        avail = torch.ones_like(map_polyline[..., 0])
        avail[nan_inds] = 0
        map_polyline = torch.nan_to_num(map_polyline, nan=0.0)
        map_polyline = torch.cat([map_polyline, avail.unsqueeze(-1)], dim=-1)

        return agent_hist, neigh_hist, map_polyline

    def on_validation_epoch_end(self):
        ade_losses = torch.cat(self.epoch_ade_losses)
        fde_losses = torch.cat(self.epoch_fde_losses)
        mode_probs = torch.cat(self.epoch_mode_probs)
        train_minade_10 = min_xde_K(ade_losses, mode_probs, K=10)
        train_minade_5 = min_xde_K(ade_losses, mode_probs, K=5)
        train_minade_1 = min_xde_K(ade_losses, mode_probs, K=1)
        train_minfde_10 = min_xde_K(fde_losses, mode_probs, K=10)
        train_minfde_5 = min_xde_K(fde_losses, mode_probs, K=5)
        train_minfde_1 = min_xde_K(fde_losses, mode_probs, K=1)

        self.log('metrics/minADE@10', train_minade_10[0])
        self.log('metrics/minADE@5', train_minade_5[0])
        self.log('metrics/minADE@1', train_minade_1[0])
        self.log('metrics/minFDE@10', train_minfde_10[0])
        self.log('metrics/minFDE@5', train_minfde_5[0])
        self.log('metrics/minFDE@1', train_minfde_1[0])
        self.epoch_ade_losses = []
        self.epoch_fde_losses = []
        self.epoch_mode_probs = []

    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1)
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1)

        return ade_losses, fde_losses
    
    def convert_action_to_state_and_action(self, x_out, curr_states):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            x_out: (B, (M), T, 2). scaled action trajectory
            curr_states: (B, (M), 4). current state
        Output:
            x_out: (B, (M), T, 6). scaled state+action trajectory
        '''
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel',
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)

        return x_out_all

    def get_action(self, obs_dict,
                    num_action_samples=1,
                    **kwargs):
        self.eval()
        
        global_t = kwargs['step_index']

        # if global_t == 0:
        #     self.choose = None
        self.stationary_mask = obs_dict['curr_speed'] < 0.5
        ego_from_agent = obs_dict["agent_from_world"][0, 0] @ obs_dict["world_from_agent"][0, 1:]
        pos, yaw = transform_agents_to_world(obs_dict["history_positions"][:, 1:].transpose(0, 1),
                                             obs_dict["history_yaws"][:, 1:].transpose(0, 1),
                                             ego_from_agent)
        
        obs_dict["history_positions"][:, 1:] = pos.transpose(0, 1)
        obs_dict["history_yaws"][:, 1:] = yaw.transpose(0, 1)

        is_nan = torch.isnan(obs_dict["history_positions"])
        obs_dict["history_positions"][torch.isnan(obs_dict["history_positions"])] = 0.0
        obs_dict["history_yaws"][torch.isnan(obs_dict["history_yaws"])] = 0.0
        obs_dict["history_speeds"][torch.isnan(obs_dict["history_speeds"])] = 0.0
        temp_extent = obs_dict["extent"].unsqueeze(2).repeat(1, 1,obs_dict["history_yaws"].shape[2], 1)[..., :2]
        temp_extent[is_nan] = 0.0

        agent_hist = torch.cat([obs_dict["history_positions"], 
                                torch.cos(obs_dict["history_yaws"]),
                                torch.sin(obs_dict["history_yaws"]),
                                obs_dict["history_speeds"].unsqueeze(-1).expand_as(obs_dict["history_yaws"]), 
                                temp_extent,
                                obs_dict["history_availabilities"].unsqueeze(-1).expand_as(obs_dict["history_yaws"])], dim=-1)[0]
        # agent_hist[~obs_dict["history_availabilities"].unsqueeze(-1).expand_as(obs_dict["history_yaws"])[0, :, :, 0]] = 0.0
        
        map_polyline = obs_dict["extras"]["closest_lane_point"][0, 0]

        nan_inds = torch.sum(torch.isnan(map_polyline).float(), dim=-1) > 0
        avail = torch.ones_like(map_polyline[..., 0])
        avail[nan_inds] = 0
        map_polyline = torch.nan_to_num(map_polyline, nan=0.0)
        map_polyline = torch.cat([map_polyline, avail.unsqueeze(-1)], dim=-1)

        neigh_hist = agent_hist[1:]
        # neigh_hist[self.stationary_mask[0, 1:]] = neigh_hist[self.stationary_mask[0, 1:]] * 0.0
        agent_hist = agent_hist[0]

        pred_obs, mode_probs = self(agent_hist.unsqueeze(0), neigh_hist.unsqueeze(0).transpose(1, 2), map_polyline.unsqueeze(0))
        # if self.choose is not None:
        #     agent_from_agent = obs_dict["agent_from_world"][0, 0] @ self.rotation
        #     self.choose = (transform_points_tensor(self.choose[0], agent_from_agent), self.choose[1])
        #     pred_obs_temp = pred_obs[..., :2].transpose(1, 2)
        #     if torch.norm(pred_obs_temp[0, 0, 0] - pred_obs_temp[0, 0, -1]) < 0.5:
        #         pred_obs_temp = torch.zeros_like(pred_obs_temp)
        #     pred_obs_yaw = yaw_from_pos(pred_obs_temp, dt=0.1)
        #     pred_obs_yaw = torch.cat([pred_obs_yaw, pred_obs_yaw[:, :, -1].unsqueeze(-2)], dim=-2)
        #     # t = 47
        #     # exp_weight = torch.tensor([0.95 ** i for i in range(t)]).to(pred_obs_temp.device)
        #     # index = (torch.norm((self.choose[0][:, 5:] - self.choose[0][:, 4:5]).unsqueeze(0) - pred_obs_temp[:, :, :-5], dim=-1) * exp_weight).sum(-1).argmin()
        #     index = (torch.norm((self.choose[0][:, 5:] - self.choose[0][:, 4:5]).unsqueeze(0) - pred_obs_temp[:, :, :-5], dim=-1)).sum(-1).argmin()
        #     self.choose = (pred_obs_temp[index], pred_obs_yaw[index])
        #     if torch.norm(self.choose[0][0, 0] - self.choose[0][0, -1]) < 0.5:
        #         self.choose = (torch.zeros_like(self.choose[0]), torch.zeros_like(self.choose[1]))
        # self.rotation = obs_dict["world_from_agent"][0, 0]
            

        mode_index = torch.sort(mode_probs, descending=True)[1][0]
        mode_topk = mode_index[:num_action_samples]
        pred_obs_topk_pos = pred_obs[mode_topk][..., :2].transpose(1, 2)
        if torch.norm(pred_obs_topk_pos[0, 0, 0] - pred_obs_topk_pos[0, 0, -1]) < 0.5:
            pred_obs_topk_pos = torch.zeros_like(pred_obs_topk_pos)
        # pred_obs_topk_yaw = yaw_from_pos(pred_obs_topk_pos, dt=0.1)
        # pred_obs_topk_yaw = torch.cat([pred_obs_topk_yaw, pred_obs_topk_yaw[:, :, -1].unsqueeze(-2)], dim=-2)
        temp2calyaw = torch.cat([torch.zeros_like(pred_obs_topk_pos[:, :, :1]), pred_obs_topk_pos], dim=2)
        deltapos = temp2calyaw[:, :, 1:] - temp2calyaw[:, :, :-1]
        pred_obs_topk_yaw = torch.arctan2(deltapos[:, :, :, 1], deltapos[:, :, :, 0]).unsqueeze(-1)

        # if self.choose is None:
        self.choose = (pred_obs_topk_pos[0], pred_obs_topk_yaw[0])

        action_pred_position = torch.cat([self.choose[0], obs_dict['action_traj_positions'][0, 1:, global_t]], dim=0)
        action_pred_yaw = torch.cat([self.choose[1], obs_dict['action_traj_yaws'][0, 1:, global_t]], dim=0)

        pred_position = torch.cat([pred_obs_topk_pos.transpose(0, 1), obs_dict['action_sample_positions'][0, 1:, global_t]], dim=0).transpose(0, 1)
        pred_yaw = torch.cat([pred_obs_topk_yaw.transpose(0, 1), obs_dict['action_sample_yaws'][0, 1:, global_t]], dim=0).transpose(0, 1)
        pred_traj = torch.cat([pred_position, pred_yaw], dim=-1)

        temp = convert_state_to_state_and_action(torch.cat([action_pred_position[:1], action_pred_yaw[:1]], dim=-1).unsqueeze(1), obs_dict["curr_speed"][:, :1], self.dt)
        action_pred_position[:1] = temp[0, :, :, :2]
        action_pred_yaw[:1] = temp[0, :, :, 3:4]


        info = dict(
            action_samples=Action(
                positions=pred_position.unsqueeze(0), # (B, N, M, T, 2)
                yaws=pred_yaw.unsqueeze(0)
            ).to_dict(),
            trajectories=pred_traj.unsqueeze(0),
        )
        action = Action(
            positions=action_pred_position.unsqueeze(0), # (B, M, T, 2)
            yaws=action_pred_yaw.unsqueeze(0),
        )

        return action, info


    
    def configure_optimizers(self):
        if self.fine_tune:
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.00005)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5, verbose=True)

        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        grad_clip_norm = 5.0
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

def min_xde_K(xdes, probs, K):
    # 获取每行前 K 个最大值的索引
    _, best_ks = probs.topk(K, dim=1, largest=True)
    # 创建行索引
    dummy_rows = torch.arange(xdes.size(0)).unsqueeze(1)
    # 提取相应的 xdes 值
    new_xdes = xdes[dummy_rows, best_ks]
    # 对提取的值进行排序并计算均值
    sorted_xdes = new_xdes.sort(dim=1).values
    mean_xdes = torch.nanmean(sorted_xdes, dim=0)
    
    return mean_xdes

def get_BVG_distributions(pred):
    B = pred.size(0)
    T = pred.size(1)
    mu_x = pred[:, :, 0].unsqueeze(2)
    mu_y = pred[:, :, 1].unsqueeze(2)
    sigma_x = pred[:, :, 2]
    sigma_y = pred[:, :, 3]
    rho = pred[:, :, 4]

    cov = torch.zeros((B, T, 2, 2)).to(pred.device)
    cov[:, :, 0, 0] = sigma_x ** 2
    cov[:, :, 1, 1] = sigma_y ** 2
    cov[:, :, 0, 1] = rho * sigma_x * sigma_y
    cov[:, :, 1, 0] = rho * sigma_x * sigma_y

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist(pred):
    return Laplace(pred[:, :, :2], pred[:, :, 2:4])


def nll_pytorch_dist(pred, data, rtn_loss=True):
    # biv_gauss_dist = get_BVG_distributions(pred)
    biv_gauss_dist = get_Laplace_dist(pred)
    if rtn_loss:
        # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(-1).sum(1)  # Laplace
    else:
        # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(dim=(1, 2))  # Laplace


def nll_loss_multimodes(pred, data, modes_pred, entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True):
    """NLL loss multimodes for training. MFP Loss function
    Args:
      pred: [K, T, B, 5]
      data: [B, T, 5]
      modes_pred: [B, K], prior prob over modes
      noise is optional
    """
    modes = len(pred)
    nSteps, batch_sz, dim = pred[0].shape

    # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=False)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = modes_pred.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(data.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=True) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions(pred[kk]).entropy())
    entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
    entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(modes_pred), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde(pred, data)
    else:
        adefde_loss = torch.tensor(0.0).to(data.device)

    return loss, kl_loss, post_entropy, adefde_loss


def l2_loss_fde(pred, data):
    fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    # fde_loss5 = torch.norm((pred[:, 4, :, :2].transpose(0, 1) - data[:, 4, :2].unsqueeze(1)), 2, dim=-1)
    # ade_loss5 = torch.norm((pred[:, :5, :, :2].transpose(1, 2) - data[:, :5, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    ade_loss = torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    # loss, min_inds = (fde_loss + ade_loss + 100 * fde_loss5 + 100 * ade_loss5).min(dim=1)
    loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    return 100.0 * loss.mean()