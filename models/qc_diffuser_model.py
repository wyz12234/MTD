import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import tbsim.utils.geometry_utils as GeoUtils
from torch.nn.utils.rnn import pad_sequence
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action
from tbsim.models.diffuser_helpers import EMA
from tbsim.models.scenediffuser import SceneDiffuserModel
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates, add_scene_dim_to_agent_data, get_stationary_mask
from trajdata.utils.arr_utils import angle_wrap
from utils.diffuser_utils import cosine_beta_schedule, convert_state_to_state_and_action, extract, unicyle_forward_dynamics, Progress
from models.qc_diffuser_encoder import QCDiffuserEncoder
from models.qc_diffuser_decoder import QCDiffuserDecoder
import tbsim.dynamics as dynamics
from tbsim.utils.guidance_loss import verify_constraint_config, apply_constraints, DiffuserGuidance, PerturbationGuidance
from copy import deepcopy
from utils.guidance_utils import extract_data_batch_for_guidance
from tbsim.utils.guidance_loss import SceneMapCollisionLoss, SceneAgentCollisionLoss
from collections import OrderedDict

class QueryDiffuserTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name, do_log=True, guidance_config=None, \
                  constraint_config=None, n_timesteps=100, dt=0.1, horizon=52, action_dim=2, observation_dim=4, output_dim=2):
        """
        Creates networks and places them into @self.nets.
        """
        super(QueryDiffuserTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        self.n_timesteps= int(n_timesteps)
        self.dt = dt
        self.horizon = horizon
        self.default_chosen_inds = [0, 1, 2, 3, 4, 5] # [x, y, vel, yaw, acc, yawvel]
        self.transition_dim = observation_dim + action_dim
        self.stride = 1

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate # agent_centric ?
        # used only when data_centric == 'scene' and coordinate == 'agent'
        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist # inf
        # to help control stationary agent's behavior
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary # current speed
        self.moving_speed_th = algo_config.moving_speed_th # 0.5
        self.stationary_mask = None
        self.stationary_mask_expand = None

        # "Observations" are inputs to diffuser that are not outputs
        # "Actions" are inputs and outputs
        # "transition" dim = observation + action this is the input at each step of denoising
        # "output" is final output of the entired denoising process.

        # TBD: extract these and modify the later logics
        if algo_config.diffuser_input_mode == 'state':
            observation_dim = 0
            action_dim = 3 # x, y, yaw
            output_dim = 3 # x, y, yaw
        elif algo_config.diffuser_input_mode == 'action':
            observation_dim = 0
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action': # default
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action_no_dyn':
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 6 # x, y, vel, yaw, acc, yawvel
        else:
            raise
        
        print('registered_name', registered_name)
        diffuser_norm_info = ([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4])
        agent_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_fut_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        if 'nusc' in registered_name: # 导入正则化参数
            diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nusc_norm_info:
                neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nusc_norm_info:
                neighbor_fut_norm_info = algo_config.nusc_norm_info['neighbor_fut']
        elif 'nuplan' in registered_name:
            diffuser_norm_info = algo_config.nuplan_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nuplan_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nuplan_norm_info:
                neighbor_hist_norm_info = algo_config.nuplan_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nuplan_norm_info:
                neighbor_fut_norm_info = algo_config.nuplan_norm_info['neighbor_fut']
        else:
            raise

        self.diffuser_norm_info = diffuser_norm_info
        self.agent_hist_norm_info = agent_hist_norm_info
        self.neighbor_hist_norm_info = neighbor_hist_norm_info
        self.neighbor_fut_norm_info = neighbor_fut_norm_info
        self.add_coeffs = np.array(self.diffuser_norm_info[0]).astype(np.float32)
        self.div_coeffs = np.array(self.diffuser_norm_info[1]).astype(np.float32)


        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill # 0.5

        self.use_rasterized_map = algo_config.rasterized_map # False
        self.use_rasterized_hist = algo_config.rasterized_history # False

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print('DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (self.cond_drop_neighbor_p))

        if algo_config["dynamics"]["type"] == "Unicycle":
            self.dyn = dynamics.Unicycle(
                    "dynamics",
                    max_steer=algo_config["dynamics"]["max_steer"],
                    max_yawvel=algo_config["dynamics"]["max_yawvel"],
                    acce_bound=algo_config["dynamics"]["acce_bound"]
                )
        else:
            raise NotImplementedError
        
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        

        self.encoder = QCDiffuserEncoder()
        self.decoder = QCDiffuserDecoder()
        self.loss = nn.MSELoss()
            
        self.current_constraints = None
        self.guidance_optimization_params = None
        self.apply_guidance_intermediate = None
        self.apply_guidance_output = None
        self.apply_guidance_output = None
        self.final_step_opt_params = None
        self.adversary = None

        self.transform_params = {'scaled_input':True, 'scaled_output':True}
        self.current_perturbation_guidance = PerturbationGuidance(self.transform, self.transform_params, self.scale_traj, self.descale_traj)

        self.cur_train_step = 0

        self.map_collision_loss = SceneMapCollisionLoss()
        self.agent_collision_loss = SceneAgentCollisionLoss(num_disks=2)

    @property
    def checkpoint_monitor_keys(self):
            return {"valLoss": "val/loss"}

    def forward(self, obs_dict, num_sample=1, global_t=0, apply_guidance=False):
        if self.disable_control_on_stationary and global_t == 0:
            self.stationary_mask = get_stationary_mask(obs_dict, self.disable_control_on_stationary, self.moving_speed_th)
            self.stationary_mask[:, 0] = False
            B, M = self.stationary_mask.shape
            # (B, M) -> (B, N, M) -> (B*N, M)
            self.stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, num_sample, M).reshape(B * num_sample, M)

        auxiliaries_info = self.get_auxiliaries_info(obs_dict)

        B, M = obs_dict["history_positions"].shape[:2]
        shape = (B, num_sample, M, self.horizon, self.transition_dim)
        output = self.p_sample_loop(shape, obs_dict, 
                                    num_sample=num_sample, 
                                    auxiliaries_info=auxiliaries_info,
                                    apply_guidance=apply_guidance)
        if apply_guidance:
            guide_loss = output["guide_loss"]
        else:
            guide_loss = None
        traj = self.descale_traj(output["traj"])
        position = traj[..., :2]
        yaw = traj[..., 3].unsqueeze(-1)
        out_dict = {
            "position": position,
            "yaw": yaw,
            "traj": traj,
            "guide_loss": guide_loss
        }
        return out_dict


    def _compute_metrics(self, pred_batch, batch):
        metrics = {}
        predictions = pred_batch["prediction"]
        sample_preds = predictions["position"]
        B, N, M, T, _ = sample_preds.shape
        # (B, N, M, T, 2) -> (B, M, N, T, 2) -> (B*M, N, T, 2)
        sample_preds = TensorUtils.to_numpy(sample_preds.permute(0, 2, 1, 3, 4).reshape(B*M, N, T, -1))
        # (B, M, T, 2) -> (B*M, T, 2)
        gt = TensorUtils.to_numpy(batch["target_positions"].reshape(B*M, T, -1))
        # (B, M, T) -> (B*M, T)
        avail = TensorUtils.to_numpy(batch["target_availabilities"].reshape(B*M, T))
        
        # compute ADE & FDE based on trajectory samples
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def on_training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)

        batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
       
        loss = self.compute_loss(batch)
        total_loss = 0

        for k, v in loss.items():
            self.log("train/" + k, v)
            loss_part = v * self.algo_config.loss_weights[k]
            total_loss += loss_part
            
        return total_loss

    
    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)

        batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)

        loss = TensorUtils.detach(self.compute_loss(batch))
        
        auxiliaries_info = self.get_auxiliaries_info(batch)

        self.stationary_mask = None
        self.stationary_mask_expand = None

        B, M = batch["history_positions"].shape[:2]
        shape = (B, self.algo_config.diffuser.num_eval_samples, M, self.horizon, self.transition_dim)
        output = self.p_sample_loop(shape, batch, 
                                    num_sample=self.algo_config.diffuser.num_eval_samples, 
                                    auxiliaries_info=auxiliaries_info)
        
        traj = self.descale_traj(output["traj"])
        position = traj[..., :2]
        yaw = traj[..., 3].unsqueeze(-1)
        out_dict = {
            "prediction":
            {
                "position": position,
                "yaw": yaw,
                "traj": traj
            },
            "curr_states": auxiliaries_info['curr_states']
        }
        metrics = self._compute_metrics(out_dict, batch)

        total_loss = 0
        for k, v in loss.items():
            loss_part = v * self.algo_config.loss_weights[k]
            total_loss += loss_part

        return_dict = {"metrics": metrics,
                       "losses": loss,
                       "loss": total_loss}
        
        for k, v in return_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    self.log("val/" + k + "_" + k2, v2)
            else:
                self.log("val/" + k, v)

        return return_dict
        

    def configure_optimizers(self):
        optim_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.AdamW(optim_params, 
                                lr=self.algo_config["optim_params"]["policy"]["learning_rate"]["initial"])
        return optimizer

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict,
                    num_action_samples=1,
                    apply_guidance=False,
                    **kwargs):
        self.encoder.eval()
        self.decoder.eval()

        self.update_guidance(global_t=kwargs['step_index'])
        
        # visualize rollout batch for debugging
        visualize_agent_batch = False
        ind_to_vis = 0
        if visualize_agent_batch:
            from matplotlib import pyplot as plt
            from tbsim.utils.trajdata_utils import plot_agent_batch_dict
            import os
            if 'agent_name' not in obs_dict:
                obs_dict['agent_name'] = [[str(i) for i in range(obs_dict['target_positions'].shape[1])]]
            ax = plot_agent_batch_dict(obs_dict, batch_idx=ind_to_vis, legend=False, show=False, close=False)
            os.makedirs('nusc_results/agent_batch_vec_map_vis', exist_ok=True)
            plt.savefig('nusc_results/agent_batch_vec_map_vis/agent_batch_'+str(kwargs['step_index'])+'.png')
            plt.close()

        pred = self(obs_dict, num_sample=num_action_samples, global_t=kwargs['step_index'],
                    apply_guidance=apply_guidance)

        # [B, N, M, T, 2]
        B, N, M, _, _ = pred["position"].shape

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((M), dtype=torch.long, device=pred["position"].device)
        if pred['guide_loss'] is not None:
            losses = pred.pop("guide_loss", None)
            choose_loss = None
            for k, v in losses.items():
                if choose_loss is None:
                    choose_loss = v
                else:
                    choose_loss += v
            act_idx = torch.argmin(choose_loss, dim=1)

        def map_act_idx(x):
            # Assume B == 1 during generation. TBD: need to change this to support general batchsize
            if len(x.shape) == 4:
                # [N, T, M1, M2] -> [M1, N, T, M2]
                x = x.permute(2,0,1,3)
            elif len(x.shape) == 5:
                # [B, N, M, T, 2] -> [N, M, T, 2] -> [M, N, T, 2]
                x = x[0].permute(1,0,2,3)
            elif len(x.shape) == 6: # for "diffusion_steps"
                x = x[0].permute(1,0,2,3,4)
            else:
                raise NotImplementedError
            # [M, N, T, 2] -> [M, T, 2]
            x = x[torch.arange(M), act_idx]
            # [M, T, 2] -> [B, M, T, 2]
            x = x.unsqueeze(0)
            return x

        pred_position = pred["position"]
        pred_yaw = pred["yaw"]
        pred_traj = pred["traj"]

        action_pred_position = map_act_idx(pred_position)
        action_pred_yaw = map_act_idx(pred_yaw)
        

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N, M)
            
            pred_position[stationary_mask_expand] = 0
            pred_yaw[stationary_mask_expand] = 0
            pred_traj[stationary_mask_expand] = 0

            action_pred_position[self.stationary_mask] = 0
            action_pred_yaw[self.stationary_mask] = 0

        info = dict(
            action_samples=Action(
                positions=pred_position, # (B, N, M, T, 2)
                yaws=pred_yaw
            ).to_dict(),

            trajectories=pred_traj,
            act_idx=act_idx,
            dyn=self.dyn,
        )
        action = Action(
            positions=action_pred_position, # (B, M, T, 2)
            yaws=action_pred_yaw
        )
        return action, info
    
    def update_guidance(self, **kwargs):
        if self.current_perturbation_guidance.current_guidance is not None:
            self.current_perturbation_guidance.update(**kwargs)

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        if len(guidance_config) > 0:
            print('Instantiating test-time guidance with configs:')
            print(guidance_config)
            self.current_perturbation_guidance.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        self.current_perturbation_guidance.clear_guidance()

    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        if constraint_config is not None and len(constraint_config) > 0:
            print('Instantiating test-time constraints with config:')
            print(constraint_config)
            self.current_constraints = constraint_config

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        self.guidance_optimization_params = guidance_optimization_params
    
    def set_diffusion_specific_params(self, diffusion_specific_params):
        self.apply_guidance_intermediate = diffusion_specific_params['apply_guidance_intermediate']
        self.apply_guidance_output = diffusion_specific_params['apply_guidance_output']
        self.final_step_opt_params = diffusion_specific_params['final_step_opt_params']
        self.stride = diffusion_specific_params['stride']

    def transform(self, x_guidance, batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states = batch_utils().get_current_states(batch, dyn_type=self.dyn.type())
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance, expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all

    def compute_loss(self, batch):
        auxiliaries_info = self.get_auxiliaries_info(batch)
        target_traj = self.get_state_and_action_from_data_batch(batch)
        x = self.scale_traj(target_traj) # 标准化后的目标轨迹
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # diffusion step
        noise_init = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise_init) # 加噪

        # 利用动作重构状态和动作
        x_action_noisy = x_noisy[..., [4, 5]]
        x_noisy = self.convert_action_to_state_and_action(x_action_noisy, auxiliaries_info['curr_states'])

        # 原始轨迹
        # x_orgin = x.detach().clone()

        auxiliaries_info = self.get_neighbor_future_relative_states(auxiliaries_info, batch) # 使用恒定速度模型给邻居添加未来信息
        encoder = self.encoder(x_noisy, auxiliaries_info)
        noise = self.decoder(auxiliaries_info, encoder, t)

        # noise_hist = noise[..., :-self.horizon, :]
        # noise_fut = noise[..., -self.horizon:, :]
        noise_fut = noise

        x_recon_action = self.predict_start_from_noise(x_action_noisy, t=t, noise=noise_fut)
        x_recon = self.convert_action_to_state_and_action(x_recon_action, auxiliaries_info['curr_states'])

        x_recon_selected = x_recon * batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)
        x_start_selected = x * batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)

        diffusion_loss = self.loss(x_recon_selected, x_start_selected)

        offroad_loss = self.map_collision_loss(x_recon_selected, batch, batch['target_availabilities']).mean()

        collision_loss = self.agent_collision_loss(x_recon_selected, batch, batch['target_availabilities']).mean()

        loss = OrderedDict(diffusion_loss=diffusion_loss, offroad_loss=offroad_loss, collision_loss=collision_loss)



        return loss

    def get_auxiliaries_info(self, batch):
        agents_hist = self.prepare_scene_agent_hist(batch["history_positions"], batch["history_yaws"], \
                                                     batch["history_speeds"], batch["extent"], batch["history_availabilities"], self.agent_hist_norm_info)
        neighbor_hist = self.get_neighbor_history_relative_states(batch, self.neighbor_hist_norm_info)
        
        # vectorized map (B, M, S_seg, S_p, K_map)
        map_feat = batch["extras"]["closest_lane_point"]

        curr_states = batch_utils().get_current_states(batch, dyn_type=self.dyn.type())

        auxiliaries_info = {
            "curr_states": curr_states,
            "agents_hist": agents_hist,
            "neighbor_hist": neighbor_hist,
            "map_feat": map_feat,
        }
        return auxiliaries_info

    def prepare_scene_agent_hist(self, pos, yaw, speed, extent, avail, norm_info, scale=True):
        '''
        Input:
        - pos : (B, M, T, 2)
        - yaw : (B, M, T, 1)
        - speed : (B, M, T)
        - extent: (B, M, 3)
        - avail: (B, M, T)
        - norm_info: [2, 5]
        Output:
        - hist_in: [B, M, (Q), T, 8] (x,y,cos,sin,v,l,w,avail)
        '''
        B, M, T, _ = pos.shape

        hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # (B, M, (Q), T, 2)
        lw = extent[..., :2].unsqueeze(-2).expand(pos.shape) # (B, M, (Q), T, 2)
        add_coeffs = torch.tensor(norm_info[0]).to(pos.device)
        div_coeffs = torch.tensor(norm_info[1]).to(pos.device)
        add_coeffs_expand = add_coeffs[None, None, None, :]
        div_coeffs_expand = div_coeffs[None, None, None, :]

        if scale:
            pos = (pos + add_coeffs_expand[...,:2]) / div_coeffs_expand[...,:2]
            speed = (speed.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
            lw = (lw + add_coeffs_expand[...,3:]) / div_coeffs_expand[...,3:]
        else:
            speed = speed.unsqueeze(-1)
        
        hist_in = torch.cat([pos, hvec, speed, lw, avail.unsqueeze(-1)], dim=-1)

        hist_in[~avail] = 0.0
        
        return hist_in
    
    def get_neighbor_history_relative_states(self, batch, norm_info, scale=True):
        '''
        get the neighbor history relative states (only need once per data_batch). We do this because fields like all_other_agents_history_positions in data_batch may not include all the agents controlled and may include agents not controlled.

        - output neighbor_hist: (B, M, M, T_hist, K_vehicle)
        - output neighbor_hist_non_cond: (B, M, M, T_hist, K_vehicle)
        '''
        B, M, T, _ = batch['history_positions'].shape

        all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents = \
            self.get_neighbor_relative_states(batch['history_positions'], batch['history_speeds'], batch['history_yaws'], \
                                              batch['agent_from_world'], batch['world_from_agent'], batch['yaw'], batch["extent"])

        # (B, M, T_hist) -> (B, 1, M, T_hist) -> (B, M, M, T_hist)
        all_other_agents_history_availabilities = batch["history_availabilities"].unsqueeze(1).repeat(1, M, 1, 1)

        # convert yaw to heading vec
        hvec = torch.cat([torch.cos(all_other_agents_history_yaws), torch.sin(all_other_agents_history_yaws)], dim=-1) # (B, M, M, T_hist, 2)
        # only need length, width for pred
        lw = all_other_agents_extents[..., :2].unsqueeze(-2).expand(all_other_agents_history_positions.shape) # (B, M, M, T_hist, 2)
        # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
        pos_self = torch.diagonal(all_other_agents_history_positions, \
                                  dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(all_other_agents_history_positions)
        pos_diff = torch.abs(all_other_agents_history_positions.detach().clone() - pos_self)
        # (B, M, M, T, 2) -> (B, M, M, T, 1)
        rel_d = torch.norm(pos_diff, dim=-1, keepdim=True)

        # (B, M, M, T, 2) -> (B, M, M, T, 1)
        lw_avg_half = (torch.mean(lw, dim=-1) / 2).unsqueeze(-1)
        # (B, M, M, T, 2) -> (B, M, T, 1)
        ego_lw_avg_half = lw_avg_half[...,torch.arange(M), torch.arange(M), :, :]
        # (B, M, M, T, 1)
        lw_avg_half_sum = lw_avg_half + ego_lw_avg_half.unsqueeze(2).expand_as(lw_avg_half)
        # (B, M, M, T, 1)
        rel_d_lw = rel_d - lw_avg_half_sum
        d_th = 20
        t_to_col_th = 20
        # normalize rel_d and rel_d_lw
        rel_d = torch.clip(rel_d, min=0, max=d_th)
        rel_d = (d_th - rel_d) / d_th
        rel_d_lw = torch.clip(rel_d_lw, min=0, max=d_th)
        rel_d_lw = (d_th - rel_d_lw) / d_th

        # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
        ego_vx = torch.diagonal(all_other_agents_history_speeds, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
        # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
        ego_lw = torch.diagonal(lw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand(B, M, M, T, 2).clone()
        ego_vx[torch.isnan(ego_vx)] = 0.0

        # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
        yaw_self = torch.diagonal(all_other_agents_history_yaws, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(all_other_agents_history_yaws)
        hvec_self = torch.cat([torch.cos(yaw_self), torch.sin(yaw_self)], dim=-1)
        vx = ego_vx * hvec_self[...,0] - all_other_agents_history_speeds * hvec[...,0]
        vy = ego_vx * hvec_self[...,1] - all_other_agents_history_speeds * hvec[...,1]

        x_dist = pos_diff[...,0] - (ego_lw[...,0]/2) - (lw[...,0]/2)
        y_dist = pos_diff[...,1] - (ego_lw[...,1]/2) - (lw[...,1]/2)
        x_t_to_col = x_dist / vx
        y_t_to_col = y_dist / vy
        # if collision has not happened and moving in opposite direction, set t_to_col to t_to_col_th
        x_t_to_col[(x_dist>0) & (x_t_to_col<0)] = t_to_col_th
        y_t_to_col[(y_dist>0) & (y_t_to_col<0)] = t_to_col_th
        # if collision already happened, set t_to_col to 0
        x_t_to_col[x_dist<0] = 0
        y_t_to_col[y_dist<0] = 0
        # both directions need to be met for collision to happen
        rel_t_to_col = torch.max(torch.cat([x_t_to_col.unsqueeze(-1), y_t_to_col.unsqueeze(-1)], dim=-1), dim=-1)[0]
        rel_t_to_col = torch.clip(rel_t_to_col, min=0, max=t_to_col_th)
        # normalize rel_t_to_col
        rel_t_to_col = (t_to_col_th - rel_t_to_col.unsqueeze(-1)) / t_to_col_th

        # normalize everything
        #  note: don't normalize hvec since already unit vector
        add_coeffs = torch.tensor(norm_info[0]).to(all_other_agents_history_positions.device)
        div_coeffs = torch.tensor(norm_info[1]).to(all_other_agents_history_positions.device)

        add_coeffs_expand = add_coeffs[None, None, None, None, :]
        div_coeffs_expand = div_coeffs[None, None, None, None, :]

        if scale:
            pos = (all_other_agents_history_positions + add_coeffs_expand[...,:2]) / div_coeffs_expand[...,:2]
            speed = (all_other_agents_history_speeds.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
            lw = (lw + add_coeffs_expand[...,3:]) / div_coeffs_expand[...,3:]
        else:
            speed = all_other_agents_history_speeds.unsqueeze(-1)
        
        speed = speed.squeeze(-1)
        # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M) -> (B, M, 1, 1) -> (B, M, M, T)
        ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1)[...,0].unsqueeze(-1).unsqueeze(-1).expand(B, M, M, T).clone()
        ego_vx[torch.isnan(ego_vx)] = 0.0
        vx = speed * hvec[...,0] - ego_vx
        vy = speed * hvec[...,1]

        vvec = torch.cat([vx.unsqueeze(-1), vy.unsqueeze(-1)], dim=-1) # (B, M, M, T, 2)

        # (B, M1, M2, T) -> (B, M2, M1, T)
        avail_perm = all_other_agents_history_availabilities.permute(0, 2, 1, 3)
        avail = all_other_agents_history_availabilities * avail_perm

        hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, avail.unsqueeze(-1)], dim=-1)
        # zero out values we don't have data for
        hist_in[~avail] = 0.0

        return hist_in
    
    def get_neighbor_future_relative_states(self, auxiliaries_info, batch):
        B, M, T, _ = batch['target_positions'].shape

        # (B*N, M, M, T_hist, K_neigh) -> (B*N, M, M, K_neigh) -> (B*N, M, M, 1, K_neigh) -> (B*N, M, M, T_fut, K_neigh) (x,y,cos,sin,speed,l,w,avail) or (x,y,cos,sin,vx,vy,l,w,avail)
        neighbor_fut = auxiliaries_info['neighbor_hist'][..., -1, :].unsqueeze(-2).repeat(1, 1, 1, T, 1)
        # Get a time weight with shape (T_fut) from 0.1 to 0.1*T_fut
        time = torch.arange(1, T+1, dtype=torch.float32, device=neighbor_fut.device) * self.dt
        # (T_fut) -> (1, 1, 1, T_fut)
        time = time.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        neighbor_fut[...,:2] = neighbor_fut[...,:2] + neighbor_fut[...,4:6] * time

        auxiliaries_info['neighbor_fut'] = neighbor_fut
        auxiliaries_info['neighbor_feat'] = torch.cat([auxiliaries_info['neighbor_hist'], auxiliaries_info['neighbor_fut']], dim=-2)

        return auxiliaries_info

    def get_neighbor_relative_states(self, relative_positions, relative_speeds, relative_yaws, data_batch_agent_from_world, data_batch_world_from_agent, data_batch_yaw, data_batch_extent):
        BN, M, _, _ = relative_positions.shape
        B = data_batch_agent_from_world.shape[0]
        N = int(BN // B)

        # [M, M]
        nb_idx = torch.arange(M).unsqueeze(0).repeat(M, 1)

        all_other_agents_relative_positions_list = []
        all_other_agents_relative_yaws_list = []
        all_other_agents_relative_speeds_list = []
        all_other_agents_extent_list = []

        # get relative states
        for k in range(BN):
            i = int(k // N)
            agent_from_world = data_batch_agent_from_world[i]
            world_from_agent = data_batch_world_from_agent[i]

            all_other_agents_relative_positions_list_sub = []
            all_other_agents_relative_yaws_list_sub = []
            all_other_agents_relative_speeds_list_sub = []
            all_other_agents_extent_list_sub = []

            for j in range(M):
                chosen_neigh_inds = nb_idx[j][nb_idx[j]>=0].tolist()

                # (Q. 3. 3)
                center_from_world = agent_from_world[j]
                world_from_neigh = world_from_agent[chosen_neigh_inds]
                center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

                fut_neigh_pos_b_sub = relative_positions[k][chosen_neigh_inds]
                fut_neigh_yaw_b_sub = relative_yaws[k][chosen_neigh_inds]

                all_other_agents_relative_positions_list_sub.append(GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub,center_from_neigh))
                all_other_agents_relative_yaws_list_sub.append(fut_neigh_yaw_b_sub+data_batch_yaw[i][chosen_neigh_inds][:,None,None]-data_batch_yaw[i][j])
                all_other_agents_relative_speeds_list_sub.append(relative_speeds[k][chosen_neigh_inds])
                all_other_agents_extent_list_sub.append(data_batch_extent[i][chosen_neigh_inds])

            all_other_agents_relative_positions_list.append(pad_sequence(all_other_agents_relative_positions_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_yaws_list.append(pad_sequence(all_other_agents_relative_yaws_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_speeds_list.append(pad_sequence(all_other_agents_relative_speeds_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_extent_list.append(pad_sequence(all_other_agents_extent_list_sub, batch_first=True, padding_value=0))

        max_second_dim = max(a.size(1) for a in all_other_agents_relative_positions_list)

        all_other_agents_relative_positions = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_positions_list], dim=0)
        all_other_agents_relative_yaws = angle_wrap(torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_yaws_list], dim=0))
        all_other_agents_relative_speeds = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_speeds_list], dim=0)
        all_other_agents_extents = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_extent_list], dim=0)

        return all_other_agents_relative_positions, all_other_agents_relative_yaws, all_other_agents_relative_speeds, all_other_agents_extents
    
    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            data_batch: dict
        Output:
            x: (batch_size, [num_agents], num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"][..., :self.horizon, :], data_batch["target_yaws"][..., :self.horizon, :]), dim=-1)
        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
    
    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale the trajectory from original scale to standard normal distribution
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        # TODO make these a buffer so they're put on the device automatically
        # device = target_traj_orig.get_device()
        device = target_traj_orig.device
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale back the trajectory from standard normal distribution to original scale
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        # device = target_traj_orig.get_device()
        device = target_traj_orig.device
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add

        return target_traj
    
    # τk = sqrt(alphas) * τ0 + sqrt(1 - alphas) * ξk, where ξk ~ N(0, I)
    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t, t, noise):
        return noise
    
    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
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

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel',
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        return x_out_all
    
    def p_sample_loop(self, shape, batch, num_sample, auxiliaries_info, apply_guidance=False):
        device = self.betas.device
        B = shape[0]

        # sample from base distribution
        x = torch.randn(shape, device=device) # (B, N, M, T(+T_hist), transition_dim)
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # (B * N, M, T(+T_hist), transition_dim)


        # (B, M, C) -> (B*N, M, C)
        auxiliaries_info = TensorUtils.repeat_by_expand_at(auxiliaries_info, repeats=num_sample, dim=0)
        auxiliaries_info = self.get_neighbor_future_relative_states(auxiliaries_info, batch)

        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]
        if apply_guidance:
            batch_for_guidance = extract_data_batch_for_guidance(batch)
        else:
            batch_for_guidance = None

        for i in steps:
            # (B*N)
            timesteps = torch.full((B * num_sample,), i, device=device, dtype=torch.long)
            x, guide_loss = self.p_sample(x, timesteps, 
                              auxiliaries_info=auxiliaries_info,
                              apply_guidance=apply_guidance,
                              batch_for_guidance=batch_for_guidance,
                              num_sample=num_sample)
        
        if guide_loss and any(guide_loss):
            print('===== GUIDANCE LOSSES ======')
            for k,v in guide_loss.items():
                print('%s: %.012f' % (k, np.nanmean(v.cpu())))

        x = TensorUtils.reshape_dimensions(x, begin_axis=0, end_axis=1, target_dims=(B, num_sample))

        out_dict = {'traj' : x}

        if guide_loss is not None:
            out_dict['guide_loss'] = guide_loss

        return out_dict
    
    def p_sample(self, x, t, auxiliaries_info, apply_guidance, batch_for_guidance, num_sample):
        B = x.shape[0]
        with torch.no_grad():
            model_mean, _, posterior_log_variance, _, _ = self.p_mean_variance(x, t, auxiliaries_info)
        sigma = (posterior_log_variance / 2).exp()

        nonzero_mask = (1 - (t == 0).float()).reshape(B, *((1,) * (len(x.shape) - 1)))

        x_initial = model_mean.clone().detach()
        x_initial.requires_grad_()

        guide_loss = None
        x_guidance = None

        if apply_guidance and self.guidance_optimization_params is not None:
            if t[0] == 0: # 最后一步是否优化，以及最后一步的参数
                apply_guidance = self.apply_guidance_output
                if apply_guidance:
                    opt_params = deepcopy(self.final_step_opt_params)
            else:
                apply_guidance = self.apply_guidance_intermediate
                if apply_guidance:
                    perturb_th = self.guidance_optimization_params['perturb_th']
                    apply_guidance_output = self.apply_guidance_output
                    lr = self.guidance_optimization_params['lr']

                    if perturb_th is None:
                        if t[0] == 0 and not apply_guidance_output:
                            perturb_th = nonzero_mask * sigma
                        else:
                            perturb_th = sigma
                    
                    opt_params = deepcopy(self.guidance_optimization_params)
                    opt_params['lr'] = lr
                    opt_params['perturb_th'] = perturb_th

            if apply_guidance:
                x_guidance, guide_loss = self.current_perturbation_guidance.perturb(x_initial, batch_for_guidance, opt_params,
                                                                                         num_samp=num_sample)
                # batch_for_guidance['x_baseline'] = None
        if x_guidance is None:
            if self.current_perturbation_guidance.current_guidance is not None:
                _, guide_loss = self.current_perturbation_guidance.compute_guidance_loss(x_initial, batch_for_guidance, num_sample)
                self.adversary = batch_for_guidance.get('adversary_mask', None)
            x_guidance = x_initial

        noise = torch.randn_like(x_guidance) * sigma * nonzero_mask
        x_out = x_guidance + noise
        x_out = self.convert_action_to_state_and_action(x_out, auxiliaries_info['curr_states'])

        return x_out, guide_loss

    
    def p_mean_variance(self, x, t, auxiliaries_info):
        encoder = self.encoder(x, auxiliaries_info)
        noise = self.decoder(auxiliaries_info, encoder, t)

        x_action_temp = x[..., 4:].detach().clone()

        x_recon = self.predict_start_from_noise(x_t=x_action_temp, t=t, noise=noise)

        inds = [4, 5] # acc, yaw_rate
        x_recon_stationary = x_recon[self.stationary_mask_expand]
        x_recon_stationary = self.descale_traj(x_recon_stationary, inds)
        x_recon_stationary[...] = 0
        x_recon_stationary = self.scale_traj(x_recon_stationary, inds)
        x_recon[self.stationary_mask_expand] = x_recon_stationary

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_action_temp, t=t)

        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x_action_temp, t), noise
        


            
