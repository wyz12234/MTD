import numpy as np
from copy import deepcopy

import numpy as np

from tbsim.configs.config import Dict
from tbsim.configs.eval_config import EvaluationConfig, l5kit_indices

class PredicitionEvalConfig(EvaluationConfig):
    def __init__(self, registered_name=''):
        super(PredicitionEvalConfig, self).__init__()
        
        # Test time registered_time can be different from training
        # it is used to select which set of parameters to use for evaluation
        self.registered_name = registered_name

        #
        # The most relevant args from EvaluationConfig. For rest, see that file.
        #
        self.name = "query_edit_eval"
        self.eval_class = "AutoBot"
        self.env = "trajdata" # only supported environment right now
        self.results_dir = "results"

        self.nnum_episode_repeats = 1

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 5
        self.policy.pos_to_yaw = False # NOTE: different 
        self.policy.yaw_correction_speed = 1.0
        self.policy.diversification_clearance = None
        self.policy.sample = True

        # if > 0.0 uses classifier-free guidance (mix of conditional and non-cond)
        # model at test time. Uses drop_fill value above.
        # 0.1 or 0.3 shown best in prior paper
        self.policy.class_free_guide_w = 0.0
        # if True, computes guidance loss only after full denoising and only uses
        #       to choose the action, not to get gradient to guide
        # TBD: this can be potentially removed given we already have "apply_guidance" parameter in the config
        self.policy.guide_as_filter_only = False
        # if True, chooses the same that's closest to GT at each planning step
        self.policy.guide_with_gt = False
        # whether to guide the predicted CLEAN or NOISY trajectory at each step
        # activated when apply_guidance = True
        self.policy.guide_clean = False # [False, "video_diff"]

        self.metrics.compute_analytical_metrics = True
        self.metrics.compute_learned_metrics = False

        
        self.trajdata.trajdata_rebuild_cache = False
        # number of simulations to run in each scene
        #       if > 1, each sim is running from a different starting point in the scene
        self.trajdata.num_sim_per_scene = 1

        self.trajdata.future_sec = 5.2 # 2.0 # 5.2 # 14.0
        self.trajdata.history_sec = 3.0 # 1.0 # 3.0
        
        # True for visualizing all action samples, False by default
        self.save_action_samples = True
        # ---------------------------------------------------------------------------------------
        # sampling and filtration based on configs for cvae, bits, diffuser 
        self.policy.num_action_samples = 5 # Diffuser: 20, CVAE: 20, BITS: 20
        # latent perturbation for cvae (trafficsim),
        # action perturbation for bc and bits,
        # action perturbation (every denoising step) for diffuser
        self.apply_guidance = True # this controls at the scene rollout level while guide_as_filter_only controls at the diffuser algorithm level 
        # constraints for diffuser
        self.apply_constraints = False

        # general optimization parameters for cvae, bc, bits, diffuser
        self.guidance_optimization_params = {
            'optimizer': 'adam', # ignored if video_diff
            'lr': 0.2,
            # Diffuser(20): 1, BC: 8, CVAE(20): 40, BITS(20): 5
            # Diffuser(1): 1, BC: 5, CVAE(1): 40, BITS(20): 1
            'grad_steps': 1, 
            'perturb_th': None, # when None, sigma is used for Diffuser; no threshold for others
        }
        # diffuser specific parameters
        self.diffusion_specific_params = {
            'stride': 1, # only for diffuser
            'apply_guidance_intermediate': True,
            'apply_guidance_output': False,
            'final_step_opt_params': {
                'optimizer': 'adam',
                'lr': 0.2,
                'grad_steps': 1,
                'perturb_th': 1,
            }
        }

        self.evaluation_vec_map_params = {
            'S_seg': 15,
            'S_point': 80,
            'map_max_dist': 80,
            'max_heading_error': 0.25*np.pi,
            'ahead_threshold': -40,
            'dist_weight': 1.0,
            'heading_weight': 0.1,
        }

        # ---------------------------------------------------------------------------------------
        ## nusc
        if 'nusc' in registered_name:
            self.trajdata.trajdata_source_test = ["nusc_trainval-val"]
            if 'ngc' in registered_name:
                self.trajdata.trajdata_cache_location = "/workspace/unified_data_cache"
                self.trajdata.trajdata_data_dirs = {
                "nusc_trainval" : "/workspace/nuscenes/",
                }  
            else:
                self.trajdata.trajdata_cache_location = "~/.unified_data_cache"
                self.trajdata.trajdata_data_dirs = {
                    "nusc_trainval" : "../behavior-generation-dataset/nuscenes",
                }
                
            self.trajdata.num_scenes_to_evaluate = 100 # 1 # 100 # 1 # 7 # 2 # 100 # 1
            self.trajdata.eval_scenes = np.arange(100).tolist() # [68] # np.arange(100).tolist() # [68] # [44] # [88] # [3] # [10] # [2] # [10] # [99] # [7] # [56] # [40] # [1] # [89] # [27] # [31, 32] # np.arange(100).tolist() # [63] # [30]
            self.trajdata.n_step_action = 5
            self.trajdata.num_simulation_steps = 100
            self.trajdata.skip_first_n = 0
        else:
            print('-'*20)
            print('not supported registered_name: '+registered_name)
            print('-'*20)
            # raise NotImplementedError('registered_name: '+registered_name)
        
        self.edits.editing_source = ['config', 'heuristic']
        # ---------------------------------------------------------------------------------------

        self.edits.guidance_config = []
        self.edits.guidance_config = [
            [
            ],
        ]

        self.edits.constraint_config = []

        # which heuristics guidances to apply on the fly
        self.edits.heuristic_sol_config = [
            {
             'name' : 'follow_traj',
             'weight' : 1.0,
             'params' : {
                            'decay_rate' : 0.9, 
                            'horizon' : 52, 
                            'guide_moving_speed_th': 5e-1,
                        },
             'agents' : None,
            },
            {
             'name' : 'agent_collision',
             'weight' : 500.0,
             'params' : {
                            'num_disks' : 2,
                            'buffer_dist': 0.5,
                            'decay_rate': 0.95,
                            'excluded_agents': None,
                        },
             'agents' : None,
            },            
            # # 4,5,6.map collision
            {
             'name' : 'map_collision',
             'weight' : 1.0,
             'params' : {
                            'num_points_lw' : (10, 10),
                            'decay_rate': 0.9,
                        },
             'agents' : None,
            },   
        ]
        self.edits.heuristic_config = [
            {
             'name' : 'adv_loss',
             'weight' : 1.0,
             'params' : {
                            'guide_moving_speed_th' : 5e-1, 
                            'crash_min_infront' : 0.0, 
                            'crash_min_t' : 0
                        },
             'agents' : None,
            },
            {
             'name' : 'agent_collision',
             'weight' : 1.0,
             'params' : {
                            'num_disks' : 2,
                            'buffer_dist': 0.2,
                            'decay_rate': 0.9,
                            'excluded_agents': None,
                        },
             'agents' : None,
            },            
            # # 4,5,6.map collision
            {
             'name' : 'map_collision',
             'weight' : 1.0,
             'params' : {
                            'num_points_lw' : (10, 10),
                            'decay_rate': 0.9,
                        },
             'agents' : None,
            },   
        ]
        

    def clone(self):
        return deepcopy(self)


class TrainTimeEvaluationConfig(EvaluationConfig):
    def __init__(self, registered_name=''):
        super(TrainTimeEvaluationConfig, self).__init__()

        self.num_scenes_per_batch = 1
        self.policy.sample = False
        # if > 0.0 uses classifier-free guidance (mix of conditional and non-cond)
        # model at test time. Uses drop_fill value above.
        # 0.1 or 0.3 shown best in prior paper
        self.policy.class_free_guide_w = 0.0
        # if True, computes guidance loss only after full denoising and only uses
        #       to choose the action, not to get gradient to guide
        # TBD: this can be potentially removed given we already have "apply_guidance" parameter in the config
        self.policy.guide_as_filter_only = False
        # whether to guide the predicted CLEAN or NOISY trajectory at each step
        self.policy.guide_clean = "video_diff" # [False, "video_diff"]

        # number of action samples to draw during evaluation at training time
        self.policy.num_action_samples = 2

        ## nusc
        if 'nusc' in registered_name:
            self.trajdata.trajdata_source_test = ["nusc_trainval-val"]
            self.trajdata.trajdata_data_dirs = {
                "nusc_trainval" : "../behavior-generation-dataset/nuscenes",
            }
            self.trajdata.num_scenes_to_evaluate = 10
            self.trajdata.eval_scenes = np.arange(0, 100, 10).tolist()
            self.trajdata.n_step_action = 5
            self.trajdata.num_simulation_steps = 100 # 5 # 100
            self.trajdata.skip_first_n = 0
        else:
            print('registered_name: '+registered_name)
            # raise NotImplementedError('registered_name: '+registered_name)