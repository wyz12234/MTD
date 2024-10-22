from tbsim.configs.base import ExperimentConfig
from tbsim.configs.config import Dict

def translate_query_centric_trajdata_cfg(cfg: ExperimentConfig):
    rcfg = Dict()
    # assert cfg.algo.step_time == 0.5  # TODO: support interpolation
    if "scene_centric" in cfg.algo and cfg.algo.scene_centric:
        rcfg.centric="scene"
    else:
        rcfg.centric="agent"
    if "standardize_data" in cfg.env.data_generation_params:
        rcfg.standardize_data = cfg.env.data_generation_params.standardize_data
    else:
        rcfg.standardize_data = True
    rcfg.step_time = cfg.algo.step_time
    rcfg.trajdata_source_root = cfg.train.trajdata_source_root
    rcfg.trajdata_source_train = cfg.train.trajdata_source_train
    rcfg.trajdata_source_train_val = cfg.train.trajdata_source_train_val
    rcfg.trajdata_source_valid = cfg.train.trajdata_source_valid
    rcfg.dataset_path = cfg.train.dataset_path
    rcfg.history_num_frames = cfg.algo.history_num_frames
    rcfg.future_num_frames = cfg.algo.future_num_frames
    rcfg.other_agents_num = cfg.env.data_generation_params.other_agents_num
    rcfg.max_agents_distance = cfg.env.data_generation_params.max_agents_distance
    rcfg.max_agents_distance_simulation = cfg.env.simulation.distance_th_close
    rcfg.pixel_size = cfg.env.rasterizer.pixel_size
    rcfg.raster_size = int(cfg.env.rasterizer.raster_size)
    rcfg.raster_center = cfg.env.rasterizer.ego_center
    rcfg.yaw_correction_speed = cfg.env.data_generation_params.yaw_correction_speed
    if "vectorize_lane" in cfg.env.data_generation_params:
        rcfg.vectorize_lane = cfg.env.data_generation_params.vectorize_lane
    else:
        rcfg.vectorize_lane = "None"
        
    rcfg.lock()
    return rcfg
