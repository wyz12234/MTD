import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tbsim.configs.base import TrainConfig
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP, get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid

from trajdata import UnifiedDataset
import gc

class QCDiffuserDataModule(pl.LightningDataModule):
    """
    Pass-through config options to unified data loader.
    This is a more general version of the above UnifiedDataModule which 
    only supports any dataset available through trajdata.
    """
    def __init__(self, data_config, train_config: TrainConfig):
        super(QCDiffuserDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None
        self.num_sem_layers = None

    @property
    def modality_shapes(self):
        """
        Returns the expected shape of combined rasterized layers
        (semantic + traj history + current)
        """
        # num_history + current
        hist_layer_size = self._data_config.history_num_frames + 1 if self._data_config.raster_include_hist \
                            else 0
        return dict(
            image=(self.num_sem_layers + hist_layer_size,  # semantic map
                   self._data_config.raster_size,
                   self._data_config.raster_size)
        )

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time # 52 * 0.1
        history_sec = data_cfg.history_num_frames * data_cfg.step_time # 30 * 0.1
        neighbor_distance = data_cfg.max_agents_distance # 50
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types] # [AgentType.VEHICLE 1]
        agent_predict_types = None
        print("data_cfg.trajdata_predict_types", data_cfg.trajdata_predict_types)
        if data_cfg.trajdata_predict_types is not None:
            if data_cfg.other_agents_num is None:
                max_agent_num = None
            else:
                max_agent_num = 1 + data_cfg.other_agents_num # 21

            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types] # [AgentType.VEHICLE 1]
        kwargs = dict(
            cache_location=data_cfg.trajdata_cache_location, # "~/.unified_data_cache"
            desired_data=data_cfg.trajdata_source_train, # ["nusc_trainval-train", "nusc_trainval-train_val"]
            desired_dt=data_cfg.step_time, # 0.1
            future_sec=(future_sec, future_sec), # (5.2, 5.2)
            history_sec=(history_sec, history_sec), # (3.0, 3.0)
            data_dirs=data_cfg.trajdata_data_dirs, # {"nusc_trainval": "../behavior-generation-dataset/nuscenes"}
            only_types=agent_only_types, # [AgentType.VEHICLE 1]
            only_predict=agent_predict_types, # [AgentType.VEHICLE 1]
            agent_interaction_distances=defaultdict(lambda: neighbor_distance), # 50
            incl_raster_map=data_cfg.trajdata_incl_map, # True
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size), # 10 10个像素代表1米
                "map_size_px": data_cfg.raster_size, # 224 地图大小
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center, # (-0.5, 0.0) 地图中心
                "no_map_fill_value": data_cfg.no_map_fill_value, # -1.0
            },
            incl_vector_map=True,
            centric=data_cfg.trajdata_centric, # "scene"
            scene_description_contains=data_cfg.trajdata_scene_desc_contains, # null
            standardize_data=data_cfg.trajdata_standardize_data, # True 
            verbose=True,
            max_agent_num = max_agent_num, # 21
            num_workers=os.cpu_count(),
            rebuild_cache=data_cfg.trajdata_rebuild_cache, # False
            rebuild_maps=data_cfg.trajdata_rebuild_cache, # False
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
                "closest_lane_point": get_closest_lane_point_wrapper(self._train_config.training_vec_map_params),
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
            },
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = data_cfg.trajdata_source_valid
        self.valid_dataset = UnifiedDataset(**kwargs)

        # set modality shape based on input
        self.num_sem_layers = 0 if not data_cfg.trajdata_incl_map else data_cfg.num_sem_layers

        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True, # since pytorch lightning only evals a subset of val on each epoch, shuffle
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=False
        )