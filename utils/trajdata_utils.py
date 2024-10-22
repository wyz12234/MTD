import torch
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig
from trajdata.data_structures.state import StateTensor,StateArray

from trajdata import AgentBatch, AgentType
from trajdata.utils.arr_utils import angle_wrap

from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadLane
from pathlib import Path
from trajdata.maps.map_api import MapAPI
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np, transform_xyh_np
from trajdata.utils.state_utils import transform_state_np_2d
from typing import Union
from torch.nn.utils.rnn import pad_sequence

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from torch import Tensor
from trajdata.visualization.vis import draw_map, draw_agent, draw_history
BATCH_RASTER_CFG = \
{
    "include_hist": True,
    "num_sem_layers": 3,
    "drivable_layers": None,
    "rgb_idx_groups": [
        [
            0
        ],
        [
            1
        ],
        [
            2
        ]
    ],
    "raster_size": 224,
    "pixel_size": 0.5,
    "ego_center": [
        -0.5,
        0.0
    ],
    "no_map_fill_value": -1.0
}

def convert_scene_obs_to_query_prep(obs):
    '''
    场景、世界、车辆坐标系之间的转换
    '''
    B, M = obs["center_from_agents"].shape[:2]
    new_obs = {}

    agent_from_raster_list = []
    raster_from_agent_list = []
    agent_from_world_list = []
    world_from_agent_list = []
    raster_from_world_list = []
    centroid_list = []
    yaw_list = []
    map_names_list = []

    center_from_raster = obs["agent_from_raster"]
    raster_from_center = obs["raster_from_agent"]
    
    raster_cfg = BATCH_RASTER_CFG
    map_res = 1.0 / raster_cfg["pixel_size"] # convert to pixels/meter
    h = w = raster_cfg["raster_size"]
    ego_cent = raster_cfg["ego_center"]

    raster_from_agent = torch.Tensor([
            [map_res, 0, ((1.0 + ego_cent[0])/2.0) * w],
            [0, map_res, ((1.0 + ego_cent[1])/2.0) * h],
            [0, 0, 1]
    ]).to(center_from_raster.device)
    

    for i in range(B):
        center_from_agents = obs["center_from_agents"][i]
        agents_from_center = obs["agents_from_center"][i]

        center_from_world = obs["agent_from_world"][i]
        world_from_center = obs["world_from_agent"][i]

        agents_from_world = agents_from_center @ center_from_world
        world_from_agents = world_from_center @ center_from_agents

        raster_from_world = raster_from_agent @ agents_from_world

        agent_from_raster_list.append(center_from_raster.repeat(M, 1, 1))
        raster_from_agent_list.append(raster_from_center.repeat(M, 1, 1)) 
        agent_from_world_list.append(agents_from_world)
        world_from_agent_list.append(world_from_agents)
        raster_from_world_list.append(raster_from_world)

        centroid_list.append(GeoUtils.transform_points_tensor(obs["history_positions"][i], world_from_center)[:, -1])
        yaw_list.append(obs["history_yaws"][i, :, -1, 0] + obs["yaw"][i])

        map_names_list.append([obs['map_names'][i] for _ in range(M)])

    new_obs['agent_from_raster'] = torch.stack(agent_from_raster_list, dim=0)
    new_obs['raster_from_agent'] = torch.stack(raster_from_agent_list, dim=0)
    new_obs['agent_from_world'] = torch.stack(agent_from_world_list, dim=0)
    new_obs['world_from_agent'] = torch.stack(world_from_agent_list, dim=0)
    new_obs['raster_from_world'] = torch.stack(raster_from_world_list, dim=0)
    new_obs['centroid'] = torch.stack(centroid_list, dim=0)
    new_obs['yaw'] = torch.stack(yaw_list, dim=0)

    # for guidance loss estimation
    new_obs['drivable_map'] = obs['drivable_map']
    # for map related guidance
    new_obs['map_names'] = map_names_list

    return new_obs