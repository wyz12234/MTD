from pathlib import Path
from trajdata.maps.map_api import MapAPI
import torch

def load_vec_map(map_name, cache_path="~/.unified_data_cache"):
    cache_path = Path(cache_path).expanduser()
    mapAPI = MapAPI(cache_path)
    vec_map = mapAPI.get_map(map_name, scene_cache=None)
    return vec_map


def extract_data_batch_for_guidance(batch):
    batch_for_guidance = {}
    for k in batch.keys():
        if k == 'extras':
            if 'extras' in batch:
                batch_for_guidance['extras'] = {}
                for j in batch['extras']:
                    B, M = batch['extras'][j].shape[:2]
                    batch_for_guidance['extras'][j] = batch['extras'][j].reshape(B * M, *batch['extras'][j].shape[2:])
        elif k == 'map_names':
            if isinstance(batch[k][0], list):
                map_name = batch[k][0][0]
            else:
                map_name = batch[k][0]
            vec_map = load_vec_map(map_name)
            batch_for_guidance['vec_map'] = vec_map
        else:
            if k == 'num_agents':
                batch_for_guidance[k] = batch[k]
            else:
                B, M = batch[k].shape[:2]
                batch_for_guidance[k] = batch[k].reshape(B * M, *batch[k].shape[2:])

    return batch_for_guidance

def check_behind(ego_fut_expand, agent_fut, agent_yaw, crash_min_infront):
    '''
    checks if each attacker is behind the target at each time step.
    :return behind_steps: True if attacker currently behind tgt
    '''
    agent_vec = torch.cat([torch.cos(agent_yaw), torch.sin(agent_yaw)], dim=-1)
    agent2ego = (ego_fut_expand - agent_fut) / torch.norm(ego_fut_expand - agent_fut, dim=-1, keepdim=True)
    cos_sim = torch.sum(agent_vec * agent2ego, dim=-1)
    behind_steps = cos_sim < crash_min_infront
    return behind_steps
    
