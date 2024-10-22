import pickle
import h5py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.geometry_utils import transform_agents_to_world
import torch.nn.functional as F
import json



def pad_and_truncate(tensor, target_length=20):
    # 获取输入张量的形状
    original_length = tensor.size(1)

    # 计算需要填充的长度
    padding_length = max(0, target_length - original_length)

    # 根据张量的维度进行填充
    if tensor.dim() == 3:
        # 三维张量填充 (仅在维度1上填充)
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_length, 0, 0))
        # 截断张量
        truncated_tensor = padded_tensor[:, :target_length, :]
    elif tensor.dim() == 4:
        # 四维张量填充 (仅在维度1上填充)
        padded_tensor = F.pad(tensor, (0, 0, 0, 0, 0, padding_length, 0, 0))
        # 截断张量
        truncated_tensor = padded_tensor[:, :target_length, :, :]

    return truncated_tensor

def add_scene_dim_to_agent_data(obs):
    '''
    A dummy wrapper that add one dimension to each field.
    '''
    new_obs = {}
    for k in obs.keys():
        if isinstance(obs[k], torch.Tensor):
            new_obs[k] = obs[k].unsqueeze(0)
        else:
            new_obs[k] = [obs[k]]
        
    if 'extras' in obs:
        new_obs['extras'] = {}
        for k in obs['extras']:
            new_obs['extras'][k] = obs['extras'][k].unsqueeze(0)
            
    return new_obs

def get_files_sorted_by_date(folder_path):
    # 获取文件夹中的所有文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 按照修改日期从远到近排序
    files.sort(key=lambda x: os.path.getmtime(x))
    
    # 提取文件名（不含后缀名）并去掉倒数第三和第四个字符
    sorted_filenames = [os.path.splitext(os.path.basename(f))[0][:-4] + os.path.splitext(os.path.basename(f))[0][-2:] for f in files]
    return sorted_filenames

def read_offroad_ego_frame(json_file_path):
    # 打开并读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 获取 "offroad_ego_frame" 字段
    offroad_ego_frame = data.get("offroad_ego_frame")
    return offroad_ego_frame

file_path = ''
pkl_path = ''
save_name = ''
data = {}
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的所有数据集
    for dataset_name in file.keys():
        # 读取数据集
        dataset = file[dataset_name]
        data[dataset_name] = {}
        for key in file[dataset_name].keys():
            data[dataset_name][key] = dataset[key][:]

pkl_exist = True

ckpt = {
    "agent_hist": [],
    "neigh_hist": [],
    "map_polyline": [],
    "target_pos": [],
}
time = None
with open(pkl_path, 'rb') as f:
    global_i = 0
    while pkl_exist:
        try:
            data_pkl = pickle.load(f)
        except:
            pkl_exist = False

        for i in range(len(data_pkl)):
            obs_dict = TensorUtils.to_torch(data_pkl[i], device='cpu', ignore_if_unspecified=True)['agents']
            obs_dict = add_scene_dim_to_agent_data(obs_dict)
            
            obs_dict["num_agents"] = torch.tensor([obs_dict['target_positions'].shape[1]],device=obs_dict['target_positions'].device)
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

            obs_dict["history_positions"] = pad_and_truncate(obs_dict["history_positions"], target_length=21)
            obs_dict["history_yaws"] = pad_and_truncate(obs_dict["history_yaws"], target_length=21)
            obs_dict["history_speeds"] = pad_and_truncate(obs_dict["history_speeds"], target_length=21)
            obs_dict["history_availabilities"] = pad_and_truncate(obs_dict["history_availabilities"], target_length=21)
            temp_extent = pad_and_truncate(temp_extent, target_length=21)

            agent_hist = torch.cat([obs_dict["history_positions"], 
                                    torch.cos(obs_dict["history_yaws"]),
                                    torch.sin(obs_dict["history_yaws"]),
                                    obs_dict["history_speeds"].unsqueeze(-1).expand_as(obs_dict["history_yaws"]), 
                                    temp_extent,
                                    obs_dict["history_availabilities"].unsqueeze(-1).expand_as(obs_dict["history_yaws"])], dim=-1)[0]
            
            map_polyline = obs_dict["extras"]["closest_lane_point"][0, 0]

            nan_inds = torch.sum(torch.isnan(map_polyline).float(), dim=-1) > 0
            avail = torch.ones_like(map_polyline[..., 0])
            avail[nan_inds] = 0
            map_polyline = torch.nan_to_num(map_polyline, nan=0.0)
            map_polyline = torch.cat([map_polyline, avail.unsqueeze(-1)], dim=-1)

            neigh_hist = agent_hist[1:]
            agent_hist = agent_hist[0]

            if obs_dict.get('first_scene_ts') is not None:
                time = str(obs_dict['first_scene_ts'][0][0].item())

            target_pos = torch.from_numpy(data[obs_dict['scene_ids'][0][0] + '_' + time]['action_traj_positions'][0, ::5])[i]

            ckpt["agent_hist"].append(agent_hist)
            ckpt["neigh_hist"].append(neigh_hist.transpose(0, 1))
            ckpt["map_polyline"].append(map_polyline)
            ckpt["target_pos"].append(target_pos)
        global_i += 1
        print(obs_dict['scene_ids'][0][0] + '_' + time)

# 删除最后20个
ckpt["agent_hist"] = ckpt["agent_hist"][:-20]
ckpt["neigh_hist"] = ckpt["neigh_hist"][:-20]
ckpt["map_polyline"] = ckpt["map_polyline"][:-20]
ckpt["target_pos"] = ckpt["target_pos"][:-20]


ckpt["agent_hist"] = torch.stack(ckpt["agent_hist"])
ckpt["neigh_hist"] = torch.stack(ckpt["neigh_hist"])
ckpt["map_polyline"] = torch.stack(ckpt["map_polyline"])
ckpt["target_pos"] = torch.stack(ckpt["target_pos"])
torch.save(ckpt, save_name)


        
    
 