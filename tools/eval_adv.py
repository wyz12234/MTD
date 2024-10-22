import h5py
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.solution_utils import check_vehicle_collision
time2coll = {}
# 打开HDF5文件
file_path = 'nusc_results_debug/query_edit_eval/adv_data.hdf5'  # 文件路径
data = {}
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的所有数据集
    for dataset_name in file.keys():
        # 读取数据集
        dataset = file[dataset_name]
        data[dataset_name] = {}
        for key in file[dataset_name].keys():
            data[dataset_name][key] = dataset[key][:]
count = 0
sum = 0
for key in data.keys():
    all_postions = data[key]["centroid"]
    all_yaws = np.expand_dims(data[key]["yaw"], axis=-1)
    all_lw = data[key]["extent"][..., :2]
    all_yaws_vector = np.concatenate([np.cos(all_yaws), np.sin(all_yaws)], axis=-1)
    all_traj = np.concatenate([all_postions, all_yaws_vector], axis=-1)
    ego_traj = all_traj[0]
    agents_traj = all_traj[1:]
    ego_lw = all_lw[0, 0]
    agents_lw = all_lw[1:, 0]
    veh_coll, coll_time = check_vehicle_collision(ego_traj, ego_lw, agents_traj, agents_lw)
    sum += 1
    if any(veh_coll):
        print(f"Collision in {key} at time {coll_time[veh_coll]}")
        time2coll[key] = min(coll_time)
    else:
        time2coll[key] = 100
        count += 1
print(f"Total {count} trajectories are collision free.")
print(f"Total {sum} trajectories are evaluated.")

# 打开HDF5文件
file_path = 'nusc_results_debug/query_edit_eval/sol_data.hdf5'  # 文件路径
sol_data = {}
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的所有数据集
    for dataset_name in file.keys():
        # 读取数据集
        dataset = file[dataset_name]
        sol_data[dataset_name] = {}
        for key in file[dataset_name].keys():
            sol_data[dataset_name][key] = dataset[key][:]


# 打开HDF5文件
file_path = 'nusc_results_debug/query_edit_eval/ori_data.hdf5'  # 文件路径
data = {}
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的所有数据集
    for dataset_name in file.keys():
        # 读取数据集
        dataset = file[dataset_name]
        data[dataset_name] = {}
        for key in file[dataset_name].keys():
            data[dataset_name][key] = dataset[key][:]
count = 0
sum = 0
norm_coll = 0
for key in data.keys():
    all_postions = data[key]["centroid"]
    all_yaws = np.expand_dims(data[key]["yaw"], axis=-1)
    all_lw = data[key]["extent"][..., :2]
    all_yaws_vector = np.concatenate([np.cos(all_yaws), np.sin(all_yaws)], axis=-1)
    all_traj = np.concatenate([all_postions, all_yaws_vector], axis=-1)
    ego_traj = all_traj[0]
    agents_traj = all_traj[1:]
    ego_lw = all_lw[0, 0]
    agents_lw = all_lw[1:, 0]
    veh_coll, coll_time = check_vehicle_collision(ego_traj, ego_lw, agents_traj, agents_lw)
    
    if coll_time.shape[0] == 0:
        coll_time = [100, 100]
    if key not in sol_data.keys():
        continue
    sum += 1
    if min(coll_time) < 100: # min(100, time2coll[key] + 5):
        print(f"Collision in {key} at time {coll_time[veh_coll]}")
        if time2coll[key] == 100:
            norm_coll += 1
    else:
        count += 1
print(f"Total {count} trajectories are collision free.")
print(f"Total {sum} trajectories are evaluated.")
print(f"Total {norm_coll} trajectories are collision after finetune.")