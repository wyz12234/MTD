import os
import h5py
import argparse
from tbsim.utils.scene_edit_utils import visualize_guided_rollout, get_trajdata_renderer
import shutil
import numpy as np
from utils.solution_utils import check_vehicle_collision

def visualize_scenarios(file_path, save_path, viz=False):
    os.makedirs(save_path, exist_ok=True)
    data = {}
    name_list = []

    if viz:

        render_rasterizer = get_trajdata_renderer(["nusc_trainval-val"],
                                                {"nusc_trainval" : "../behavior-generation-dataset/nuscenes", },
                                                future_sec=5.2,
                                                history_sec=3.0,
                                                raster_size=400,
                                                px_per_m=2.0,
                                                rebuild_maps=False,
                                                cache_location='~/.unified_data_cache')

    with h5py.File(file_path, 'r') as file:
        for dataset_name in file.keys():
            dataset = file[dataset_name]
            data[dataset_name] = {}
            for key in file[dataset_name].keys():
                data[dataset_name][key] = dataset[key][:]

    for key, scene_buffer in data.items():
        if key != 'scene-0107_31':
            continue
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
        # if len(coll_time) != 0:
        #     scene_buffer['centroid'] = scene_buffer['centroid'][:, :50]

        # scene_buffer['centroid'][4] = scene_buffer['centroid'][4][0] * np.ones_like(scene_buffer['centroid'][4])
        # scene_buffer['yaw'][4] = scene_buffer['yaw'][4][0] * np.ones_like(scene_buffer['yaw'][4])
        # scene_buffer['centroid'][10] = scene_buffer['centroid'][10][0] * np.ones_like(scene_buffer['centroid'][10])
        # scene_buffer['yaw'][10] = scene_buffer['yaw'][10][0] * np.ones_like(scene_buffer['yaw'][10])
        if viz:
            start_frame_index = int(key[-2:])
            si = key[:-3]
            visualize_guided_rollout(save_path, render_rasterizer, si, scene_buffer,
                                    guidance_config=None,
                                    constraint_config=None,
                                    fps=(1.0 / 0.1),
                                    n_step_action=5,
                                    viz_diffusion_steps=False,
                                    first_frame_only=True,
                                    sim_num=start_frame_index,
                                    save_every_n_frames=5,
                                    draw_mode='entire_traj')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default='/data1/wyz/exp1/qcdiffuser/viz_exp4/normal.hdf5',
                        help="Path to the hdf5 file")
    parser.add_argument("--save_path", type=str, default='./viz_exp4/107_normal_visualize',
                        help="Path to save the visualized results")
    parser.add_argument("--viz", action='store_false',
                        help="Visualize the data")
    args = parser.parse_args()                       

    visualize_scenarios(args.file_path, args.save_path, args.viz)