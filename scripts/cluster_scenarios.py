
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import torch
import argparse
from utils.solution_utils import check_vehicle_collision, transform2frame
from tbsim.utils.scene_edit_utils import visualize_guided_rollout, get_trajdata_renderer
import shutil



def compute_coll_feat(ori_data):
    all_postions = ori_data["centroid"]
    all_yaws = np.expand_dims(ori_data["yaw"], axis=-1)
    all_lw = ori_data["extent"][..., :2]
    all_yaws_vector = np.concatenate([np.cos(all_yaws), np.sin(all_yaws)], axis=-1)
    all_traj = np.concatenate([all_postions, all_yaws_vector], axis=-1)
    ego_traj = all_traj[0]
    agents_traj = all_traj[1:]
    ego_lw = all_lw[0, 0]
    agents_lw = all_lw[1:, 0]
    veh_coll, coll_time = check_vehicle_collision(ego_traj, ego_lw, agents_traj, agents_lw)

    if any(veh_coll):
        atk_agent_index = np.argmin(coll_time)
        plan_coll_states = torch.tensor(ego_traj[coll_time[atk_agent_index]: coll_time[atk_agent_index] + 1])
        atk_coll_states = torch.tensor(agents_traj[atk_agent_index, coll_time[atk_agent_index]: coll_time[atk_agent_index] + 1])
        local_atk_states = transform2frame(plan_coll_states, atk_coll_states.unsqueeze(1))[0, 0]

        coll_h = torch.atan2(local_atk_states[3], local_atk_states[2]).item()
        coll_hvec = [local_atk_states[2].item(), local_atk_states[3].item()]
        coll_pos = local_atk_states[:2] / torch.norm(local_atk_states[:2], dim=0)
        coll_ang = torch.atan2(coll_pos[1], coll_pos[0]).item()

        feat = {
        'h' : coll_h,
        'hvec': coll_hvec,
        'ang' : coll_ang,
        'angvec' : coll_pos.numpy().tolist()
        }

        valid = True
    else:
        feat = {
        'h' : 0,
        'hvec': [0, 0],
        'ang' : 0,
        'angvec' : [0, 0]
        }
        valid = False


    return feat, valid


def cluster_scenarios(file_path, num_clusters, save_path, viz=False):
    os.makedirs(save_path, exist_ok=True)
    data = {}
    feat_list = []
    name_list = []
    if viz:
        # initialize rasterizer once for all scenes
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
        scene_feat, valid = compute_coll_feat(scene_buffer)
        if valid:
            feat_list.append(scene_feat)
            name_list.append(key)

        if viz and valid:
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
                                    draw_mode='entire_traj',) # ['action', 'entire_traj', 'map']

    angvec = np.array([feat['angvec'] for feat in feat_list])
    hvec = np.array([feat['hvec'] for feat in feat_list])
    scene_feats = np.concatenate([angvec, hvec], axis=1)
    print(scene_feats.shape)

    # perform clustering
    print('Clustering using k=%d clusters...' % (num_clusters))
    clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(scene_feats)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'cluster.pkl'), 'wb') as f:
        pickle.dump(clustering, f)

    fig, axs = plt.subplots(1, 2)
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    axs[0].plot(x, y, '--b', alpha=0.15)
    axs[0].title.set_text('collision direction')
    axs[1].plot(x, y, '--b', alpha=0.15)
    axs[1].title.set_text('adversary heading')
    axs[0].axis('equal')
    axs[1].axis('equal')
    for ki in np.unique(labels):
        axs[0].plot(angvec[:,0][labels == ki], angvec[:,1][labels == ki], 'o', markersize=4, label='%d'%ki)
        axs[1].plot(hvec[:,0][labels == ki], hvec[:,1][labels == ki], 'o', markersize=4, label='%d'%ki)
        # move all videos from this cluster to corresponding folder
        if viz:
            cur_clust_out = os.path.join(save_path, 'viz_clust%02d' % (ki))
            os.makedirs(cur_clust_out, exist_ok=True)
            for si in np.nonzero(labels == ki)[0]:
                new_name = name_list[si][:-2] + '%04d' % int(name_list[si][-2:])
                shutil.move(os.path.join(save_path, new_name + '_000.svg'), 
                            os.path.join(cur_clust_out, new_name + '_000.svg'))

    # 调整图例位置到图形下方，2行5列排布
    axs[0].legend(title="Clusters", loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize='small', frameon=False)
    axs[1].legend(title="Clusters", loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize='small', frameon=False)

    # 使用subplots_adjust调整图形和图例之间的距离
    plt.subplots_adjust(bottom=0.3)

    plt.savefig(os.path.join(save_path, 'cluster_k%d.jpg' % (num_clusters)))
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default='/home/wyz/diffusion_trajection/logs/exp2_mtd_0829/query_edit_eval/data.hdf5',
                        help="Path to the hdf5 file")
    parser.add_argument("--num_clusters", type=int, default=8,
                        help="Number of clusters")
    parser.add_argument("--save_path", type=str, default='./logs/mtd_cluster',
                        help="Path to save the cluster centers")
    parser.add_argument("--viz", action='store_true',
                        help="Visualize the clusters")
    args = parser.parse_args()                       

    cluster_scenarios(args.file_path, args.num_clusters, args.save_path, args.viz)