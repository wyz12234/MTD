import argparse
import numpy as np
import json
import random
import yaml
import importlib
from pprint import pprint
from copy import deepcopy
import os
import torch
import pytorch_lightning as pl
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from configs.eval_config import PredicitionEvalConfig
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder
from tbsim.utils.env_utils import rollout_episodes
import h5py

from tbsim.policies.wrappers import (
    RolloutWrapper,
    Pos2YawWrapper,
)

from tbsim.utils.tensor_utils import map_ndarray
import tbsim.utils.tensor_utils as TensorUtils

from utils.solution_utils import check_vehicle_collision, run_find_solution_optim


def run_evaluation(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg):
    file_path = eval_cfg.file_path
    # file_path = '/home/wyz/diffusion_trajection/logs/preexp4_mtd_adv_0906/query_edit_eval/data.hdf5'  # adv data 
    # file_path = '/home/wyz/diffusion_trajection/logs/normal_0910/query_edit_eval/data.hdf5' # new ori data
    # file_path = '/home/wyz/diffusion_trajection/logs/original_for_exp4/query_edit_eval/data.hdf5' # old ori data
    data = {}
    with h5py.File(file_path, 'r') as file:
        # 遍历文件中的所有数据集
        for dataset_name in file.keys():
            # 读取数据集
            dataset = file[dataset_name]
            data[dataset_name] = {}
            for key in file[dataset_name].keys():
                data[dataset_name][key] = dataset[key][:]
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
        
    set_global_batch_type("trajdata")
    if eval_cfg.env == "nusc":
        set_global_trajdata_batch_env("nusc_trainval")
    elif eval_cfg.env == "trajdata":
        # assumes all used trajdata datasets use share same map layers
        set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    # print(eval_cfg)

    # for reproducibility
    torch.cuda.manual_seed(eval_cfg.seed)
    pl.seed_everything(eval_cfg.seed)
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    
    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # print(exp_config.algo)
    # ----------------------------------------------------------------------------------
    print('policy', policy)
    # ----------------------------------------------------------------------------------

    # create env

    if eval_cfg.env == "trajdata":
        env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env()

    # eval loop
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]


    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location='~/.unified_data_cache')

    result_stats = None
    scene_i = 0
    count = []
    eval_scenes = eval_cfg.eval_scenes
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        print('scene_indices', scene_indices)

        # check to make sure all the scenes are valid at starting step
        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
        scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        if len(scene_indices) == 0:
            print('no valid scenes in this batch, skipping...')
            torch.cuda.empty_cache()
            continue


        # if requested, split each scene up into multiple simulations
        start_frame_index = [[exp_config.algo.history_num_frames+1]] * len(scene_indices)
        if eval_cfg.num_sim_per_scene > 1:
            start_frame_index = []
            for si in range(len(scene_indices)):
                cur_scene = env._current_scenes[si].scene
                sframe = exp_config.algo.history_num_frames+1
                # want to make sure there's GT for the full rollout
                eframe = cur_scene.length_timesteps - eval_cfg.num_simulation_steps
                scene_frame_inds = np.linspace(sframe, eframe, num=eval_cfg.num_sim_per_scene, dtype=int).tolist()
                start_frame_index.append(scene_frame_inds)

        # how many sims to run for the current batch of scenes
        print('Starting frames in current scenes:', start_frame_index)
        for ei in range(eval_cfg.num_sim_per_scene):
            cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
            # double check all scenes are valid at the current start step
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
            sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
            if len(sim_scene_indices) == 0:
                torch.cuda.empty_cache()
                continue

        if len(sim_scene_indices) == 0:
            print('No scenes with valid heuristic configs in this scene, skipping...')
            torch.cuda.empty_cache()
            continue

        if eval_cfg.policy.pos_to_yaw:
            policy = Pos2YawWrapper(
                policy,
                dt=exp_config.algo.step_time,
                yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
            )

        scene_data = data[env.current_scene_names[0] +'_' + str(sim_start_frames[0])]
        
        # right now assume control of full scene
        rollout_policy = RolloutWrapper(agents_policy=policy)
        stats, info, renderings, adjust_plans = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=eval_cfg.num_episode_repeats,
            n_step_action=eval_cfg.n_step_action,
            skip_first_n=eval_cfg.skip_first_n,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            horizon=eval_cfg.num_simulation_steps,
            scene_data=scene_data,
        )
        
        print(info["scene_index"])
        print(sim_start_frames)
        pprint(stats)

        # compute adversaril generation success
        all_postions = info["buffer"][0]["centroid"]
        all_yaws = np.expand_dims(info["buffer"][0]["yaw"], axis=-1)
        all_lw = info["buffer"][0]["extent"][..., :2]
        all_yaws_vector = np.concatenate([np.cos(all_yaws), np.sin(all_yaws)], axis=-1)
        all_traj = np.concatenate([all_postions, all_yaws_vector], axis=-1)
        ego_traj = all_traj[0]
        agents_traj = all_traj[1:]
        ego_lw = all_lw[0, 0]
        agents_lw = all_lw[1:, 0]
        veh_coll, coll_time = check_vehicle_collision(ego_traj, ego_lw, agents_traj, agents_lw)
        if coll_time.shape[0] == 0:
            veh_coll = False
            coll_time = 100
        else:
            min_index = np.argmin(coll_time)
            coll_time = np.min(coll_time)
            veh_coll = veh_coll[min_index]
        print('is collision: ', veh_coll)
        if veh_coll:
            count.append(info["scene_index"][0])


        # aggregate stats from the same class of guidance within each scene
        #       this helps parse_scene_edit_results
        guide_agg_dict = {}
        pop_list = []
        for k,v in stats.items():
            if k.split('_')[0] == 'guide':
                guide_name = '_'.join(k.split('_')[:-1])
                guide_scene_tag = k.split('_')[-1][:2]
                canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                if canon_name not in guide_agg_dict:
                    guide_agg_dict[canon_name] = []
                guide_agg_dict[canon_name].append(v)
                # remove from stats
                pop_list.append(k)
        for k in pop_list:
            stats.pop(k, None)
        # average over all of the same guide stats in each scene
        for k,v in guide_agg_dict.items():
            scene_stats = np.stack(v, axis=0) # guide_per_scenes x num_scenes (all are nan except 1)
            stats[k] = np.mean(scene_stats, axis=0)

        # aggregate metrics stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                if k not in result_stats:
                    result_stats[k] = stats[k]
                else:
                    result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

        # write stats to disk
        with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
            stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
            json.dump(stats_to_write, fp)

        if render_to_video or render_to_img:
            # high quality
            from tbsim.utils.scene_edit_utils import visualize_guided_rollout
            scene_cnt = 0
            for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
                viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
                visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                            fps=(1.0 / exp_config.algo.step_time),
                                            n_step_action=eval_cfg.n_step_action,
                                            viz_diffusion_steps=False,
                                            first_frame_only=render_to_img,
                                            sim_num=sim_start_frames[scene_cnt],
                                            save_every_n_frames=render_cfg['save_every_n_frames'],
                                            draw_mode=render_cfg['draw_mode'],)
                scene_cnt += 1

        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                sim_start_frames,
                h5_path=eval_cfg.experience_hdf5_path
            )
        torch.cuda.empty_cache()
    
    print('count: ', len(count), count)
    print('collision rate: ', len(count) / eval_cfg.num_scenes_to_evaluate)


def dump_episode_buffer(buffer, scene_index, start_frames, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    for ei, si, scene_buffer in zip(start_frames, scene_index, buffer):
        for mk in scene_buffer:
            h5key = "/{}_{}/{}".format(si, ei, mk)
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "trajdata"],
        help="Which env to run editing in",
        default="trajdata"
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default='AutoBot',
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default='/home/wyz/diffusion_trajection/query_centric_diffuser_trained_models/autobot/0909',
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default='model.ckpt',
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ------ for BITS ------
    parser.add_argument(
        "--planner_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--planner_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--predictor_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--predictor_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ----------------------

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="exp4_mtd_0918_finetune_adv",
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/home/wyz/behavior-generation-dataset/nuscenes',
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_query',
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to only render the first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )
    
    parser.add_argument(
        "--file_path",
        type=str,
        default='/home/wyz/diffusion_trajection/logs/preexp4_mtd_adv_0906/query_edit_eval/data.hdf5',
    )
    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "ui", "none"],
        nargs="+",
        default=["config", "heuristic"],
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. UI will use interactive interface. \
              config and heuristic may be used together. If none, does not use edits."
    )


    args = parser.parse_args()

    cfg = PredicitionEvalConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.planner_ckpt_dir is not None:
        cfg.ckpt.planner.ckpt_dir = args.planner_ckpt_dir
        cfg.ckpt.planner.ckpt_key = args.planner_ckpt_key

    if args.predictor_ckpt_dir is not None:
        cfg.ckpt.predictor.ckpt_dir = args.predictor_ckpt_dir
        cfg.ckpt.predictor.ckpt_key = args.predictor_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join('logs', args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    
    # add eval_class into the results_dir
    # cfg.results_dir = os.path.join(cfg.results_dir, cfg.eval_class)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None
    
    if args.file_path is not None:
        cfg.file_path = args.file_path

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        # can only handle one scene with UI
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")
    cfg.experience_sol_hdf5_path = os.path.join(cfg.results_dir, "sol_data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    cfg.lock()
    run_evaluation(
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
    )
