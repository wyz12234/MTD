import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from trajdata import AgentBatch, AgentType, UnifiedDataset
import os
import numpy as np
from tbsim.utils.trajdata_utils import get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid
from models.autobot_model import AutoBotEgoModel
from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict

def main(args):
    pl.seed_everything(42)
    vec_map_params = {
        'S_seg' : 15,
        'S_point' : 80,
        'map_max_dist' : 80,
        'max_heading_error' : 0.25 * np.pi,
        'ahead_threshold' : -40,
        'dist_weight' : 1.0,
        'heading_weight' : 0.1,
    }

    kwargs = dict(
        cache_location = '~/.unified_data_cache',
        desired_data = ['nusc_trainval-train', 'nusc_trainval-train_val'],
        desired_dt = 0.1,
        future_sec = (5.2, 5.2),
        history_sec = (3.0, 3.0),
        data_dirs = {
            'nusc_trainval': args.dataset_path,
            'nusc_test': args.dataset_path,
            'nusc_mini': args.dataset_path,
        },
        only_types = [AgentType.VEHICLE],
        only_predict = [AgentType.VEHICLE],
        agent_interaction_distances = defaultdict(lambda: 50),
        incl_raster_map = True,
        raster_map_params = {
            'px_per_m' : int(1 / 0.1),
            'map_size_px' : 224,
            'return_rgb' : False,
            'offset_frac_xy' : (-0.5, 0.0),
            'no_map_fill_value' : -1.0,
        },
        incl_vector_map = True,
        centric = 'agent',
        scene_description_contains = None,
        standardize_data = True,
        verbose = True,
        max_neighbor_num = 20,
        num_workers = os.cpu_count(),
        rebuild_cache = False,
        rebuild_maps = False,
        extras={
            "closest_lane_point": get_closest_lane_point_wrapper(vec_map_params),
            "full_fut_traj": get_full_fut_traj,
            "full_fut_valid": get_full_fut_valid,
        },
    )
    print(kwargs)

    train_dataset = UnifiedDataset(**kwargs)

    kwargs['desired_data'] = ['nusc_trainval-val']
    valid_dataset = UnifiedDataset(**kwargs)

    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=256,
                        shuffle=True, 
                        num_workers=8,
                        drop_last=True,
                        collate_fn=train_dataset.get_collate_fn(return_dict=True),
                        persistent_workers=False)
    valid_loader = DataLoader(
                        dataset=valid_dataset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=6,
                        drop_last=True,
                        collate_fn=valid_dataset.get_collate_fn(return_dict=True),
                        persistent_workers=False)
    
    model = AutoBotEgoModel()

    callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=5,
        save_last=True,
        filename='autobot-{epoch:02d}-{val/loss:.2f}',
        verbose=True,
    )

    logger = WandbLogger(name=None, project='autobot', entity='wyz12234')
    # logger = None

    trainer = pl.Trainer(
        default_root_dir='autobot_logs',
        enable_checkpointing=True,
        logger=logger,
        max_epochs=20,
        devices=1,
        accelerator='gpu',
        callbacks=[callback],
    )
    trainer.fit(model, train_loader, valid_loader)

    









if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/home/wyz/behavior-generation-dataset/nuscenes',
        help="(optional) if provided, override the dataset root path",
    )

    args = parser.parse_args()

    main(args)

