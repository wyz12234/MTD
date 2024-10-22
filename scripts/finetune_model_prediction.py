import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from models.autobot_model import AutoBotEgoModel

class FinetuneDataset(Dataset):
    def __init__(self, file_path):
        # 加载保存的数据
        ckpt = torch.load(file_path)
        
        # 提取各部分数据
        self.agent_hist = ckpt["agent_hist"]
        self.neigh_hist = ckpt["neigh_hist"]
        self.map_polyline = ckpt["map_polyline"]
        self.target_pos = ckpt["target_pos"]
        
        # 确定数据集大小
        self.dataset_size = self.agent_hist.size(0)
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # 返回数据项
        return {
            "agent_hist": self.agent_hist[idx],
            "neigh_hist": self.neigh_hist[idx],
            "map_polyline": self.map_polyline[idx],
            "target_pos": self.target_pos[idx]
        }
    
def main(args):
    pl.seed_everything(42)
    dataset = FinetuneDataset(args.dataset_path)
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]) 
    train_dataset = dataset
    valid_dataset = dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = AutoBotEgoModel.load_from_checkpoint(args.ckpt_path)
    model.fine_tune = True

    callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=5,
        save_last=True,
        filename='autobot-{epoch:02d}-{val/loss:.2f}',
        verbose=True,
    )

    logger = WandbLogger(
    name=None,
    project='autobot_finetune',
    entity='wyz12234',
    save_dir="./logs"  # 设置日志存储路径
    )
    # logger = None

    trainer = pl.Trainer(
        enable_checkpointing=True,
        logger=logger,
        max_epochs=100,
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
        default='/data/merged_ckpt_6_0910.pth',
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='/home/wyz/diffusion_trajection/query_centric_diffuser_trained_models/autobot/run0/checkpoints/loss=-13835.57.ckpt',
        help="(optional) if provided, override the ckpt root path",
    )

    args = parser.parse_args()

    main(args)

