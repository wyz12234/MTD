# STD: Generating Realistic and Diverse Safety-Critical Scenarios for Autonomous Vehicles

STD is a query-centric multi-agent traffic generation model for safety-critical scenarios. STD integrates adversarial guidance and data-driven generation, offering improvements in handling dynamic interactions, enhancing scenario diversity, and boosting the performance of autonomous driving planners.

## Setup

### Python Enviroment

You can install our environment using either `pip` or `conda`.

**1.Installing with `pip`**

For `pip`, we can use the following command to install:


To install the environment using `pip`, use the following command:

```bash
pip install -r requirements.txt
```

**2.Installing with `conda`**

For ```conda```, we can use the following command to install:

```bash
conda env create -f environment.yml
```

### Installing `trajdata_std` designed specifically for STD

To install `trajdata_std` designed for STD, you may need to delete any references to related projects in `requirements.txt` or `environment.yml` to prevent errors.

```bash
cd ..
git clone https://github.com/wyz12234/trajdata_std.git
cd trajdata_std
pip install -e .
```

### Install `Pplan`

```bash
cd ..
git clone https://github.com/NVlabs/spline-planner.git
cd Pplan
pip install -e .
```

## Quick Start

### 1. Download nuScenes Dataset

To download nuScenes you need to go to the [Download page](https://www.nuscenes.org/download), 
create an account and agree to the nuScenes [Terms of Use](https://www.nuscenes.org/terms-of-use).

Download the nuScenes dataset (with the map extension pack) and organize the dataset directory as follows:
```
nuscenes/
│   maps/
│   └── expansion/
│   v1.0-mini/
│   v1.0-trainval/
```
You may also need to install `nuscenes-devkit`. For the installation guide, please refer to the following link: [`nuscenes-devkit`](https://github.com/nutonomy/nuscenes-devkit)

### 2. Train STD Model

The current configuration is based on the default settings from the paper's environment. If you need to modify the configuration and directory paths, you may need to update the following files:

- `configs/qc_diffuser_config.py`
- `scripts/train.py`

Before running the training script, you may need to add the current working directory to the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Run `scripts/train.py` to train the STD model:

```bash
python scripts/train.py --dataset_path <path-to-nuscenes-data-directory> --config_name trajdata_nusc_query --wandb_project_name std
```

### 3. Generate Safety-critical Scenarios and Corresponding Expert solutions

The current configuration is based on the default settings from the paper's environment. If you need to modify the configuration and directory paths, you may need to update the following files:

- `configs/query_edit_config.py`
- `scripts/scene_editor.py`

```bash
python scripts/scene_editor.py 
  --dataset_path <path-to-nuscenes-data-directory> \
  --results_root_dir <results_directory> \
  --env "trajdata" \
  --policy_ckpt_dir <path-to-ckpt-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class QCDiffuser \
  --training_data_dir <path-to-save-expert-training-data>
  --editing_source 'config' 'heuristic' \
  --registered_name "trajdata_nusc_query" \
  --render
```

### 4. Generate Long-term Regular Scenarios

First, you need to comment out the `adv_loss` adversarial guidance function in `configs/query_edit_config.py` to generate regular scenarios. Additionally, you need to change `self.trajdata.num_sim_per_scene` to `5` and `self.trajdata.num_simulation_steps` to `200` to measure the long-term generation performance of STD and the diversity of the generated scenarios.

```bash
python scripts/generate_regular_scene.py 
  --dataset_path <path-to-nuscenes-data-directory> \
  --results_root_dir <results_directory> \
  --env "trajdata" \
  --policy_ckpt_dir <path-to-ckpt-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class QCDiffuser \
  --editing_source 'config' 'heuristic' \
  --registered_name "trajdata_nusc_query" \
  --render
```

### 5. Parse Results for Scenarios

```bash
python scripts/parse_result.py
  --results_dir <results_directory> \
  --results_file <file-to-parse> \
  --estimate_dist
```

### 6. Create a Training Dataset Consisting of Expert Solutions

```bash
python tools/data_processer.py
```

You need to change the `file path`, `pkl_path`, and `save_name` to correspond to the hdf5 file and pkl file you generated, as well as the saved path and name.

### 7. Pre-Train Planner

We have selected [Autobots](https://github.com/roggirg/AutoBots) as our autonomous driving planner here, and pre-trained it using the nuScenes dataset by running the following script:

```bash
python scripts/train_model_prediction.py
```

### 8. Fine-Tune Planner

```bash
python scripts/finetune_model_prediction.py \
 --dataset_path <path-to-finetune-dataset> \
 --ckpt_path <path-to-planner-ckpt>
```

### 9. Test and Evaluate Planner

We can evaluate the performance of the pre-trained and fine-tuned planner in both safety-critical scenarios and regular scenarios, depending on your `file_path` configuration.

```bash
python scripts/eval_model_prediction.py \
  --dataset_path <path-to-nuscenes-data-directory> \
  --results_root_dir <results_directory> \
  --env "trajdata" \
  --policy_ckpt_dir <path-to-ckpt-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class AutoBot \
  --file_path <path-to-scenario-trajectories>
  --registered_name "trajdata_nusc_query" \
  --render
```

### 10. Cluster Scenarios

```bash
python scripts/cluster_scenarios.py \
  --file_path <path-to-scenarios-hdf5> \
  --num_clusters <num-clusters> \
  ---save_path <path-to-save-dir> \
  --viz
```

## Checkpoints

We have provided checkpoints for STD [here](https://drive.google.com/drive/folders/1QZg0M-9YiWIOOicAzKEwLAH9W9hMobdB?usp=sharing). The pre-trained models are provided under the **CC-BY-NC-SA-4.0 license**.

## Acknowledgements

This project builds on top of [CTG++](https://github.com/NVlabs/CTG) and [BITS](https://github.com/NVlabs/traffic-behavior-simulation). Thanks for their groundbreaking efforts and contributions to the field.

## Contact

If you have any problem, feel free to contact w514375542@gmail.com.