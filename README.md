# Finetuning Offline World Models in the Real World

Official PyTorch implementation of [Finetuning Offline World Models in the Real World](https://yunhaifeng.com/FOWM) (CoRL 2023 Oral)

[Paper](https://arxiv.org/abs/2310.16029) | [Website](https://yunhaifeng.com/FOWM) | [Dataset (sim)](https://drive.google.com/file/d/1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya/view?usp=sharing) | [Dataset (real)](https://drive.google.com/file/d/1PRCqANEOV0SICLEWEvUL9AnyJOe2UMYK/view?usp=sharing)

![Framework](figures/teaser.png)

## Installation


Install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate fowm
```

## Training

After installing dependencies, you can train an agent by
```
python src/train_off2on.py task=antmaze-medium-play-v2
```
Supported tasks from [D4RL](https://github.com/Farama-Foundation/D4RL): `antmaze-medium-play-v2`, `antmaze-medium-diverse-v2`, `hopper-medium-v2`, `hopper-medium-replay-v2`.

To run experiments on xArm tasks, first download our released offline datasets
```
python scripts/download_datasets.py
```
Datasets will be saved at the directory `data`:
```
data
├── xarm_lift_medium
├── xarm_lift_medium_replay
├── xarm_push_medium
└── xarm_push_medium_replay
```

Then start training with 
```
python src/train_off2on.py modality=all task=xarm_lift dataset_dir=data/xarm_lift_medium_replay
```
You can choose `xarm_lift` or `xarm_push` as `task` and use `dataset_dir` to specify the offline dataset.

The training script supports both local logging as well as cloud-based logging with [Weights & Biases](https://wandb.ai). To use W&B, provide a key by setting the environment variable `WANDB_API_KEY=<YOUR_KEY>` and add your W&B project and entity details to `cfgs/config.yaml`.

## Citation
If you find our work useful in your research, please consider citing with the following BibTeX:
```
@inproceedings{feng2023finetuning,
  title={Finetuning Offline World Models in the Real World},
  author={Feng, Yunhai and Hansen, Nicklas and Xiong, Ziyan and Rajagopalan, Chandramouli and Wang, Xiaolong},
  booktitle={Proceedings of the 7th Conference on Robot Learning (CoRL)},
  year={2023}
}
```

## License & Acknowledgements
This repository is licensed under the MIT license. The codebase is based on the original implementations of [TD-MPC](https://github.com/nicklashansen/tdmpc). 
