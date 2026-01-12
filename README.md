# 双足机器人强化学习运动学习项目 / Bipedal Robot RL Locomotion Learning Project

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## 概述 / Overview

该仓库用于训练和仿真双足机器人，例如[limxdynamics TRON1](https://www.limxdynamics.com/en/tron1)。
借助[Isaac Lab](https://github.com/isaac-sim/IsaacLab)框架，我们可以训练双足机器人在不同环境中行走，包括平地、粗糙地形和楼梯等。

This repository is used to train and simulate bipedal robots, such as [limxdynamics TRON1](https://www.limxdynamics.com/en/tron1).
With the help of [Isaac Lab](https://github.com/isaac-sim/IsaacLab), we can train the bipedal robots to walk in different environments, such as flat, rough, and stairs.

在本项目中基本要求来源于SDM5008课程项目：https://iyrna6v2lz.feishu.cn/wiki/XCLMwwHrpiaI60kpblwcOkspnTb

master保留了最原始的分支，能够完成任务2+3+4的代码在分支 [feature/task234new](https://github.com/Limozknight/limxtron1lab-training/tree/feature/task234new) 上

**关键词 / Keywords:** isaaclab, locomotion, bipedal, pointfoot, TRON1

## 环境配置 / Environment Initialization

- 本项目基本配置环境为 Isaac-sim 4.5 + Isaac-lab 2.1.0
- 强烈建议使用云平台 [Gradmotion](https://spaces.gradmotion.com/cloudDesktop)进行训练, 相关配置教程可查看 [官方使用手册](https://cwjgfm21di.feishu.cn/docx/Lx4jdTexeofu3kxbjh3ced6XnYe)

### 以下是官方环境提供

- 【官方】Isaaclab官网安装
  Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal. 


## quick Start

- 将仓库克隆到Isaac Lab安装目录之外的独立位置（即在`IsaacLab`目录外）：

  Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# 选项 1: 尝试原有未更改项目
git clone http://8.141.22.226/Bobbin/limxtron1lab.git

# 选项 2 ：克隆本仓库
git clone -b feature/task234new https://github.com/Limozknight/limxtron1lab-training.git your_folder_name
cd your_folder_name

```

```bash
# Enter the repository
conda activate isaaclab
cd your_folder_name
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e exts/bipedal_locomotion
```

- 为了使用MLP分支，需要安装该库 / To use the mlp branch, install the library

```bash
cd bipedal_locomotion_isaaclab/rsl_rl
python -m pip install -e .
```

初次可能会遇到问题，逐步执行：

```bash
pip install -e rsl_rl
pip uninstall rsl_rl_lib -y
pip uninstall rsl_rl -y
pip install -e rsl_rl
cd rsl_rl
python -m pip install -e .
```


## IDE设置（可选）/ Set up IDE (Optional)

要设置IDE，请按照以下说明操作：
To setup the IDE, please follow these instructions:

- 将.vscode/settings.json中的路径替换成使用者所使用的Isaaclab和python路径，这样当使用者对Isaaclab官方函数或变量进行检索的时候，可以直接跳入配置环境代码的定义。

- Replace the path in .vscode/settings.json with the Isaaclab and python paths used by the user. This way, when the user retrieves the official functions or variables of Isaaclab, they can directly jump into the definition of the configuration environment code.

### 训练

```bash
# Task 2.2: 平地速度追踪
python scripts/train.py --task Isaac-Limx-PF-Blind-Flat-v0 \
    --headless --max_iterations 3000 --run_name=Phase1_Flat

# Task 2.3: 扰动拒绝（从 Task 2.2 继续）
python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Disturbance-Rejection-v0 --headless --run_name=Task23_Push --resume True --load_run=[time_stamp]_Phase1_Flat --checkpoint=model_3000.pt

# Task 2.4: 地形遍历（从 Task 2.3 继续）
python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Stair-Training-v0 --headless --run_name=Phase3_Stairs --resume=True --load_run=[time_stamp]_Task23_Push --checkpoint=model_6000.pt
```
- 以下参数可用于自定义运行：
  The following arguments can be used to customize the playing:
    * --num_envs: 要运行的并行环境数量 / Number of parallel environments to run
    * --headless: 以无头模式运行仿真 / Run the simulation in headless mode
    * --checkpoint_path: 要加载的检查点路径 / Path to the checkpoint to load
    * --run_name: 输出文件命名 / Name ouput folder
    * --resume True/False : 是否由前期模型加载训练 / Whether training using previous model
    * --load_run ： 加载前期模型文件 / Using previous model

### 生成曲线图

```bash
# 进入输出文件夹如 pf_tron_1a_flat
tensorboard --logdir=./2026-01-11_18-19-22_Task2-3-4_stair_base_Combov2

# 点击输出本地地址网页查看
```

### 训练后运行示例

```bash
# 根目录下
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Unified-Play-v0 --load_run=2026-01-12_10-47-39_Phase3_Stairs --num_envs=32
```



## 在Mujoco中运行导出模型（仿真到仿真）/ Running exported model in mujoco (sim2sim)

- 运行模型后，策略已经保存。您可以将策略导出到mujoco环境，并参照在github开源的部署工程[tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python)在[pointfoot-mujoco-sim](https://github.com/limxdynamics/pointfoot-mujoco-sim)中运行。

  After playing the model, the policy has already been saved. You can export the policy to mujoco environment and run it in mujoco [pointfoot-mujoco-sim]((https://github.com/limxdynamics/pointfoot-mujoco-sim)) by using the [tron1-rl-deploy-python]((https://github.com/limxdynamics/tron1-rl-deploy-python)).

- 按照说明正确安装，并用您训练的`policy.onnx`和`encoder.onnx`替换原始文件。

  Following the instructions to install it properly and replace the origin policy by your trained `policy.onnx` and `encoder.onnx`.

## 在真实机器人上运行导出模型（仿真到现实）/ Running exported model in real robot (sim2real)
<p align="center">
    <img alt="Figure2 of CTS" src="./media/learning_frame.png">
</p>

**学习框架概述 / Overview of the learning framework.**

- 策略使用PPO在异步actor-critic框架内进行训练，动作由历史观察信息编码器和本体感受确定。**灵感来自论文CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion. ([H. Wang, H. Luo, W. Zhang, and H. Chen (2024)](https://doi.org/10.1109/LRA.2024.3457379))**

  The policies are trained using PPO within an asymmetric actor-critic framework, with actions determined by history observations latent and proprioceptive observation. **Inspired by the paper CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion. ([H. Wang, H. Luo, W. Zhang, and H. Chen (2024)](https://doi.org/10.1109/LRA.2024.3457379))**

- 实机部署详情见 https://support.limxdynamics.com/docs/tron-1-sdk/rl-training-results-deployment 8.1~8.2章节

  Real deployment details see section https://support.limxdynamics.com/docs/tron-1-sdk/rl-training-results-deployment 8.1 ~ 8.2


## 视频演示 / Video Demonstration

### Isaac Lab中的仿真 / Simulation in Isaac Lab
- **点足盲目平地 / Pointfoot Blind Flat**:

![play_isaaclab](./media/play_isaaclab.gif)

- **复杂地形 / Terrain Environment**:

![play_isaaclab](./media/play_isaaclab.gif)


## 📚 完整文档 / Complete Documentation

本项目包含详细的中文文档，帮助从小白到专家的所有开发者：

This project includes comprehensive Chinese documentation for developers from beginners to experts:

### 🚀 快速开始 / Quick Start

#### 新手必读（按顺序阅读）:
1. **[训练工作流指南](docs/05_Training_Workflow_Guide.md)** ⭐⭐⭐⭐⭐ - 如何启动训练、查看结果
2. **[常见问题解答](docs/06_FAQ.md)** ⭐⭐⭐⭐⭐ - 模型输出、视频录制、工时估算、文件修改、GitHub上传


#### 有经验的开发者:
1. **[架构概览](docs/01_Architecture_Overview.md)** ⭐⭐⭐⭐ - 系统架构和技术细节
2. **[项目文件结构](docs/04_Project_File_Structure.md)** ⭐⭐⭐⭐ - 完整文件树和修改优先级

### 📖 完整文档列表

#### 核心文档 / Core Documentation
- **[00_文档总览](docs/00_Documentation_Summary.md)** - 所有文档的索引
- **[01_架构概览](docs/01_Architecture_Overview.md)** - 详细的系统架构说明
- **[02_项目结构](docs/02_Project_Structure.md)** - 项目组织说明
- **[03_学习资源](docs/03_Learning_Resources.md)** - 外部学习资源
- **[04_项目文件结构](docs/04_Project_File_Structure.md)** - Tree格式的完整文件结构

#### 工作流文档 / Workflow Documentation
- **[05_训练工作流指南](docs/05_Training_Workflow_Guide.md)** - 完整训练启动和流程
- **[06_常见问题解答](docs/06_FAQ.md)** - 模型输出、视频录制、工时估算、文件修改等
- **[07_limx_base_env_cfg_QA](docs/07_limx_base_env_cfg_QA.md)** - `limx_base_env_cfg.py` 配置详解与任务改动指引

### 🎯 关键问题快速查找

- ❓ **如何启动训练？** → [05_Training_Workflow_Guide.md](docs/10_Training_Workflow_Guide.md)
- ❓ **需要修改哪些文件？** → [07_FAQ.md](docs/11_FAQ.md#q4-主要修改哪些文件)
---

## 致谢 / Acknowledgements

本项目使用以下开源库：
This project uses the following open-source libraries:
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl/tree/master)
- [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab)
- [tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab)

**贡献者 / Contributors:**
- WU Weizhi

