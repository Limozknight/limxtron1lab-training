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

**关键词 / Keywords:** isaaclab, locomotion, bipedal, pointfoot, TRON1

## 安装 / Installation

- 【非官方】强烈推荐使用一键安装脚本(pip)！

   本脚本同时支持Isaacsim v1.4.1和v2.x.x版本。一键脚本可直接安装Isaacsim、Isaaclab以及配套miniconda虚拟环境，已在ubuntu22.04与20.04测试通过，在终端中执行以下命令：
   ```bash
   wget -O install_isaaclab.sh https://docs.robotsfan.com/install_isaaclab.sh && bash install_isaaclab.sh
   ```

   感谢一键安装脚本作者[@fan-ziqi](https://github.com/fan-ziqi)，感谢大佬为机器人社区所做的贡献。该仓库所使用Isaacsim版本为2.1.0，使用一键脚本安装时请选择该版本。

- 【官方】Isaaclab官网安装
  Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal. 


- 将仓库克隆到Isaac Lab安装目录之外的独立位置（即在`IsaacLab`目录外）：

  Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# 选项 1: HTTPS / Option 1: HTTPS
git clone http://8.141.22.226/Bobbin/limxtron1lab.git

# 选项 2: SSH
git clone git@8.141.22.226:Bobbin/limxtron1lab.git
```

```bash
# Enter the repository
conda activate isaaclab
cd bipedal_locomotion_isaaclab
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

## IDE设置（可选）/ Set up IDE (Optional)

要设置IDE，请按照以下说明操作：
To setup the IDE, please follow these instructions:

- 将.vscode/settings.json中的路径替换成使用者所使用的Isaaclab和python路径，这样当使用者对Isaaclab官方函数或变量进行检索的时候，可以直接跳入配置环境代码的定义。

- Replace the path in .vscode/settings.json with the Isaaclab and python paths used by the user. This way, when the user retrieves the official functions or variables of Isaaclab, they can directly jump into the definition of the configuration environment code.

## 训练双足机器人智能体 / Training the bipedal robot agent

- 使用`scripts/rsl_rl/train.py`脚本直接训练机器人，指定任务：
  Use the `scripts/rsl_rl/train.py` script to train the robot directly, specifying the task:

```bash
python3 scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Flat-v0 --headless
```

- 以下参数可用于自定义训练：
  The following arguments can be used to customize the training:
    * --headless: 以无渲染模式运行仿真 / Run the simulation in headless mode
    * --num_envs: 要运行的并行环境数量 / Number of parallel environments to run
    * --max_iterations: 最大训练迭代次数 / Maximum number of training iterations
    * --save_interval: 保存模型的间隔 / Interval to save the model
    * --seed: 随机数生成器的种子 / Seed for the random number generator

## 运行训练好的模型 / Playing the trained model

- 要运行训练好的模型：
  To play a trained model:

```bash
python3 scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint
```

- 以下参数可用于自定义运行：
  The following arguments can be used to customize the playing:
    * --num_envs: 要运行的并行环境数量 / Number of parallel environments to run
    * --headless: 以无头模式运行仿真 / Run the simulation in headless mode
    * --checkpoint_path: 要加载的检查点路径 / Path to the checkpoint to load

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
### Mujoco中的仿真 / Simulation in Mujoco
- **点足盲目平地 / Pointfoot Blind Flat**:

![play_mujoco](./media/play_mujoco.gif)

### 真实机器人部署 / Deployment in Real Robot
- **点足盲目平地 / Pointfoot Blind Flat**:

![play_mujoco](./media/rl_real.gif)

## 📚 完整文档 / Complete Documentation

本项目包含详细的中文文档，帮助从小白到专家的所有开发者：

This project includes comprehensive Chinese documentation for developers from beginners to experts:

### 🚀 快速开始 / Quick Start

#### 新手必读（按顺序阅读）:
1. **[训练工作流指南](docs/10_Training_Workflow_Guide.md)** ⭐⭐⭐⭐⭐ - 如何启动训练、查看结果
2. **[常见问题解答](docs/11_FAQ.md)** ⭐⭐⭐⭐⭐ - 模型输出、视频录制、工时估算、文件修改、GitHub上传
3. **[任务 2.2 详细指南](docs/07_Task2.2_Detailed_Guide.md)** ⭐⭐⭐⭐⭐ - 平地速度跟随任务（初学者友好）

#### 有经验的开发者:
1. **[架构概览](docs/01_Architecture_Overview.md)** ⭐⭐⭐⭐ - 系统架构和技术细节
2. **[项目文件结构](docs/05_Project_File_Structure.md)** ⭐⭐⭐⭐ - 完整文件树和修改优先级

### 📖 完整文档列表

#### 核心文档 / Core Documentation
- **[00_文档总览](docs/00_Documentation_Summary.md)** - 所有文档的索引
- **[01_架构概览](docs/01_Architecture_Overview.md)** - 详细的系统架构说明
- **[02_项目结构](docs/02_Project_Structure.md)** - 项目组织说明
- **[03_任务实现指南](docs/03_Task_Implementation_Guide.md)** - 如何实现新任务
- **[04_学习资源](docs/04_Learning_Resources.md)** - 外部学习资源
- **[05_项目文件结构](docs/05_Project_File_Structure.md)** - Tree格式的完整文件结构

#### 任务指南 / Task Guides (按推荐顺序)
- **[07_任务2.2详细指南](docs/07_Task2.2_Detailed_Guide.md)** - 平地速度跟随（基础任务）
- **[09_任务2.3详细指南](docs/09_Task2.3_Detailed_Guide.md)** - 抗干扰鲁棒性（中级任务）
- **[08_任务2.4详细指南](docs/08_Task2.4_Detailed_Guide.md)** - 复杂地形穿越（高级任务）

#### 工作流文档 / Workflow Documentation
- **[10_训练工作流指南](docs/10_Training_Workflow_Guide.md)** - 完整训练启动和流程
- **[11_常见问题解答](docs/11_FAQ.md)** - 模型输出、视频录制、工时估算、文件修改等
- **[12_limx_base_env_cfg_QA](docs/12_limx_base_env_cfg_QA.md)** - `limx_base_env_cfg.py` 配置详解与任务改动指引

### 🎯 关键问题快速查找

- ❓ **如何启动训练？** → [10_Training_Workflow_Guide.md](docs/10_Training_Workflow_Guide.md)
- ❓ **训练会输出模型文件吗？** → [11_FAQ.md](docs/11_FAQ.md#q1-任务-22-和-23-会输出模型文件吗)
- ❓ **支持录制视频吗？** → [11_FAQ.md](docs/11_FAQ.md#q2-训练命令支持输出-video-吗)
- ❓ **需要多少工时？** → [11_FAQ.md](docs/11_FAQ.md#q3-完成这几个任务大概需要几个工时)
- ❓ **需要修改哪些文件？** → [11_FAQ.md](docs/11_FAQ.md#q4-主要修改哪些文件)
- ❓ **如何上传到 GitHub？** → [11_FAQ.md](docs/11_FAQ.md#q5-如何将项目上传到-github)
- ❓ **任务有依赖关系吗？** → [10_Training_Workflow_Guide.md](docs/10_Training_Workflow_Guide.md#5-任务依赖关系分析)
- ❓ **不同任务的完成路径？** → [10_Training_Workflow_Guide.md](docs/10_Training_Workflow_Guide.md#6-三种完成路径建议)

---

## 致谢 / Acknowledgements

本项目使用以下开源库：
This project uses the following open-source libraries:
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl/tree/master)
- [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab)
- [tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab)

**贡献者 / Contributors:**
- Hongwei Xiong 
- Bobin Wang
- Wen
- Haoxiang Luo
- Junde Guo

