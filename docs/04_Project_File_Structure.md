# 项目文件结构详解

## 完整项目树状结构

```
limxtron1lab-main/
│
├── LICENCE                                    # 项目许可证文件
├── README.md                                  # 项目根目录说明
├── pyproject.toml                             # Python 项目配置（根级）
│
│
├── media/                                     # 🖼️ 媒体资源目录
│   └── [存放项目相关的图片/视频/演示]
│
├── exts/                                      # 🤖 Isaac Lab 扩展（核心）
│   │
│   └── bipedal_locomotion/                    # 双足机器人扩展包
│       │
│       ├── pyproject.toml                     # 本扩展包的 Python 配置
│       ├── setup.py                           # 安装脚本
│       │
│       ├── config/
│       │   └── extension.toml                 # Isaac Lab 扩展配置（注册扩展）
│       │
│       ├── docs/
│       │   └── CHANGELOG.rst                  # 变更日志
│       │
│       └── bipedal_locomotion/                # 👈 Python 包主目录
│           │
│           ├── __init__.py                    # 包初始化，导出主要接口
│           │
│           ├── ui_extension_example.py        # UI 扩展示例（可选）
│           │
│           ├── assets/                        # 🏗️ 机器人资产管理
│           │   ├── __init__.py                # 资产包初始化
│           │   │
│           │   ├── config/                    # 📋 机器人配置文件
│           │   │   ├── __init__.py
│           │   │   ├── pointfoot_cfg.py       # ✓ Point Foot 机器人配置
│           │   │   │   # - USD 模型路径
│           │   │   │   # - 关节定义（6 个自由度）
│           │   │   │   # - 执行器配置（刚度/阻尼）
│           │   │   │   # - 初始状态
│           │   │   │   # - 物理参数（质量/摩擦/阻尼）
│           │   │   │
│           │   │   ├── solefoot_cfg.py        # ✓ Sole Foot 机器人配置
│           │   │   │   # - 类似 Point Foot，但足部类型不同
│           │   │   │   # - 可能有脚趾关节
│           │   │   │   # - 不同的接触模型
│           │   │   │
│           │   │   └── wheelfoot_cfg.py       # ✓ Wheel Foot 机器人配置
│           │   │       # - 轮足配置
│           │   │       # - 滚动动力学参数
│           │   │
│           │   └── usd/                       # 🎨 USD 3D 模型资源
│           │       │
│           │       ├── PF_TRON1A/             # Point Foot 模型
│           │       │   ├── config.yaml        # 模型配置参数
│           │       │   ├── PF_TRON1A.usd      # 主要 USD 文件
│           │       │   └── configuration/     # 模型组件
│           │       │       ├── PF_TRON1A_base.usd      # 基座刚体
│           │       │       ├── PF_TRON1A_physics.usd   # 物理参数
│           │       │       └── PF_TRON1A_sensor.usd    # 传感器定义
│           │       │
│           │       ├── SF_TRON1A/             # Sole Foot 模型（结构相同）
│           │       │   ├── config.yaml
│           │       │   ├── SF_TRON1A.usd
│           │       │   └── configuration/
│           │       │       ├── SF_TRON1A_base.usd
│           │       │       ├── SF_TRON1A_physics.usd
│           │       │       └── SF_TRON1A_sensor.usd
│           │       │
│           │       └── WF_TRON1A/             # Wheel Foot 模型（结构相同）
│           │           ├── config.yaml
│           │           ├── WF_TRON1A.usd
│           │           └── configuration/
│           │               ├── WF_TRON1A_base.usd
│           │               ├── WF_TRON1A_physics.usd
│           │               └── WF_TRON1A_sensor.usd
│           │
│           ├── tasks/                         # 🎯 强化学习任务定义
│           │   ├── __init__.py                # 任务包初始化
│           │   │
│           │   └── locomotion/                # 👣 移动任务（主要）
│           │       ├── __init__.py            # 导出任务类
│           │       │
│           │       ├── agents/                # 🧠 策略网络配置
│           │       │   ├── __init__.py
│           │       │   └── limx_rsl_rl_ppo_cfg.py  # PPO Actor-Critic 网络
│           │       │       # - Actor 网络：输入观测，输出动作
│           │       │       # - Critic 网络：输入观测，输出价值
│           │       │       # - 网络深度、隐藏层大小等
│           │       │
│           │       ├── cfg/                   # 📋 任务配置文件
│           │       │   ├── __init__.py
│           │       │   ├── PF/                # Point Foot 任务配置
│           │       │   │   └── limx_base_env_cfg.py   # ⭐ 主要配置文件
│           │       │   │       # 包含:
│           │       │   │       # 1. 场景配置 (Scene)
│           │       │   │       #    - 地形类型
│           │       │   │       #    - 机器人模型
│           │       │   │       #    - 传感器配置
│           │       │   │       #
│           │       │   │       # 2. 观测配置 (Observations)
│           │       │   │       #    - 政策观测项 (Policy Obs)
│           │       │   │       #    - 特权观测项 (Privileged Obs)
│           │       │   │       #    - 噪声配置
│           │       │   │       #
│           │       │   │       # 3. 奖励配置 (Rewards)
│           │       │   │       #    - 奖励项权重
│           │       │   │       #    - 奖励函数参数
│           │       │   │       #
│           │       │   │       # 4. 动作配置 (Actions)
│           │       │   │       #    - 动作空间定义
│           │       │   │       #    - PD 控制参数
│           │       │   │       #
│           │       │   │       # 5. 命令生成器配置 (Commands)
│           │       │   │       #    - 速度命令范围
│           │       │   │       #    - 步态命令范围
│           │       │   │
│           │       │   ├── SF/                # Sole Foot 任务配置
│           │       │   │   └── limx_base_env_cfg.py   # 结构类似
│           │       │   │
│           │       │   └── WF/                # Wheel Foot 任务配置
│           │       │       └── limx_base_env_cfg.py   # 结构类似
│           │       │
│           │       ├── mdp/                   # 🔄 MDP 组件（核心算法）
│           │       │   ├── __init__.py        # 导出所有 MDP 组件
│           │       │   │
│           │       │   ├── observations.py    # 👁️ 观测函数实现
│           │       │   │   # 实现的函数：
│           │       │   │   # - projected_gravity()      投影重力向量
│           │       │   │   # - base_ang_vel()           基座角速度
│           │       │   │   # - joint_pos_rel()          相对关节位置
│           │       │   │   # - joint_vel()              关节速度
│           │       │   │   # - last_action()            上一步动作
│           │       │   │   # - get_gait_phase()         步态相位
│           │       │   │   # - get_gait_command()       步态命令
│           │       │   │   # - height_scan()            高度扫描
│           │       │   │   # - robot_joint_torque()     关节扭矩（教师用）
│           │       │   │   # - robot_joint_acc()        关节加速度（教师用）
│           │       │   │   # - feet_lin_vel()           足部线速度（教师用）
│           │       │   │
│           │       │   ├── rewards.py         # 🎁 奖励函数实现
│           │       │   │   # 实现的函数/类：
│           │       │   │   # - stay_alive()             存活奖励
│           │       │   │   # - base_tracking()          基座速度追踪
│           │       │   │   # - GaitReward               步态奖励类
│           │       │   │   # - feet_regulation()        足部调节奖励
│           │       │   │   # - ActionSmoothnessPenalty  动作平滑惩罚
│           │       │   │   # - contact_penalty()        接触惩罚
│           │       │   │   # - 权重组合逻辑
│           │       │   │
│           │       │   ├── actions.py         # 🕹️ 动作处理
│           │       │   │   # 实现的类/函数：
│           │       │   │   # - JointPositionActionCfg   关节位置动作
│           │       │   │   # - 动作缩放和映射
│           │       │   │   # - PD 控制器集成
│           │       │   │
│           │       │   ├── curriculums.py     # 📚 课程学习配置
│           │       │   │   # 实现的逻辑：
│           │       │   │   # - 难度级别定义
│           │       │   │   # - 命令范围递增
│           │       │   │   # - 域随机化参数调度
│           │       │   │   # - 进度判断函数
│           │       │   │
│           │       │   ├── events.py          # 📡 环境事件回调
│           │       │   │   # 实现的事件：
│           │       │   │   # - reset_base()             重置基座位置
│           │       │   │   # - add_base_mass()          增加基座质量
│           │       │   │   # - randomize_friction()     随机摩擦系数
│           │       │   │   # - randomize_stiffness()    随机刚度
│           │       │   │   # - 其他域随机化事件
│           │       │   │
│           │       │   └── commands/          # 💬 命令生成器
│           │       │       ├── __init__.py
│           │       │       ├── velocity_commands.py      # 速度命令
│           │       │       └── gait_commands.py          # 步态命令
│           │       │
│           │       └── robots/                # 🦾 机器人环境类
│           │           ├── __init__.py        # 导出环境类
│           │           ├── limx_pointfoot_env_cfg.py      # Point Foot 环境配置
│           │           │   # - 组合 Scene + MDP 配置
│           │           │   # - 任务特定参数
│           │           ├── limx_solefoot_env_cfg.py       # Sole Foot 环境配置
│           │           └── limx_wheelfoot_env_cfg.py      # Wheel Foot 环境配置
│           │
│           └── utils/                         # 🛠️ 工具函数
│               ├── __init__.py
│               └── wrappers/                  # 🔌 环境包装器
│                   └── rsl_rl/                # RSL-RL 框架集成
│                       ├── __init__.py
│                       └── rl_mlp_cfg.py      # MLP 神经网络配置
│                           # - 网络层数
│                           # - 激活函数
│                           # - 初始化方式
│
├── rsl_rl/                                    # 🎓 强化学习算法库（核心）
│   ├── __init__.py                            # 包初始化
│   ├── pyproject.toml                         # 本库的 Python 配置
│   ├── setup.py                               # 安装脚本
│   │
│   ├── licenses/                              # 📜 依赖许可证
│   │   └── dependencies/                      # 第三方库许可证集合
│   │       ├── black-license.txt              # 代码格式化工具
│   │       ├── codespell-license.txt          # 拼写检查
│   │       ├── flake8-license.txt             # 风格检查
│   │       ├── isort-license.txt              # 导入排序
│   │       ├── numpy_license.txt              # 数值计算
│   │       ├── onnx-license.txt               # 模型格式
│   │       ├── pre-commit-hooks-license.txt   # Git 钩子
│   │       ├── pre-commit-license.txt         # Git 钩子框架
│   │       ├── pyright-license.txt            # 类型检查
│   │       ├── pyupgrade-license.txt          # Python 升级
│   │       └── torch_license.txt              # PyTorch
│   │
│   └── rsl_rl/                                # 👈 Python 包主目录
│       ├── __init__.py                        # 包初始化
│       │
│       ├── algorithm/                         # 🧠 强化学习算法实现
│       │   ├── __init__.py
│       │   └── ppo.py                         # ⭐ PPO 算法核心
│       │       # 实现：
│       │       # - PPO 损失函数
│       │       #   - Actor 损失（策略梯度）
│       │       #   - Critic 损失（价值函数）
│       │       #   - 熵正则化项
│       │       # - 优势估计（GAE）
│       │       # - 数据采样和存储
│       │       # - 参数更新逻辑
│       │
│       ├── env/                               # 🌍 环境包装接口
│       │   ├── __init__.py
│       │   └── vec_env.py                     # 向量化环境
│       │       # 实现：
│       │       # - 多环境并行处理
│       │       # - 统一接口
│       │       # - 批处理优化
│       │
│       ├── modules/                           # 🧩 神经网络模块
│       │   ├── __init__.py
│       │   ├── actor_critic.py                # Actor-Critic 网络
│       │   │   # 实现：
│       │   │   # - Actor 子网络（策略网络）
│       │   │   # - Critic 子网络（价值网络）
│       │   │   # - 共享编码器
│       │   │   # - 前向传播
│       │   │
│       │   └── mlp_encoder.py                 # MLP 编码器
│       │       # 实现：
│       │       # - 多层感知机
│       │       # - 激活函数选择
│       │       # - Layer normalization
│       │
│       ├── runner/                            # 🏃 训练循环
│       │   ├── __init__.py
│       │   └── on_policy_runner.py            # ⭐ 同步策略训练器
│       │       # 实现：
│       │       # - 主训练循环
│       │       #   1. 数据收集（rollout）
│       │       #   2. PPO 更新（mini-batch SGD）
│       │       #   3. 日志记录
│       │       # - 检查点保存/加载
│       │       # - 学习率调度
│       │       # - 观测归一化
│       │
│       └── storage/                           # 💾 数据存储
│           ├── __init__.py
│           └── rollout_storage.py             # 轨迹存储缓冲
│               # 实现：
│               # - 存储经验元组 (s, a, r, s', d)
│               # - GAE 计算
│               # - mini-batch 采样
│               # - 内存管理
│
└── scripts/                                   # 🚀 执行脚本（入口点）
    │
    ├── rsl_rl/                                # RSL-RL 训练脚本
    │   ├── cli_args.py                        # 命令行参数解析
    │   │   # 定义：
    │   │   # - --task: 任务名称
    │   │   # - --headless: 无头模式
    │   │   # - --checkpoint: 检查点路径
    │   │   # - 其他超参数
    │   │
    │   ├── train.py                           # ⭐ 训练脚本入口
    │   │   # 使用：
    │   │   # $ python train.py --task=PointFootLocomotion --headless
    │   │   # 功能：
    │   │   # 1. 加载任务配置
    │   │   # 2. 初始化环境和网络
    │   │   # 3. 运行 on_policy_runner
    │   │   # 4. 保存模型检查点
    │   │
    │   └── play.py                            # ⭐ 推理脚本
    │       # 使用：
    │       # $ python play.py --task=PointFootLocomotion --checkpoint=logs/.../model.pt
    │       # 功能：
    │       # 1. 加载已训练模型
    │       # 2. 运行推理（不更新参数）
    │       # 3. 可视化机器人行为
    │       # 4. 接收实时命令
    │
    └── [其他实用脚本]

```

---

## 关键文件详解

### 🔴 最重要的文件（必读）

#### 1. **`exts/bipedal_locomotion/bipedal_locomotion/assets/config/pointfoot_cfg.py`**
```
功能：定义机器人的硬件配置
包含：
  ✓ USD 模型路径
  ✓ 关节名称和初始状态
  ✓ 执行器配置（刚度 K_p=25.0，阻尼 K_d=0.8）
  ✓ 物理参数（最大力矩、最大速度）

修改场景：当需要改变机器人的物理特性时
```

#### 2. **`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py`**
```
功能：主环境配置文件（最复杂）
包含：
  ✓ 场景定义（地形、光照、传感器）
  ✓ 观测空间（59维）和噪声配置
  ✓ 奖励函数（7个奖励项和权重）
  ✓ 动作空间（6维关节位置）
  ✓ 命令生成器（速度和步态命令）

修改场景：调整任务难度、奖励权重、观测噪声等

⭐ 这是任务修改的主要文件
```

#### 3. **`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/observations.py`**
```
功能：观测函数的具体实现
包含：
  ✓ 投影重力、基座速度等基础观测
  ✓ 关节位置和速度测量
  ✓ 步态相位计算
  ✓ 教师网络的特权信息

修改场景：添加新的观测项或修改噪声模型
```

#### 4. **`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py`**
```
功能：奖励函数的具体实现
包含：
  ✓ 速度追踪奖励
  ✓ 步态奖励
  ✓ 动作平滑性惩罚
  ✓ 其他稳定性奖励

修改场景：微调奖励逻辑或添加新奖励项

⭐ 用于优化任务 2.2/2.3/2.4
```

#### 5. **`rsl_rl/rsl_rl/algorithm/ppo.py`**
```
功能：PPO 算法核心实现
包含：
  ✓ PPO 损失函数计算
  ✓ 参数更新逻辑
  ✓ 熵正则化

修改场景：一般不需要改，除非调整 PPO 超参数
```

#### 6. **`rsl_rl/rsl_rl/runner/on_policy_runner.py`**
```
功能：训练主循环
包含：
  ✓ 数据收集（rollout）
  ✓ PPO 更新步骤
  ✓ 日志和检查点保存

修改场景：修改训练流程或添加自定义回调
```

#### 7. **`scripts/rsl_rl/train.py`** & **`scripts/rsl_rl/play.py`**
```
train.py 功能：
  ✓ 训练脚本入口
  ✓ 使用：python train.py --task=PointFootLocomotion --headless

play.py 功能：
  ✓ 推理脚本（运行已训练模型）
  ✓ 使用：python play.py --task=PointFootLocomotion --checkpoint=...

修改场景：添加自定义命令接口或评估逻辑
```

---

## 文件之间的依赖关系

```
train.py / play.py
    ↓
    ├─→ PointFootLocomotion (任务定义)
    │   ├─→ limx_base_env_cfg.py (主配置)
    │   │   ├─→ pointfoot_cfg.py (机器人配置)
    │   │   ├─→ observations.py (观测函数)
    │   │   ├─→ rewards.py (奖励函数)
    │   │   └─→ actions.py (动作处理)
    │   │
    │   └─→ on_policy_runner.py (训练循环)
    │       └─→ ppo.py (PPO 算法)
    │
    ├─→ Actor-Critic Network
    │   └─→ actor_critic.py + mlp_encoder.py
    │
    ├─→ Environments (4096 个并行)
    │   └─→ vec_env.py
    │
    └─→ Rollout Storage
        └─→ rollout_storage.py
```

---

## 文件修改优先级（用于任务 2.2-2.4）

### 优先级 1️⃣（必改）
```
┌─────────────────────────────────────────────────────┐
│ 1. rewards.py                                       │
│    └─ 调整奖励权重以改进任务表现                   │
│    └─ 添加新的奖励项（如抗干扰奖励）               │
│                                                     │
│ 2. limx_base_env_cfg.py                            │
│    └─ 修改观测/动作/场景配置                       │
│    └─ 启用/禁用 Domain Randomization               │
└─────────────────────────────────────────────────────┘
```

### 优先级 2️⃣（可能需要）
```
┌─────────────────────────────────────────────────────┐
│ 1. observations.py                                  │
│    └─ 添加特定观测项（如地形高度）                 │
│    └─ 调整噪声模型                                 │
│                                                     │
│ 2. curriculums.py                                  │
│    └─ 设计课程学习策略                             │
│    └─ 递增难度等级                                 │
└─────────────────────────────────────────────────────┘
```

### 优先级 3️⃣（高级）
```
┌─────────────────────────────────────────────────────┐
│ 1. train.py / play.py                              │
│    └─ 添加自定义评估指标                           │
│    └─ 集成命令接收接口                             │
│                                                     │
│ 2. actions.py                                      │
│    └─ 修改 PD 参数                                 │
│    └─ 调整动作缩放                                 │
└─────────────────────────────────────────────────────┘
```

---

## 文件大小参考

```
核心算法实现：
  ppo.py                  ~500 行  (PPO 算法)
  on_policy_runner.py     ~800 行  (训练循环)
  actor_critic.py         ~300 行  (网络结构)

任务定义：
  limx_base_env_cfg.py    ~700 行  (环境配置)
  observations.py         ~600 行  (观测函数)
  rewards.py              ~500 行  (奖励函数)

配置文件：
  pointfoot_cfg.py        ~150 行  (机器人配置)
  limx_rsl_rl_ppo_cfg.py  ~100 行  (网络配置)

执行脚本：
  train.py                ~200 行
  play.py                 ~150 行
```

---

## 快速导航

| 我想做... | 应该打开 |
|---------|----------|
| 改变机器人硬件参数 | `pointfoot_cfg.py` |
| 调整奖励权重 | `rewards.py` + `limx_base_env_cfg.py` |
| 添加新的观测 | `observations.py` + `limx_base_env_cfg.py` |
| 修改 PD 控制参数 | `pointfoot_cfg.py` (actuators) |
| 调整动作空间 | `limx_base_env_cfg.py` (actions) |
| 启用地形随机化 | `limx_base_env_cfg.py` + `curriculums.py` |
| 修改训练循环 | `on_policy_runner.py` |
| 改变 PPO 算法 | `ppo.py` |
| 添加自定义传感器 | `observations.py` + `limx_base_env_cfg.py` |
| 实现推理接口 | `play.py` |

---

## 文件编辑检查清单

### 当编辑配置文件时，检查：
- [ ] 数据类型是否正确（int/float/bool）
- [ ] 数值范围是否合理
- [ ] 所有引用的函数是否存在
- [ ] 缩进是否正确（Python 对缩进敏感）

### 当编辑函数时，检查：
- [ ] 输入/输出维度是否匹配
- [ ] 是否处理 CUDA/CPU 设备切换
- [ ] 是否需要梯度计算
- [ ] 是否有浮点精度问题

### 当修改奖励时，检查：
- [ ] 权重总和是否过大（>5）
- [ ] 是否所有奖励项都在合理范围
- [ ] 是否有奖励项相互冲突

---

## 相关文件总数统计

```
总计：
  ├─ Python 文件: ~25 个
  ├─ 配置文件: ~15 个
  ├─ USD 模型: ~3 个（每个模型有 4-5 个子组件）
  ├─ 文档文件: ~6 个
  └─ 脚本文件: ~3 个

行数统计：
  ├─ 算法代码: ~3,000 行
  ├─ 任务定义: ~2,500 行
  ├─ 配置代码: ~1,000 行
  └─ 文档: ~5,000 行
```

