# 项目结构与组织

> 深入理解 LIMX TRON1A 项目的文件组织、配置继承关系和数据流

## 项目总体架构

```
limxtron1lab-main/
│
├── exts/                                    # Isaac Lab 扩展模块
│   └── bipedal_locomotion/                 # 双足机器人扩展
│       ├── pyproject.toml                  # 扩展配置
│       ├── setup.py                        # 安装脚本
│       ├── config/
│       │   └── extension.toml              # Isaac 扩展注册信息
│       │
│       └── bipedal_locomotion/             # 主要代码包
│           ├── __init__.py
│           ├── ui_extension_example.py     # 可视化界面（可选）
│           │
│           ├── assets/                     # ★ 场景配置
│           │   ├── __init__.py
│           │   ├── config/
│           │   │   ├── pointfoot_cfg.py    # PF 机器人配置
│           │   │   ├── solefoot_cfg.py     # SF 机器人配置
│           │   │   └── wheelfoot_cfg.py    # WF 机器人配置
│           │   │
│           │   └── usd/                    # USD 模型资产
│           │       ├── PF_TRON1A/
│           │       │   ├── PF_TRON1A.usd               # 完整模型
│           │       │   ├── config.yaml                 # 配置参数
│           │       │   └── configuration/
│           │       │       ├── PF_TRON1A_base.usd      # 几何体
│           │       │       ├── PF_TRON1A_physics.usd   # 物理
│           │       │       └── PF_TRON1A_sensor.usd    # 传感器
│           │       ├── SF_TRON1A/
│           │       └── WF_TRON1A/
│           │
│           ├── tasks/                     # ★ 任务定义
│           │   └── locomotion/
│           │       ├── __init__.py
│           │       ├── agents/            # 策略网络配置
│           │       │   ├── __init__.py
│           │       │   └── limx_rsl_rl_ppo_cfg.py  # PPO 网络架构
│           │       │
│           │       ├── cfg/               # ★ 环境配置（关键）
│           │       │   ├── __init__.py
│           │       │   ├── PF/
│           │       │   │   ├── __init__.py
│           │       │   │   ├── limx_base_env_cfg.py      # 基础配置
│           │       │   │   ├── limx_flat_env_cfg.py      # 平地配置
│           │       │   │   ├── limx_rough_env_cfg.py     # 复杂地形配置
│           │       │   │   └── [其他任务配置]
│           │       │   ├── SF/
│           │       │   └── WF/
│           │       │
│           │       └── mdp/               # ★ 决策过程（观测/奖励/动作）
│           │           ├── __init__.py
│           │           ├── observations.py    # 观测函数库
│           │           ├── rewards.py         # 奖励函数库
│           │           ├── actions.py         # 动作处理器
│           │           ├── curriculums.py     # 课程学习策略
│           │           ├── events.py          # 环境事件（摔倒等）
│           │           └── commands/
│           │               └── [命令生成器]
│           │
│           └── utils/                     # 辅助工具
│               └── wrappers/
│                   └── rsl_rl/
│                       ├── __init__.py
│                       └── rl_mlp_cfg.py  # MLP 网络配置
│
├── rsl_rl/                                 # ★ RSL_RL 强化学习库
│   ├── pyproject.toml
│   ├── setup.py
│   ├── licenses/                          # 许可证
│   │
│   └── rsl_rl/
│       ├── __init__.py
│       │
│       ├── algorithm/
│       │   ├── __init__.py
│       │   └── ppo.py                    # PPO 算法实现
│       │
│       ├── env/
│       │   ├── __init__.py
│       │   └── vec_env.py                # 向量化环境包装
│       │
│       ├── modules/                       # 神经网络模块
│       │   ├── __init__.py
│       │   ├── actor_critic.py           # Actor-Critic 架构
│       │   └── mlp_encoder.py            # MLP 编码器
│       │
│       ├── runner/
│       │   ├── __init__.py
│       │   └── on_policy_runner.py       # 训练主循环
│       │
│       └── storage/
│           ├── __init__.py
│           └── rollout_storage.py        # 轨迹缓冲存储
│
├── scripts/                               # ★ 可执行脚本
│   └── rsl_rl/
│       ├── cli_args.py                   # 命令行参数
│       ├── train.py                      # 训练脚本（入口）
│       └── play.py                       # 推理脚本
│
├── media/                                 # 媒体资源
│   └── [图片/视频等]
│
├── LICENCE                                # 许可证
├── pyproject.toml                         # 顶级项目配置
└── README.md                              # 项目说明
```

---

## 关键文件详解

### 1. 配置文件层级关系

```
bipedal_locomotion/assets/config/pointfoot_cfg.py
│
└─→ 定义机器人关节、执行器、质量等底层参数


bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py
│
├─→ 继承: ManagerBasedRLEnvCfg
├─→ 包含:
│   ├── scene: 使用 pointfoot_cfg.py 中的 POINTFOOT_CFG
│   ├── observations: 定义观测空间
│   ├── rewards: 定义奖励函数
│   ├── actions: 定义动作处理
│   └── commands: 定义速度/步态命令
│
└─→ limx_flat_env_cfg.py (继承 limx_base_env_cfg.py)
    └─→ 重写: 地形/任务特定参数


bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py
│
└─→ 定义 Actor-Critic 网络架构


scripts/rsl_rl/train.py
│
└─→ 加载所有配置，启动训练
    ├── import env_cfg
    ├── import agent_cfg
    ├── 创建环境
    ├── 创建 PPO 算法
    └── 运行训练循环
```

### 2. 配置导入流程

```python
# train.py 的简化流程

from bipedal_locomotion.tasks.locomotion.cfg import PF_BASE_ENV_CFG  # 环境配置
from bipedal_locomotion.tasks.locomotion.agents import LIMX_PPO_CFG    # 网络配置

# PF_BASE_ENV_CFG 的完整组成:
PF_BASE_ENV_CFG.scene
  ├── terrain: TerrainImporterCfg
  ├── robot: ArticulationCfg (来自 pointfoot_cfg.POINTFOOT_CFG)
  │   └── actuators:
  │       └── ImplicitActuatorCfg(stiffness=25.0, damping=0.8, ...)
  ├── sensors: ContactSensorCfg, RayCasterCfg
  └── light: DomeLightCfg

PF_BASE_ENV_CFG.observations
  ├── policy: 策略观测 (59 维)
  │   ├── proj_gravity
  │   ├── base_ang_vel
  │   ├── joint_pos
  │   ├── joint_vel
  │   └── ...
  └── history: 教师观测 (80 维)

PF_BASE_ENV_CFG.rewards
  ├── stay_alive (权重: 0.5)
  ├── base_tracking (权重: 1.0)
  ├── gait_reward (权重: 0.5)
  └── ...

PF_BASE_ENV_CFG.actions
  └── joint_pos: 6 维 (关节位置控制)
```

---

## 数据流与执行流程

### 完整的训练循环

```
┌─────────────────────────────────────────────────────────────┐
│                      scripts/train.py                       │
│                   (主训练脚本 - 入口点)                     │
└────┬────────────────────────────────────────────────────────┘
     │
     ├─→ Step 1: 加载配置
     │   ├── env_cfg = PF_BASE_ENV_CFG (from cfg/PF/)
     │   └── agent_cfg = LIMX_PPO_CFG (from agents/)
     │
     ├─→ Step 2: 创建环境
     │   ├── env = ManagerBasedRLEnv(env_cfg)
     │   │   └── 初始化场景 (scene_cfg)
     │   │       ├── 加载 USD 模型 (assets/usd/)
     │   │       ├── 创建 4096 个并行环境
     │   │       └── 初始化传感器
     │   │
     │   ├── obs_manager = ObservationManager(obs_cfg)
     │   │   ├── 注册 projection_gravity()
     │   │   ├── 注册 joint_pos_rel()
     │   │   └── [来自 mdp/observations.py]
     │   │
     │   ├── rew_manager = RewardManager(rew_cfg)
     │   │   ├── 注册 stay_alive()
     │   │   ├── 注册 base_tracking()
     │   │   └── [来自 mdp/rewards.py]
     │   │
     │   └── act_manager = ActionManager(act_cfg)
     │       └── [来自 mdp/actions.py]
     │
     ├─→ Step 3: 创建策略网络
     │   ├── actor = MLP(in=59, hidden=256, out=6)
     │   │   └── [来自 agents/limx_rsl_rl_ppo_cfg.py]
     │   │
     │   └── critic = MLP(in=59, hidden=256, out=1)
     │       └── [来自 modules/actor_critic.py]
     │
     ├─→ Step 4: 创建 PPO 算法
     │   └── ppo = PPO(actor, critic, lr=1e-4)
     │       └── [来自 algorithm/ppo.py]
     │
     └─→ Step 5: 主训练循环 (on_policy_runner.py)
         │
         ├─ Episode Loop (n=2500 步/episode)
         │  │
         │  ├─ reset: env.reset()
         │  │  └── 随机初始化机器人位置/方向
         │  │
         │  ├─ Step Loop (2500 次)
         │  │  │
         │  │  ├─ get_obs: o = obs_manager.compute()
         │  │  │  │
         │  │  │  ├─ 调用 observations.py 中的函数
         │  │  │  │   ├── projected_gravity(asset) → (3,)
         │  │  │  │   ├── joint_pos_rel(asset) → (6,)
         │  │  │  │   ├── joint_vel(asset) → (6,)
         │  │  │  │   └── ...
         │  │  │  │
         │  │  │  └─ 应用噪声 (std=0.025 etc)
         │  │  │     └─ o_noisy = o + N(0, σ²)
         │  │  │
         │  │  ├─ get_action: a = actor(o)
         │  │  │  │
         │  │  │  └─ a ∈ [-1, 1] (6 维)
         │  │  │
         │  │  ├─ process_action: τ = action_manager(a)
         │  │  │  │
         │  │  │  ├─ 缩放: a_scaled = a * 0.25
         │  │  │  ├─ 偏移: q_target = q_default + a_scaled
         │  │  │  ├─ PD 控制: τ = Kp(q_target-q) + Kd(-q_dot)
         │  │  │  │          = 25.0*e_pos - 0.8*q_dot
         │  │  │  └─ 限制: τ = clip(τ, -300, 300)
         │  │  │
         │  │  ├─ step: o', r, done = env.step(τ)
         │  │  │  │
         │  │  │  ├─ 物理仿真 (5 ms)
         │  │  │  │   └─ 更新关节位置/速度
         │  │  │  │
         │  │  │  ├─ 传感器更新
         │  │  │  │   ├── 接触传感器 → contact_state
         │  │  │  │   └── 高度扫描器 → heights
         │  │  │  │
         │  │  │  └─ 获取新观测 o'
         │  │  │
         │  │  ├─ compute_reward: r = rew_manager.compute()
         │  │  │  │
         │  │  │  ├─ 调用 rewards.py 中的函数
         │  │  │  │   ├── stay_alive() → 1.0
         │  │  │  │   ├── base_tracking() → exp(...)
         │  │  │  │   ├── gait_reward() → ...
         │  │  │  │   └── ...
         │  │  │  │
         │  │  │  └─ r_total = Σ(w_i * r_i)
         │  │  │              = 0.5*r_sa + 1.0*r_vel + ...
         │  │  │
         │  │  ├─ check_termination: done = env.is_done()
         │  │  │  └─ 检查: 摔倒/超界/超时
         │  │  │
         │  │  └─ store_transition: storage.add(o, a, r, o', done)
         │  │
         │  └─ [重复 2500 次]
         │
         ├─ Compute Advantages
         │  └─ A[t] = r[t] + γV(o[t+1]) - V(o[t])
         │     (GAE 方法)
         │
         ├─ Update Policy (PPO 算法)
         │  ├─ ~20 epochs
         │  ├─ 最小化: L_PPO = -min(r*A, clip(r,1±ε)*A)
         │  └─ actor 网络梯度下降
         │
         └─ Update Value (Critic)
            ├─ 最小化: L_V = (V(o) - Return)²
            └─ critic 网络梯度下降
```

---

## 配置继承关系

### 环境配置 (Env Config)

```python
# 最底层：机器人配置
pointfoot_cfg.py
  └── POINTFOOT_CFG = ArticulationCfg(
        spawn=UsdFileCfg(
          usd_path=".../PF_TRON1A.usd",
          rigid_props=RigidBodyPropertiesCfg(...),
        ),
        init_state=ArticulationCfg.InitialStateCfg(...),
        actuators={"legs": ImplicitActuatorCfg(...)}
      )

# 中间层：场景配置
limx_base_env_cfg.py
  └── class PFSceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(...)
        robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        contact_sensor = ContactSensorCfg(...)
        light = DomeLightCfg(...)

# 最上层：完整环境配置
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
  scene: PFSceneCfg
  observations: ObservationsCfg
  rewards: RewardsCfg
  actions: ActionsCfg
  commands: CommandsCfg
```

### 具体例子：修改刚度

```python
# 修改路径 1: 直接修改 pointfoot_cfg.py
# bipedal_locomotion/assets/config/pointfoot_cfg.py

POINTFOOT_CFG = ArticulationCfg(
  ...
  actuators={
    "legs": ImplicitActuatorCfg(
      stiffness=25.0,  # ← 修改这里
      damping=0.8,
      ...
    )
  }
)

# 之后所有引用 POINTFOOT_CFG 的配置都会受影响
```

---

## 模块功能分解

### bipedal_locomotion/assets/

| 文件 | 作用 | 修改频率 |
|------|------|---------|
| `config/pointfoot_cfg.py` | 定义机器人关节、执行器参数 | 🔴 低 |
| `config/solefoot_cfg.py` | SF 机器人配置 | 🔴 低 |
| `config/wheelfoot_cfg.py` | WF 机器人配置 | 🔴 低 |
| `usd/*/config.yaml` | 物理参数 (质量、重心等) | 🔴 低 |
| `usd/*/*.usd` | 几何体/物理模型 | 🔴 极低 |

### bipedal_locomotion/tasks/locomotion/cfg/

| 文件 | 作用 | 修改频率 |
|------|------|---------|
| `PF/limx_base_env_cfg.py` | 基础环境配置 | 🟡 中 |
| `PF/limx_flat_env_cfg.py` | 平地任务配置 | 🟡 中 |
| `PF/limx_rough_env_cfg.py` | 复杂地形配置 | 🟡 中 |

### bipedal_locomotion/tasks/locomotion/mdp/

| 文件 | 作用 | 修改频率 |
|------|------|---------|
| `observations.py` | 观测函数库 | 🟡 中 |
| `rewards.py` | 奖励函数库 | 🟢 高 |
| `actions.py` | 动作处理器 | 🔴 低 |
| `curriculums.py` | 课程学习 | 🟡 中 |
| `events.py` | 环境事件 | 🟡 中 |

### rsl_rl/rsl_rl/

| 文件 | 作用 | 修改频率 |
|------|------|---------|
| `algorithm/ppo.py` | PPO 算法 | 🔴 极低 |
| `modules/actor_critic.py` | 网络架构 | 🔴 低 |
| `runner/on_policy_runner.py` | 训练循环 | 🔴 低 |
| `storage/rollout_storage.py` | 数据存储 | 🔴 极低 |

---

## 关键参数速查表

### 环境相关

```python
# limx_base_env_cfg.py
timestep = 0.005              # 物理步长 (5 ms)
episode_length_s = 12.5       # episode 时长 (秒)
decimation = 4                # 决策间隔 (每 4 个物理步执行一次决策)
num_actions = 6               # 动作维度
num_observations = 59         # 观测维度
```

### 奖励相关

```python
# limx_base_env_cfg.py → RewardsCfg
reward_scales = {
  "stay_alive": 0.5,          # 存活奖励权重
  "base_tracking": 1.0,       # 速度追踪权重
  "gait_reward": 0.5,         # 步态奖励权重
  "feet_regulation": -0.1,    # 足部调节惩罚
  "action_smoothness": -0.01, # 动作平滑惩罚
}
```

### 执行器相关

```python
# assets/config/pointfoot_cfg.py
stiffness = 25.0              # PD P增益 (N⋅m/rad)
damping = 0.8                 # PD D增益 (N⋅m⋅s/rad)
effort_limit = 300            # 最大力矩 (N⋅m)
velocity_limit = 100.0        # 最大速度 (rad/s)
```

### 网络相关

```python
# agents/limx_rsl_rl_ppo_cfg.py
actor_hidden_dims = [256, 128]  # Actor 网络隐层
critic_hidden_dims = [256, 128] # Critic 网络隐层
activation_fn = nn.ReLU         # 激活函数
```

### PPO 相关

```python
# scripts/train.py / runner/on_policy_runner.py
learning_rate = 1e-4           # 学习率
gamma = 0.99                   # 衰减因子
gae_lambda = 0.95              # GAE λ
clip_epsilon = 0.2             # PPO 裁剪参数
num_mini_batches = 4           # 小批次数
num_epochs = 5                 # PPO 更新轮次
```

---

## 调试与修改建议

### 场景不稳定？

1. 检查 `pointfoot_cfg.py` 中的 `stiffness`/`damping`
2. 减少 `effort_limit` 防止过度控制
3. 增加 `solver_position_iteration_count` 提高物理精度

### 机器人行走缓慢？

1. 增加 `base_tracking` 奖励权重 (1.0 → 2.0)
2. 增加 `stiffness` (25.0 → 35.0)
3. 检查 `action_smoothness` 惩罚是否过大

### 步态不稳定？

1. 增加 `gait_reward` 权重
2. 增加 `damping` 参数 (0.8 → 1.2)
3. 检查 `feet_regulation` 惩罚是否有效

### 训练收敛缓慢？

1. 增加 `num_envs` (4096 → 8192，如显存允许)
2. 减小 `learning_rate` (1e-4 → 5e-5)
3. 增加 `clip_epsilon` (0.2 → 0.3)

---

## 文件关联图

```
┌─────────────────────────────────────────────────┐
│ scripts/rsl_rl/train.py (入口)                  │
└────┬────────────────────────────────────────────┘
     │
     ├─→ bipedal_locomotion.tasks.locomotion.cfg
     │   └─→ limx_base_env_cfg.PFSceneCfg
     │       └─→ bipedal_locomotion.assets.config.pointfoot_cfg
     │           └─→ POINTFOOT_CFG (机器人关节/执行器)
     │
     ├─→ bipedal_locomotion.tasks.locomotion.mdp
     │   ├─→ observations.py (观测函数)
     │   ├─→ rewards.py (奖励函数)
     │   ├─→ actions.py (动作处理)
     │   └─→ curriculums.py (课程学习)
     │
     ├─→ bipedal_locomotion.tasks.locomotion.agents
     │   └─→ limx_rsl_rl_ppo_cfg.py (网络架构)
     │
     └─→ rsl_rl.rsl_rl
         ├─→ algorithm.ppo (PPO 算法)
         ├─→ runner.on_policy_runner (训练循环)
         ├─→ modules.actor_critic (网络模块)
         └─→ storage.rollout_storage (轨迹存储)
```

---

## 快速定位表

| 需求 | 文件位置 | 行号范围 |
|------|---------|---------|
| 修改机器人质量 | `pointfoot_cfg.py` | ~30-50 |
| 修改关节刚度 | `pointfoot_cfg.py` | ~50-70 |
| 修改奖励权重 | `limx_base_env_cfg.py` | 观测管理器部分 |
| 添加新奖励项 | `rewards.py` | 文件末尾 |
| 修改观测空间 | `observations.py` | 各函数定义 |
| 修改网络架构 | `limx_rsl_rl_ppo_cfg.py` | 网络配置部分 |
| 调整超参数 | `scripts/train.py` | 命令行参数 |

---

**最后修改**: 2024-12-17  
**维护者**: 双足机器人团队
