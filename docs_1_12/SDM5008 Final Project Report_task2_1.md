# SDM5008 Final Project Report

by 12211759 吴蔚芷

# 2.1 框架理解与代码总结（Code Review & Architecture）

## 2.1.0 Manager-Based RL 架构总体说明

本项目基于 **Isaac Lab 的 Manager-Based Reinforcement Learning Environment** 架构进行实现。该架构通过将强化学习环境拆解为若干功能明确、职责单一的 Manager（如 Scene、Action、Observation、Reward、Termination、Event、Curriculum），实现了高度模块化、可组合、可扩展的环境设计范式。

在代码层面，环境由一个继承自 `ManagerBasedRLEnvCfg` 的顶层配置类统一描述，各类 Manager 以配置对象（Cfg）的形式被注入环境。在运行时，`ManagerBasedRLEnv` 负责协调各 Manager 的调用顺序，并在每一个环境 step 中完成动作执行、物理仿真、观测构建、奖励计算与终止判断。

本项目中，`PFEnvCfg` 作为顶层环境配置，完整定义了以下子模块：

- `scene`: 物理世界与机器人资产的定义
- `actions`: 策略输出到控制指令的映射方式
- `observations`: 策略与价值网络的观测空间构建
- `rewards`: 训练目标函数（reward shaping）的组成
- `terminations`: 回合终止条件
- `events`: 随机化与扰动机制
- `curriculum`: 训练难度随时间变化的策略

该设计使得环境行为并非通过单一脚本逻辑硬编码，而是由配置驱动、由 Manager 在运行时自动调度完成，体现了 Isaac Lab 框架在复杂机器人任务中的工程化优势。

## 2.1.1 环境执行流程与模块协同关系（Runtime Architecture）

从运行时视角看，一个强化学习 step 在本环境中可概括为以下逻辑链路：

首先，**Action Manager** 接收来自策略网络的动作输出。该动作通常是一个已归一化的向量，其维度与受控关节数量一致。Action Manager 根据动作配置（如动作类型、缩放比例、参考姿态）将其映射为低层控制目标。

随后，环境进入物理仿真阶段。仿真以固定的 physics timestep 推进，而策略动作按照设定的 decimation 频率生效，即一个策略动作在多个连续仿真步中保持不变。这种设计在保证数值稳定性的同时，降低了策略更新频率，使学习过程更符合实际控制系统的时序特性。

在物理状态更新完成后，**Observation Manager** 从仿真系统与场景对象中读取所需状态量（如机器人基座速度、关节状态、接触信息等），并按预定义的观测组（policy / critic / history 等）进行组织。对于策略网络使用的观测，系统可选择性地注入噪声，以模拟真实传感器不确定性；而价值网络使用的观测则保持无噪声，甚至包含特权信息，以提升训练稳定性。

随后，**Reward Manager** 根据当前状态计算各个奖励项。每一项奖励由一个独立的函数计算得到，并按其权重加权求和形成总 reward。奖励项的设计与权重分配直接决定了策略优化的方向。

在 reward 计算完成后，**Termination Manager** 对回合是否结束进行判定，例如是否超时或发生非法接触（如机器人倒地）。若满足终止条件，当前回合结束并触发 reset 流程。

最后，**Event Manager** 与 **Curriculum Manager** 在特定时机介入。Event Manager 负责在 startup、reset 或 episode 中途施加随机化或扰动；Curriculum Manager 则根据训练进度动态调整环境难度参数（如地形等级）。

这一执行流程体现了 Isaac Lab Manager 架构中“**解耦职责 + 明确时序**”的核心思想，各模块既相互独立，又通过运行时调度形成完整闭环。

# 2.1.2 Scene Configuration（物理场景与机器人资产配置）

Scene Configuration 定义了强化学习环境中的**物理世界抽象层**，包括机器人资产（USD Articulation）、地形（Terrain）、传感器（Sensors）以及与之相关的物理与时序属性。在 Isaac Lab 的 Manager-Based 架构中，Scene 并不直接参与策略优化，而是为 Observation、Reward、Termination 等模块提供统一、可复用的物理状态来源。

本项目中，Scene 的配置遵循“**骨架定义 + 变体注入**”的工程范式：基础 Scene 类仅描述场景的通用结构，而具体机器人模型与地形类型在派生环境中完成绑定。

---

## Scene 配置类的代码结构与职责划分

Scene 由 `PFSceneCfg`（继承自 `InteractiveSceneCfg`）定义，其核心职责是**声明环境中存在的物理对象类型**，而非绑定具体实现。

```python
classPFSceneCfg(InteractiveSceneCfg):
"""Configuration for the scene."""

# 地形配置（默认）
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        debug_vis=False,
    )

# 机器人与部分传感器在基类中作为占位
    robot: ArticulationCfg = MISSING
    height_scanner: RayCasterCfg = MISSING

# 接触传感器（始终存在）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True, 
        update_period=0.0，
    )
```

从配置可以看出，Scene 被刻意设计为**不完整状态**（通过 `MISSING` 标记），从而强制要求在更高层级的环境配置中补齐关键资产。这一做法避免了在基类中硬编码具体机器人型号，使 Scene 可在多个任务与机器人之间复用。

---

## 机器人 USD 资产的注入机制（Articulation Binding）

具体的机器人资产绑定发生在派生环境配置（如 `PFBaseEnvCfg`）的 `__post_init__()` 方法中。该阶段在 Python 层完成，属于**配置装配（configuration assembly）阶段**，而非仿真运行阶段。

```python
def__post_init__(self):
	super().__post_init__()
	
	# 注入具体机器人资产
	self.scene.robot = POINTFOOT_CFG.replace(
	        prim_path="{ENV_REGEX_NS}/Robot"
	    )
```

其中：

- `POINTFOOT_CFG` 是一个完整的 `ArticulationCfg`，内部定义了：
    - USD 文件路径
    - 关节层级与自由度
    - 初始物理参数（质量、惯量、阻尼等）
- `replace(prim_path=...)` 用于将机器人复制并挂载到每一个并行子环境命名的正则表达式路径中。

### 设计含义（算法视角）

这一设计使得**策略网络与具体 USD 资产解耦**。从算法角度看，策略仅依赖于关节状态、基座状态等抽象量，而不关心资产文件的来源或组织方式。这对于后续 sim-to-sim / sim-to-real 的迁移尤为重要。

---

## 关节命名、控制接口与 Scene 的一致性约束

虽然关节控制逻辑主要由 Action Manager 定义，但 Scene 在结构层面必须保证以下一致性：

- 机器人 USD 中的关节命名必须与 ActionCfg 中的 `joint_names` 完全一致；
- 初始关节角（`init_state.joint_pos`）必须覆盖同一组关节。

在本项目中，六个受控关节（左右髋外展、髋俯仰、膝关节）在 Scene 中以 Articulation 的形式存在，而其**控制语义完全由 Action Manager 决定**。Scene 本身只负责提供关节的物理存在与状态接口。

---

## 地形配置：Plane 与 Generator 的切换逻辑

在基础 Scene 配置中，地形采用固定平面（`terrain_type="plane"`），用于调试与初期训练。而在更复杂的环境变体中，地形被切换为生成器模式。

```python
self.scene.terrain.terrain_type ="generator"
self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
```

其中 `BLIND_ROUGH_TERRAINS_CFG` 定义在独立的 `terrains_cfg.py` 中，用于描述随机地形的统计分布与生成规则。

### 算法层面的意义

- **Scene 负责“环境分布的支持能力”**：是否允许复杂地形存在；
- **Curriculum 负责“分布采样策略”**：训练过程中如何逐步增加地形难度。

这种拆分使策略学习过程具备更清晰的分阶段目标，有利于稳定收敛。Curriculum能够使得机器人的训练循序渐进，在完成后期任务中尤为重要。

---

## 传感器在 Scene 中的组织方式与数据流角色

### 接触传感器（Contact Sensor）

接触传感器通过正则表达式路径绑定至机器人所有刚体：

```python
ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=3,
    update_period=0.0,
)
```

该传感器在 Scene 中承担三类潜在角色：

1. **Termination 判定**：检测 base link 是否发生非法接触；
2. **Observation 输入**：提供足端/机体接触信息；
3. **Reward 计算依据**：用于惩罚拖地、错误支撑等行为。

### 高度扫描传感器（RayCaster）

高度扫描传感器在基类 Scene 中被标记为 `MISSING`，并仅在特定环境（如 blind locomotion）中启用。这表明该传感器并非所有任务的必需组件，而是与任务假设（是否允许显式地形感知）直接相关。

---

## 传感器时间尺度与控制频率的对齐

在环境初始化阶段，Scene 中部分传感器的 `update_period` 会根据仿真与控制参数动态调整：

```python
# 接触力需要高频更新以捕捉瞬态
self.scene.contact_forces.update_period = self.sim.dt 
# 高度扫描只需按控制频率更新
if self.scene.height_scanner is not None:
    self.scene.height_scanner.update_period = self.decimation * self.sim.dt
```

这一设计保证：

- 接触信息以最高时间分辨率更新（用于安全与终止）；
- 地形高度信息以控制频率更新（与策略决策节奏一致）。

从算法角度看，这避免了策略接收到“时间尺度不一致”的观测信号，有助于提高策略稳定性与可解释性。

---

## Training vs Play：Scene 层面的配置差异

在 **Play / Evaluation 配置** 中，Scene 本身的几何结构通常保持不变，但其**随机性与扰动来源被显著削弱或关闭**。在代码中体现为：

- 地形生成参数固定或限制在较低难度；
- Scene 相关的随机 Event（push、质量扰动）被禁用；
- 传感器仍然存在，但 Observation 层面的噪声注入被关闭（由 ObservationCfg 控制）。

从算法角度看：

- **Training Scene** 用于定义“策略需要泛化到的环境分布”；
- **Play Scene** 用于评估在“单一或弱随机环境”下的稳定性与行为质量。

这种区分确保了评估结果能够反映策略的真实控制能力，而非随机扰动下的鲁棒性下界。

---

## Scene 层级结构示意

1. **路径前缀**: 所有的环境实例都位于 /World/envs 下，每个子环境被命名为 `env_0`, `env_1` 等。
2. **机器人根节点**: 机器人被挂载在 `.../env_N/Robot` 路径下。
3. **连杆 (Links)**:
    - `base_Link`: 机器人的躯干/基座。
    - `abad_[L/R]_Link`: 髋部侧摆连杆。
    - `hip_[L/R]_Link`: 大腿连杆。
    - `knee_[L/R]_Link`: 小腿连杆。
    - `foot_[L/R]_Link`: 足端连杆 (这是接触检测 `foot_landing` 和 feet_distance 惩罚的关键位置)。
4. **关节 (Joints)**: 连接这些连杆的关节，如 `abad_L_Joint` 等。

```
/World
  ├── ground                          <-- 地形 (Terrain)
  │    └── collision_mesh             <-- 物理碰撞体
  ├── skyLight                        <-- 环境光
  ├── envs                            <-- 并行环境容器
  │    ├── env_0                      <-- 第0个子环境
  │    │    └── Robot                 <-- PointFoot 机器人 (Articulation Root)
  │    │         ├── base_Link        <-- 基座 (Root Body)
  │    │         ├── abad_L_Link      <-- 左髋侧摆连杆
  │    │         ├── hip_L_Link       <-- 左大腿连杆
  │    │         ├── knee_L_Link      <-- 左小腿连杆 (包含膝盖)
  │    │         ├── foot_L_Link      <-- 左足端 (接触点)
  │    │         ├── abad_R_Link      <-- 右髋侧摆连杆
  │    │         ├── hip_R_Link       <-- 右大腿连杆
  │    │         ├── knee_R_Link      <-- 右小腿连杆
  │    │         └── foot_R_Link      <-- 右足端
  │    ├── env_1
  │    │    └── Robot
  │    │         └── ...
  │    └── ... 
```

Scene Configuration 在本项目中承担的是**物理现实建模与信息源定义**的角色，而非策略逻辑本身。其核心设计特点包括：

- 基类 Scene 的占位式定义，支持多机器人、多任务复用；
- 机器人 USD 资产的后绑定机制，确保动作与状态接口的一致性；
- 地形生成与课程调度的职责解耦；
- 传感器作为跨 Manager 的共享物理信息源；
- 在 training 与 play 阶段对随机性与扰动的明确区分。

该模块为后续 Observation、Reward、Termination 的算法设计提供了稳定、可控且可扩展的物理基础。

# 2.1.3 Observation Manager（观测空间构建与噪声注入）

Observation Manager 在 Isaac Lab 的 Manager-Based 架构中承担着**状态抽象与信息建模**的核心职责。其目标并非简单地“暴露仿真状态”，而是将高维、异构的物理信息组织为**适合策略学习的观测向量**，并在训练阶段通过噪声注入与信息不对称设计，提高策略的鲁棒性与泛化能力。

在本项目中，Observation Manager 由 `ObservarionsCfg`（环境配置层）进行定义，并在运行时由 `ObservationManager` 自动实例化与调度。

---

## Observation 的整体组织方式：ObsGroup 机制

Isaac Lab 中的观测并非单一向量，而是通过 **Observation Group（ObsGroup）** 进行组织。每一个 ObsGroup 对应一套观测项（Observation Terms），并可独立配置噪声、裁剪、缩放等行为。

在本项目中，主要定义了以下观测组：

- **Policy Observation Group**：供策略网络（Actor）使用
- **Critic Observation Group**：供价值网络（Critic）使用
- （可选）History / Command 等辅助观测组

这种分组设计为后续实现**不对称 Actor–Critic（Asymmetric A–C）** 提供了结构基础。

---

## Policy Observation Group：面向可部署策略的观测设计

### 观测内容构成

Policy 观测组聚焦于**策略在真实系统中可获得的信息**，通常不包含任何仿真特权量。其典型组成包括：

- 机器人基座线速度（xy 平面）
- 机器人基座角速度（绕 z 轴）
- 投影重力向量（等价于姿态信息）
- 关节位置（相对默认姿态）
- 关节速度
- 上一时刻动作（last action）
- 任务命令（如期望线速度 / 角速度）

这些量共同构成了一个闭环控制所需的最小信息集合。

示例代码结构如下（节选，语义示意）：

```python
 class PolicyCfg(ObsGroup):
        """策略网络观测组配置"""

        # 1. 机器人基座状态 (Base State)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05), # 模拟陀螺仪噪声
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,              # 重力投影 (感知倾斜)
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 2. 关节状态 (Joint State)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,                  # 相对默认姿态的偏移
            noise=GaussianNoise(mean=0.0, std=0.01), # 模拟编码器误差
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            scale=0.05,                              # 速度通常数值较大，需缩放
        )

        # 3. 动作与任务 (Action & Task)
        last_action = ObsTerm(func=mdp.last_action)
        gait_command = ObsTerm(
            func=mdp.get_gait_command, 
            params={"command_name": "gait_command"}  # 步态参数 (频率/相位/偏移)
        )
```

---

### 噪声注入（Corruption）的算法动机

Policy 观测组在训练阶段显式启用了噪声注入：

```python
def __post_init__(self):
	self.enable_corruption =True
```

从算法角度看，这一设计有三层含义：

1. **模拟真实传感器噪声**
    
    真实 IMU 和编码器不可避免存在噪声，仿真中的高斯噪声 (std=0.05/0.01) 模拟了这一特性。直接在仿真中加入噪声，有助于缩小 sim-to-real gap。
    
2. **防止策略过拟合仿真精度**
    
    若观测始终为“完美状态”，策略容易依赖高精度信息做出脆弱决策，在环境分布变化时性能骤降。
    
3. **提升策略对局部扰动的鲁棒性**
    
    高斯噪声相当于在状态空间中进行小范围随机扰动，有助于学习平滑且稳定的控制策略。
    

---

### 裁剪（Clip）与缩放（Scale）的数值稳定性考虑

多个观测项配置了 `clip` 或 `scale` 参数（如基座角速度 `0.25`，关节速度 `0.05`）。其作用并非任务逻辑，而是**数值层面的稳定性保障**：

- **Clip**：防止异常状态（如数值爆炸）导致网络输入失控；
- **Scale**：统一不同物理量的数量级，使神经网络训练过程中的梯度更加均衡。

在算法实现上，这一步相当于在进入神经网络前执行一次轻量级的特征归一化。

---

## Critic Observation Group：特权信息与训练稳定性

### Critic 观测的设计目标

Critic 的职责是估计状态价值函数 `V(s)`，其核心目标是**降低策略梯度估计的方差、提高训练稳定性**。因此，Critic 并不受“可部署性”的约束，可以使用策略在真实系统中不可获得的信息。

在本项目中，Critic Observation Group 包含：

- 与 Policy 相同的基础状态信息（但无噪声）
- 额外的**特权信息（Privileged Information）**，例如：
    - 机器人质量与惯量参数
    - 执行器刚度与阻尼
    - 地形或物理材质相关参数

在 CriticCfg 中，我们看到了大量的**特权信息 (Privileged Information)**：

```python
    @configclass
    class CriticCfg(ObsGroup):
        # ... (包含 Policy 所有基础信息，但不加噪声) ...

        # 特权信息 (仅仿真上帝视角可见)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)  # 真实关节力矩
        robot_feet_contact_force = ObsTerm(                        # 真实足部接触力
            func=mdp.robot_feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", ...)},
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)                  # 随机化后的真实质量
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness) # 随机化后的真实刚度
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties) # 地面摩擦系数
        
        def __post_init__(self):
            self.enable_corruption = False  # 关闭噪声
```

---

### 不对称 Actor–Critic（Asymmetric A–C）的架构意义

Critic 观测组在配置中明确关闭了噪声注入：

```python
def__post_init__(self):
self.enable_corruption =False
```

这一不对称设计的算法意义在于：

- **Actor (Policy)**：在“虽然看不清摩擦力是多少，但感觉脚底打滑”的条件下，学习如何稳住身体。
- **Critic**：在“明确知道摩擦力系数只有 0.4”的条件下，准确判断当前状态的好坏，指导 Actor 学习。

该结构不会影响最终部署的策略网络，却能显著改善训练阶段的收敛速度与稳定性，是现代机器人强化学习中的常见设计。

---

## 历史观测（History Observation）的时间建模作用

除瞬时状态外，本项目显式配置了 HistoryObsCfg：

```python
		
    class HistoryObsCfg(ObsGroup):
	    # ... 复刻 Policy 的观测项 ...
	    def __post_init__(self):
	        self.history_length = 10          # 记录过去 10 帧
	        self.flatten_history_dim = False  # 保持时间维度独立 (N, 10, Dim)
```

其设计目的在于为策略或价值网络提供**短期时间上下文**，从而弥补纯 Markov 状态在动力学系统中的不足。

从算法角度看：

- flatten_history_dim=False 意味着输出张量的维度是 3维的，这为使用 **LSTM/GRU** 或 **Transformer** 提取时序特征留出了接口（或者是使用 TCN 进行卷积处理）。这允许网络捕捉速度的一阶导数（加速度）甚至接触的周期性规律。

---

## Training vs Play：Observation 层面的关键差异

在 **Training 配置** 中：

- Policy Observation 启用噪声注入（`enable_corruption=True`）；
- Critic 使用无噪声、含特权信息的观测；
- History 观测用于增强时间建模能力。

在 **Play / Evaluation** 配置 (PFBaseEnvCfg_PLAY) 中：

```python
self.observations.policy.enable_corruption = False
```

即在评估阶段，**所有噪声注入被关闭**。这确保了评估结果反映策略在“理想传感条件”下的性能上限，同时也符合 Sim-to-Real 的常规验证流程（先在无噪环境下看基准表现，再评估抗噪能力）。

---

## Observation Manager 与其他模块的接口关系

在运行时，Observation Manager 与其他模块存在以下关键接口关系：

- **Scene → Observation**：从 Articulation 与 Sensor 中读取物理状态；
- **Action → Observation**：通过 `last_action` 观测项引入控制历史；
- **Observation → Reward**：部分 reward term 直接复用观测计算结果；
- **Observation → Policy / Critic**：分别作为 Actor 与 Critic 的输入。

这种设计保证了信息流向的清晰性，避免了模块间的隐式依赖。

---

### Sum

Observation Manager 在本项目中不仅负责“提供状态”，更承担了**信息建模与训练策略引导**的角色。其核心特点包括：

- 通过 ObsGroup 实现结构化观测组织；
- 利用不对称观测支持稳定的 Actor–Critic 训练；
- 通过噪声注入与历史观测提升策略鲁棒性；
- 明确区分 training 与 play 阶段的观测配置。

这一模块直接决定了策略网络“看到什么信息”，从算法层面对最终学习效果具有决定性影响。

# 2.1.4 Action Manager（动作空间定义与控制接口）

Action Manager 定义了**策略输出如何转化为对物理系统的控制指令**。在 Isaac Lab 的 Manager-Based 架构中，该模块处于策略网络与底层物理仿真之间，是连接“学习决策”与“动力学执行”的关键桥梁。

本项目采用 **Joint Position Target（关节位置目标）** 作为动作空间，并通过仿真系统内部的 PD 控制机制将目标位置转化为实际力矩。

---

## 动作空间的类型选择：Joint Position Action

在环境配置中，Action Manager 通过 `JointPositionActionCfg` 进行定义：

```python
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "abad_L_Joint",
            "abad_R_Joint",
            "hip_L_Joint",
            "hip_R_Joint",
            "knee_L_Joint",
            "knee_R_Joint",
        ],
        scale=0.25,
        use_default_offset=True,
    )
```

这一定义明确了三件事情：

1. **控制对象：**动作作用于名为 `"robot"` 的 Articulation，即 Scene 中注入的机器人资产。
2. **动作维度与语义：**动作向量维度等于关节数量（6 维），且每一维动作都与一个具体的物理关节一一对应。
3. **动作的物理含义：**策略输出并非直接表示力矩，而是表示**相对于默认关节姿态的目标位置偏移**。

---

## Residual Joint Position Control（残差式关节位置控制）

`use_default_offset=True` 表明动作采用 **Residual Control** 形式：

$$
q_{\text{target}} = q_{\text{default}} + \alpha \cdot a
$$

其中：

- $q_{\text{default}}$为默认（或初始）关节角；
- $a \in [-1, 1]^6$ 为策略网络输出；
- $\alpha = \text{scale} = 0.25$ 为动作幅度缩放系数。

### 算法层面的意义

这种残差式设计在腿足机器人控制中具有重要优势：

- **降低学习难度：**策略不需要从零学习“如何站立”，而是在一个合理姿态附近进行微调。
- **缩小探索空间：**通过 scale 限制动作幅度，避免早期训练阶段出现极端关节目标，提升训练稳定性。
- **增强物理可行性：**默认姿态通常已满足基本的力学与稳定性要求，残差控制更容易生成可执行动作。

---

## Action Scale 的作用与选择依据

在本项目中，`scale=0.25` 是一个关键超参数。其作用并非简单的数值缩放，而是**直接决定了策略可探索的关节空间范围**。

从算法角度看：

- **Scale 过大**：策略容易生成大幅关节摆动，增加倒地风险，reward 梯度不稳定。
- **Scale 过小**：策略表达能力受限，难以完成快速调整或大幅步态变化。

因此，该参数在控制表达能力与稳定性之间起到平衡作用，是动作空间设计中的关键超参数。

---

## 动作更新频率：Decimation 与控制时序

Action Manager 的输出并非在每一个仿真步都更新，而是通过 **decimation** 参数进行控制。

在 `PFBaseEnvCfg` 中：

```python
self.sim.dt = 0.005 # 5 ms
self.decimation = 4
```

这意味着：

- 仿真步频率：$f_{\text{sim}} = 200\ \text{Hz}$
- 控制（策略）更新频率：$f_{\text{ctrl}} = \frac{200}{4} = 50\ \text{Hz}$

即：**每一个策略动作在连续 4 个仿真步中保持不变**。

---

### 算法视角下的 Decimation 设计意义

1. **时间尺度分离**：策略网络在较低频率上决策，物理系统在高频率上积分动力学。
2. **数值稳定性**：避免策略在极短时间尺度内频繁改变目标，利于 PD 控制器平稳跟踪。
3. **Sim-to-Real**：50 Hz 的控制频率与真实机器人的机载算力和通信带宽相匹配。

---

## PD 控制接口与力矩生成（Action → Physics）

虽然 Action Manager 本身只输出关节目标位置，但在仿真执行阶段，这些目标会被 Articulation 内部的 **PD 控制器** 转化为实际力矩：

$$
⁍
$$

*(注：通常 JointPositionAction 不指定 Target Velocity，因此 Damping 项主要表现为对当前速度的阻尼)*

其中 ($K_p, K_d)$ 由机器人执行器参数决定，并可在 Event Manager 中被随机化（training 阶段）。

### 架构意义

- Action Manager 与 PD 控制器解耦：
    - 策略只关心“想要到哪里”；
    - 底层控制负责“如何到达”。
- 为 sim-to-real 预留接口：
    - PD 增益可替换为真实机器人控制参数；
    - 策略结构无需修改。

---

## Training vs Play：Action 层面的差异

在 **Training 配置** 中：

- 动作作用于带随机化的执行器参数（如 stiffness、damping）；
- 策略需在执行器不确定性下学习稳定控制；
- Decimation 与 scale 通常保持不变，以保证策略结构一致性。

在 **Play / Evaluation 配置** 中：

- 执行器随机化被关闭；
- PD 参数固定；
- Action Manager 的结构与接口保持完全一致。

从算法评估角度看，这确保了：

- **训练阶段强调鲁棒性**；
- **评估阶段反映策略在确定动力学下的控制质量**。

---

## Action Manager 与其他模块的接口关系

在整体架构中，Action Manager 与其他模块的关系如下：

- **Policy → Action Manager：**策略输出归一化动作向量。
- **Action Manager → Physics / Scene：**将动作映射为关节目标并施加到 Articulation。
- **Action Manager → Observation Manager：**当前动作通过 `last_action` 观测项反馈给策略，形成控制闭环。
- **Action Manager → Reward Manager：**动作变化幅度、力矩等可作为正则项参与 reward 计算。

---

## Sum

本项目中的 Action Manager 采用 **Residual Joint Position Control + Decimation** 的设计，在算法与工程层面均具有明确优势：

- 动作空间语义清晰、物理可行；
- 控制与仿真时间尺度合理分离；
- 通过 scale 与 default offset 平衡表达能力与稳定性；
- 为后续 sim-to-real 或控制器替换保留结构空间。

Action Manager 决定了策略“**能做什么样的动作**”，其设计直接影响学习难度、收敛速度以及最终行为质量。

---

# 2.1.5 Reward Manager（奖励项设计与权重影响）

Reward Manager 定义了强化学习任务的**优化目标函数**，决定了策略在训练过程中“被鼓励什么行为、被抑制什么行为”。在 Isaac Lab 的 Manager-Based 架构中，Reward Manager 并不直接参与状态更新或动作执行，而是在每个环境 step 中，根据当前状态与动作结果计算标量奖励信号，并将其反馈给学习算法。

本项目的 Reward Manager 由 `RewardsCfg` 统一配置，各奖励项（Reward Terms）以函数级别的方式定义在 `rewards.py` 中，并通过加权求和的方式构成最终 reward。

---

## Reward Manager 的组织结构：RewTerm + 权重聚合

在配置层面，奖励由多个 `RewTerm` 组成，每一个 `RewTerm` 包含三个核心要素：

- **计算函数（func）**：定义奖励项的数学形式
- **权重（weight）**：控制该奖励项在总 reward 中的相对重要性
- **参数（params）**：控制函数内部的尺度或阈值

在运行时，总奖励按照以下形式计算：

$$
R_t = \sum_{i=1}^{N} w_i \cdot r_i(s_t, a_t)
$$

其中 $r_i$ 为单个 reward term，$w_i$ 为对应权重。

这一设计使得 reward shaping 具备良好的**可解释性与可调性**：每一项奖励都可被单独分析、调试与重加权。

---

## 奖励项的功能分类

从算法设计角度，本项目中的奖励项可划分为三类：**任务驱动项、稳定性约束项与正则化项**。这种分类方式有助于理解不同奖励项在梯度更新中的角色。

---

### 1. 任务驱动奖励（Task / Tracking Rewards）

任务驱动奖励用于引导策略完成主要目标，在本项目中主要体现为**速度跟踪**。

代码示例：

```python
rew_lin_vel_xy_precise = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=3.0,                  # [Review 注] 权重极高，主导梯度方向
        params={"std": math.sqrt(0.2)} # 指数核函数的宽度，决定了对误差的容忍度
    )
```

该类奖励具有以下特点：

- 奖励值通常在 $[0,1]$ 区间内；
- 对应权重相对较大，是策略梯度的**主要来源**；
- 决定了策略“走多快、朝哪个方向走”。

从算法角度看，这类奖励定义了**优化目标的主方向**，其设计直接决定最终行为是否符合任务需求。

---

### 2. 稳定性与安全约束（Stability / Safety Penalties）

稳定性奖励项用于防止策略通过“作弊行为”获得高 reward，例如：

- 倒地后仍尝试输出动作；
- 通过剧烈抖动来短暂匹配速度命令；
- 产生物理上不可接受的接触模式。

在本项目中，这类约束通常与以下信息相关：

- 接触传感器（base contact、foot contact）；
- 姿态偏差（如基座倾角）；
- 不合理的运动模式（如拖地、跳跃异常）。

例如，与接触相关的惩罚项在 `rewards.py` 中可能表现为：

```python
    # 防双脚碰撞惩罚：绝对红线
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100.0,             # [Review 注] 极大的惩罚权重
        params={
            "min_feet_distance": 0.115,  # 最小安全距离阈值
            "feet_links_name": ["foot_[RL]_Link"]
        }
    )

    # 着陆缓冲惩罚：保护机械结构
    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,
        weight=-0.5,
        params={
            "about_landing_threshold": 0.08  # 检测即将触地的阈值
        },
    )
```

这些奖励项往往与 **Termination Manager** 紧密配合：

- reward 负责在“尚未终止”时持续施加惩罚；
- termination 在严重违规时直接终止回合。

---

### 3. 正则化奖励（Regularization Terms）

正则化项并不直接与任务目标相关，而是用于塑造策略的**行为质量**，例如：

- 减少能耗（关节力矩、关节速度）；
- 平滑动作变化（惩罚 action difference）；
- 防止高频振荡。

代码示例：

```python
    # 动作平滑性惩罚：抑制高频抖动
    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,
        weight=-0.04  # 权重较小，用于微调
    )
    
    # 机械功惩罚 (L1 Fan-norm)
    pen_joint_powers = RewTerm(
        func=mdp.joint_powers_l1,
        weight=-5e-04 # 用于长期训练中的能效优化
    )
```

从算法角度看，这类奖励：

- 权重通常较小（1e-3 ~ 1e-4 量级）；
- 不主导策略方向；
- 但在长期训练中显著影响行为的平滑性与可执行性。

---

## Reward 权重的算法意义与尺度匹配问题

### （1）权重作为“梯度分配器”

在多奖励项场景下，reward 权重的核心作用并非简单的“偏好设置”，而是**决定不同 reward 项对梯度更新的贡献比例**。

- **权重过大**（如 pen_feet_distance）：产生“硬约束”效果，优先级最高。
- **权重过小**（如 pen_joint_torque）：仅作为“软约束”，在不影响主任务的前提下优化。

因此，权重设计本质上是一个**多目标优化中的权衡问题**。

---

### （2）Reward 尺度匹配（Scale Matching）

由于不同 reward 函数的数值尺度天然不同（例如 exp tracking vs. 二次惩罚），权重还承担着**数值尺度对齐**的职责。

在本项目中，可以观察到：

- **Tracking (归一化)**：值域 [0, 1]，权重设为 **3.0**。
- **Joint Power (物理量)**：值域可能高达 100~1000 W，权重设为 **5e-04**。这种设计确保了最终进入 Loss 函数的各分量在数值上是可比的，避免了某一物理量因为数值过大而掩盖了其他信号。

这种配置体现了 reward shaping 中常见的“**主任务 + 辅助约束 + 基线奖励**”结构。

---

## Reward 与 Observation / Action 的耦合关系

Reward Manager 并非孤立模块，其输入信息高度依赖于 Observation 与 Action 的设计。

- **Observation → Reward：**多数 reward term 直接复用观测中已有的状态量（如 base velocity、joint velocity），避免重复计算。
- **Action → Reward：**正则化奖励（如 action smoothness）依赖于当前与上一时刻动作，形成对控制信号的反馈约束。
- **Scene / Sensor → Reward：**接触相关奖励项直接依赖 Scene 中的 Contact Sensor 输出。

这种耦合关系意味着：**reward 的有效性高度依赖于 observation 的信息质量与 action 的物理语义。**

---

## Training vs Play：Reward 层面的配置差异

- **Training 阶段**：Reward 信号作为 Critic 的监督信号（TD Error），驱动策略网络参数更新。
- **Play 阶段**：Reward 计算逻辑保持不变，但仅用于**指标监控（Metrics Logging）**，评估策略性能，不再反向传播。

从算法评估角度看，这种设计确保了训练与评估指标的一致性，同时避免了在评估阶段引入额外的干扰因素。

---

## Reward Manager 与 Termination 的协同关系

Reward 与 Termination 之间存在明确的职责分工：

- **Reward (软约束)**：在合法状态空间内连续塑造行为（例如：稍微有点歪，扣点分）。
- **Termination (硬约束)**：在不可接受状态下立即终止（例如：base_contact 检测到躯干触地，直接重置 Episode）。

这种“软约束（reward）+ 硬约束（termination）”的组合，是稳定腿足机器人训练的常见做法。

---

## Sum

本项目的 Reward Manager 采用**多项加权 reward shaping**的设计，其核心特点包括：

- 奖励项函数级定义，结构清晰、可解释；
- 权重作为梯度分配与尺度对齐的关键手段；
- 明确区分任务驱动、稳定性约束与正则化奖励；
- 与 Observation、Action、Termination 紧密耦合，形成完整闭环。

Reward Manager 决定了策略“**为什么要这样行动**”，是连接物理行为与优化目标的核心模块。

# 2.1.6 Manager-Based 架构的整体协同与运行逻辑总结

在本项目中，Isaac Lab 提供的 Manager-Based Reinforcement Learning 架构被完整采用，用于组织复杂的腿足机器人控制任务。各 Manager 模块在功能上高度解耦，但在运行时通过明确的信息流与控制流形成闭环，共同支撑策略训练与评估。

本节从**运行时视角**对各 Manager 的协同关系进行总结，并进一步概括 training 与 play 两种模式下的系统差异。

---

## 2.1.6.1 单步环境执行中的信息流（Information Flow）

从一次环境 step 的执行过程来看，系统中的主要信息流可概括为以下闭环：

$$
Observation→Policy→Action→Physics→Observation
$$

```python
graph LR
    subgraph Env [Environment / Physics]
        Scene[Scene & Sensors] -->|Raw Physics State| ObsManager
        ActionManager -->|Joint Targets| Scene
    end

    subgraph Agent [RL Agent]
        ObsManager -->|Noisy Obs| Policy[Actor Network]
        ObsManager -->|Privileged Obs| Critic[Critic Network]
        Policy -->| normalized actions | ActionManager
    end
    
    Scene -->|Contact Data| Reward[Reward Manager]
    Scene -->|State Data| Termin[Termination Manager]
```

具体而言：

1. **Scene → Observation Manager：**Scene 中的 Articulation 与 Sensors 提供底层物理状态（关节、基座、接触、地形信息等），Observation Manager 对其进行抽象、裁剪、缩放与组织。
2. **Observation Manager → Policy / Critic：**Policy 接收带噪、可部署的观测；Critic 接收无噪、含特权信息的观测，用于价值估计。
3. **Policy → Action Manager：**策略网络输出归一化动作向量，表示期望的关节残差控制指令。
4. **Action Manager → Physics / Scene：**动作被映射为关节目标位置，通过 PD 控制器作用于机器人，并在 decimation 控制下推进物理仿真。
5. **Physics → Observation Manager（下一步）：**更新后的物理状态再次被采集，形成下一个 step 的观测输入。

这一闭环体现了强化学习控制系统中典型的**感知–决策–执行–反馈**结构。

---

## 奖励与终止：优化信号与安全边界的分工

在上述信息流之外，Reward Manager 与 Termination Manager 构成了一条**并行的评价与约束通道**。

- **Reward Manager (连续度量)**：
    - ROLE: 在每个 step 中，根据当前状态与动作计算标量奖励。
    - INPUT: 参与策略梯度计算，通过多项加权 reward shaping，引导策略逐步逼近目标。
- **Termination Manager (状态边界)**：
    - ROLE: 在检测到不可接受状态（如倒地、非法接触）时立即终止 episode。
    - INPUT: 防止策略在失效状态下继续采样，提供了“硬约束”。

---

## Events 与 Curriculum：环境分布的动态调度

与核心闭环并行存在的，是 **Event Manager** 与 **Curriculum Manager**，它们不直接参与策略决策，但深刻影响策略最终学到的行为分布。

- **Event Manager (随机化与扰动)**：
    - 定义了 Training 时的环境分布族（Domain Randomization）。
    - 负责在 `startup`（质量/摩擦力）、`reset`（初始姿态）和 `interval`（外力推挤）阶段介入。
- **Curriculum Manager (难度演化)**：
    - 根据训练进度，动态调整地形难度（地形等级）。
    - 作用：早期防止 Collapse，后期提升泛化性。

---

## 2.1.6.4 Training vs Play：整体系统配置差异总结

从系统层面看，training 与 play 并非两个不同的环境，而是**同一架构在不同配置下的两种运行模式：**

| 特性 | Training 模式 | Play / Evaluation 模式 |
| --- | --- | --- |
| **Observation** | 启用噪声注入 (Noise Injection) | 关闭噪声，使用确定性观测 |
| **Critic** | 启用特权信息 (Privileged Info) | 通常不计算 / 不使用 |
| **Action** | 执行器参数 (Kp/Kd) 随机化 | 执行器参数固定 |
| **Events** | 启用推力、质量等随机扰动 | 随机扰动关闭或最小化 |
| **Reward** | 计算梯度，驱动优化 | 仅用于 Log 评估，不更新网络 |
| **Curriculum** | 动态调整地形难度 | 固定在指定难度 |
- **Training 目标**：在分布化、不确定的环境中学习**鲁棒策略**。
- **Play 目标**：评估策略在确定条件下的**控制性能上限**。

---

## 2.1.6.5 架构设计的整体优势总结

综合来看，本项目所采用的 Isaac Lab Manager-Based 架构具备以下显著优势：

1. **模块解耦，职责清晰**：避免了传统强化学习代码中“几千行 `step()` 函数”的混乱局面。
2. **配置驱动 (Config-Driven)**：环境行为主要由 Python Class 描述，便于实验复现与版本管理。
3. **天然支持 Sim-to-Real**：通过 Observation 噪声注入、Action 延时/Decimation 和 Event 域随机化，构建了完整的虚实迁移路径。
4. **不对称 A-C 支持**：结构上原生支持特权信息分离，极大提升了训练效率。

---

## 小结（2.1 章节总结）

在 **2.1 框架理解与代码总结** 中，本项目从架构层面对 Isaac Lab 的 Manager-Based 强化学习环境进行了系统梳理。通过对 Scene、Observation、Action、Reward 等核心模块的代码级分析，并结合运行时信息流与控制流的整体视角，展示了该架构如何支撑复杂腿足机器人任务的稳定训练与评估。

这一章节为后续 **环境搭建、奖励设计迭代与参数调优** 提供了清晰的系统背景与技术基础。