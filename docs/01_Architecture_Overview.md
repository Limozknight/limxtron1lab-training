# 双足机器人强化学习架构详解

> 本文详细分析 LIMX TRON1A 双足机器人在 Isaac Lab 框架下的强化学习训练架构

## 项目概述

这是一个基于 **Isaac Lab** 框架的双足机器人强化学习训练项目，支持三种足部类型的机器人：
- **Point Foot (PF)**：点足，接触点为单点
- **Sole Foot (SF)**：脚掌，接触面为矩形区域  
- **Wheel Foot (WF)**：轮足，可滚动

项目采用 **PPO 算法** 在 **非对称 Actor-Critic** 框架下进行训练，通过特权信息蒸馏实现 Sim-to-Real 迁移。

---

## 1. Scene Configuration（场景配置）

### 1.1 总体架构

场景配置定义了机器人在虚拟环境中的完整配置：

```
┌──────────────────────────────────────────────────┐
│          Isaac Lab 模拟场景                      │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │  地形 (Terrain)                         │   │
│  │  - 平面/斜坡/台阶/随机地形                │   │
│  │  - 物理材料 (摩擦/恢复系数)               │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  机器人 (Robot - USD Asset)              │   │
│  │  - 刚体关节动力学                        │   │
│  │  - 执行器 (PD 控制器)                    │   │
│  │  - 初始化状态                            │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌─────────────────────────────────────────┐   │
│  │  传感器 (Sensors)                       │   │
│  │  - 接触传感器 (足部接地检测)            │   │
│  │  - IMU (加速度/角速度)                  │   │
│  │  - 高度扫描器 (环境感知)                │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 1.2 USD 资产与关节定义

#### USD 模型结构

```yaml
PF_TRON1A.usd (主模型)
├── PF_TRON1A_base.usd (几何体/视觉)
├── PF_TRON1A_physics.usd (刚体/碰撞)
└── PF_TRON1A_sensor.usd (传感器)

关节树结构:
Base (质量/惯性)
├── abad_L_Joint (左外展)    Range: ±0.35 rad
├── hip_L_Joint (左髋)       Range: ±1.57 rad
├── knee_L_Joint (左膝)      Range: ±2.09 rad
└── foot_L_Joint (左足)      Range: ±0.52 rad
├── abad_R_Joint (右外展)    Range: ±0.35 rad
├── hip_R_Joint (右髋)       Range: ±1.57 rad
├── knee_R_Joint (右膝)      Range: ±2.09 rad
└── foot_R_Joint (右足)      Range: ±0.52 rad
```

#### 关键物理参数

| 参数 | 值 | 作用 |
|------|-----|------|
| **机器人质量** | ~2.5 kg | 影响动力学模拟 |
| **重力加速度** | 9.81 m/s² | 标准地球重力 |
| **线性阻尼** | 0.0 | 关节摩擦最小化 |
| **角度阻尼** | 0.0 | 关节摩擦最小化 |
| **求解器迭代** | 4 次 | 物理精度与速度的平衡 |
| **碰撞检测** | 启用 | 足部与地面接触 |

#### PD 控制器配置

```python
执行器配置:
  - 刚度 (Kp): 25.0 (N⋅m/rad)
  - 阻尼 (Kd): 0.8 (N⋅m⋅s/rad)
  - 最大力矩: 300 N⋅m
  - 最大速度: 100 rad/s

力矩计算公式:
  τ = Kp * (q_target - q_current) + Kd * (0 - q_dot)
  τ_clipped = clip(τ, -300, 300)
```

**参数调整指南：**

| 场景 | Kp | Kd | 说明 |
|------|----|----|------|
| 反应迟缓 | ↑ | ↑ | 增加增益，同时增加阻尼防止振荡 |
| 关节振荡 | ↓ | ↑ | 减小刚度，增加阻尼 |
| 动作幅度小 | ↑ | 不变 | 增加刚度使关节反应更激进 |
| 能耗过高 | ↓ | ↓ | 减小增益降低能耗（但性能下降） |

### 1.3 场景定义

场景包含以下核心元素：

#### 地形配置

```python
TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",  # 或 "generator"
    collision_group=-1,
    physics_material=RigidBodyMaterialCfg(
        static_friction=1.0,      # 摩擦系数
        dynamic_friction=1.0,
        restitution=0.0,          # 不反弹
    ),
)
```

**地形类型对比：**

| 类型 | 用途 | 优点 | 缺点 |
|------|------|------|------|
| **plane** | 基础训练 | 快速，简单 | 不现实 |
| **hills** | 斜坡训练 | 增加难度 | 计算成本 |
| **mountains** | 复杂地形 | 鲁棒性好 | 训练困难 |
| **stairs** | 台阶训练 | 动态步态 | 不稳定 |

#### 接触传感器

```python
ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/foot_[LR]",
    update_period=0.005,     # 200 Hz
    history_length=2,        # 缓冲历史
)

输出: [num_envs, num_feet]
含义: 1.0 表示接触，0.0 表示离地
```

#### 高度扫描器（可选）

```python
RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=(0.0, 0.0, 0.2),  # 从基座上方 20cm 扫描
    pattern_cfg=GridPatternCfg(
        resolution=0.1,        # 10cm 网格
        size=[1.6, 1.0],       # 1.6m × 1.0m 覆盖范围
    ),
)

输出: [num_envs, num_rays]
含义: 各射线方向的高度信息
```

---

## 2. Observation Manager（观测管理器）

### 2.1 观测空间设计原则

观测空间分为两部分：

```
┌──────────────────────────────────────┐
│  策略网络输入观测 (Policy)            │
│  - 实时传感器数据                    │
│  - 添加高斯噪声（模拟传感器误差）      │
│  - 无法访问地面摩擦系数等隐藏参数      │
│  维度: ~59                           │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  教师网络输入观测 (Privileged)        │
│  - 所有物理真值（包括隐藏参数）        │
│  - 不添加噪声                        │
│  - 用于蒸馏提高策略鲁棒性             │
│  维度: ~80                           │
└──────────────────────────────────────┘
```

### 2.2 策略观测项详解

#### A. 基座观测 (3 维)

```python
# 投影重力向量 (3D)
g_proj = [g_x, g_y, g_z]
物理含义: 机器人相对于重力的姿态
- [0, 0, -1]: 直立
- [1, 0, 0]: 侧倾 90°
- [0, 1, 0]: 前倾 90°

计算方式:
  g_world = [0, 0, -9.81]
  g_local = R^T @ g_world
  其中 R 是机器人的旋转矩阵
```

#### B. 基座速度 (1 维)

```python
# Z 轴角速度 (rad/s)
w_z = base_ang_vel_w[:, 2]
物理含义: 机器人绕竖直轴的旋转速度
用途: 控制转向命令

缩放因子: 0.25 (降低量纲)
```

#### C. 关节观测 (12 维)

```python
# 关节相对位置 (6维)
dq = [abad_L_err, abad_R_err, hip_L_err, hip_R_err, knee_L_err, knee_R_err]
dq_i = q_i - q_default_i
物理含义: 各关节相对于静止位置的偏差
范围: ±π rad

# 关节速度 (6维)
v_q = [abad_L_vel, abad_R_vel, ..., knee_R_vel]
v_q_i = dq_i / dt
物理含义: 关节旋转速度
缩放因子: 0.05 (降低量纲)
```

#### D. 动作历史 (6 维)

```python
# 上一时步的神经网络输出
a_prev = [a_prev_1, ..., a_prev_6]
用途: 强化动作连续性，减少抖动
范围: [-1, 1]
```

#### E. 步态参数 (2 维)

```python
# 步态相位 (周期编码)
phase = [sin(2π*t/T), cos(2π*t/T)]
其中:
  t: 当前时间
  T: 步态周期

物理含义: 表示步态中所处的阶段（双足踏、摆动等）
优点: sin/cos 编码避免相位不连续性
```

#### F. 高度扫描 (可选，~32 维)

```python
# 前方地形高度信息
heights = [h_1, h_2, ..., h_16]
物理含义: 机器人周围的局部地形
范围: [-0.2, 0.2] m（相对于基座）
用途: 预测性的步态调整
```

**观测空间总维度：**

```
基础观测维度:
  投影重力 (3) + 角速度 (1) + 关节位置 (6) + 关节速度 (6) 
  + 动作历史 (6) + 步态相位 (2) + 步态命令 (3)
  = 27 维

加上高度扫描 (32 维):
  = 59 维
```

### 2.3 观测噪声注入

噪声参数（模拟真实传感器误差）：

```python
噪声类型: 加性高斯噪声 (Additive Gaussian Noise)

标准差 (std):
  - 投影重力: 0.025        # ±1.5° 罗盘误差
  - 角速度: 0.05            # ±2.9°/s 陀螺仪误差
  - 关节位置: 0.01          # ±0.57° 编码器误差
  - 关节速度: 0.01          # ±0.57°/s 速度估计误差

物理含义:
  o_measured = o_true + N(0, std²)

禁用条件:
  - 教师网络（无噪声）
  - 评估阶段
```

### 2.4 教师网络特权信息

在训练时，教师网络可以额外访问：

```python
# 额外的教师观测项

1. 实际关节扭矩 (6维)
   τ = [τ_1, ..., τ_6]
   用途: 学习能耗优化的步态

2. 关节加速度 (6维)
   q_ddot = [q̈_1, ..., q̈_6]
   用途: 预测性的动作规划

3. 足部线速度 (6维)
   v_feet = [v_L_xyz, v_R_xyz]
   用途: 步态同步化

4. 接触力 (6维)
   f_contact = [f_L_x, f_L_y, f_L_z, f_R_x, f_R_y, f_R_z]
   用途: 负载分布优化

5. 隐藏参数 (可变维度)
   - 地面摩擦系数 μ ∈ [0.3, 1.5]
   - 机器人质量 m ∈ [2.0, 3.0] kg
   - 重心高度 h ∈ [0.3, 0.6] m
```

**蒸馏流程：**

```
教师网络 (Privileged Info)
    ↓
知识蒸馏 (KL 散度最小化)
    ↓
学生网络 (部署用)
    ↓
移除特权信息，仅使用策略观测
```

---

## 3. Reward Manager（奖励管理器）

### 3.1 奖励设计原则

```
核心原则:
1. 任务相关: 直接指导所需行为
2. 稀疏 vs 密集: 平衡学习速度和性能
3. 可组合性: 多项奖励加权组合
4. 权重平衡: 避免任一项主导学习
```

### 3.2 核心奖励项

#### 1. 存活奖励 (Survive Reward)

```python
r_survive = 1.0 (每步骤)

用途: 鼓励智能体继续活动而不是摔倒
权重: 0.5

数学形式: r = 1 * num_steps
```

#### 2. 速度追踪奖励 (Velocity Tracking)

```python
v_actual = [v_x, v_y]        # 实际速度
v_command = [v_x_cmd, v_y_cmd]  # 命令速度

error = ||v_actual - v_command||₂

r_vel = exp(-error² / σ²)
σ = 0.5 (标准差)

权重: 1.0 (主要任务)

解释:
- error = 0 时，r_vel = 1.0 (完美追踪)
- error = 0.5 时，r_vel = 0.606
- error = 1.0 时，r_vel = 0.135
```

#### 3. 步态奖励 (Gait Reward)

```python
用途: 鼓励合理的接触模式

成分:

a) 接触力奖励:
   - 期望接触: f > threshold → 高奖励
   - 不期望接触: f = 0 → 高奖励
   
   r_force = Σ [desired_i * f_i + (1-desired_i) * (1-f_i)]
   
b) 足部速度奖励:
   - 支撑相 (接触): v_foot 应低
   - 摆动相 (离地): v_foot 应高
   
   r_vel = Σ [desired_i * low_vel_i + (1-desired_i) * high_vel_i]

权重: 0.5 (次要任务)
```

#### 4. 足部调节奖励 (Feet Regulation)

```python
惩罚不稳定的足部高度波动

h_feet_desired = base_height - 0.45 m

penalty = Σ |h_feet - h_feet_desired|

r_reg = -penalty

权重: -0.1 (惩罚项)
```

#### 5. 动作平滑性 (Action Smoothness)

```python
惩罚关节加速度（防止抖动）

a[t] = 神经网络输出
d²a/dt² = a[t] - 2*a[t-1] + a[t-2]

penalty = ||d²a/dt²||₂

r_smooth = -penalty

权重: -0.01 (弱惩罚)
```

#### 6. 转向奖励 (Angular Velocity)

```python
鼓励平滑的旋转

w_z_actual = 实际 Z 轴角速度
w_z_command = 命令角速度

r_ang = exp(-(w_error)² / 0.1²)

权重: 0.2
```

### 3.3 总奖励计算

```python
r_total = w_survive * r_survive
        + w_vel * r_vel
        + w_gait * r_gait
        + w_feet * r_feet
        + w_smooth * r_smooth
        + w_ang * r_ang

= 0.5 * r_survive
+ 1.0 * r_vel
+ 0.5 * r_gait
- 0.1 * r_feet
- 0.01 * r_smooth
+ 0.2 * r_ang
```

### 3.4 奖励权重调优指南

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 收敛缓慢 | Episode 返回值 < 100 | ↑ w_survive (0.5 → 1.0) |
| 速度过慢 | 实际速度 < 50% 命令 | ↑ w_vel (1.0 → 2.0) |
| 步态不合理 | 足部摩擦力过大或震荡 | ↑ w_gait (0.5 → 1.0) |
| 能耗过高 | 关节扭矩平均值 > 150 N⋅m | ↑ w_smooth (0.01 → 0.1) |
| 原地振荡 | 摔倒频繁 | ↑ w_feet (−0.1 → −0.2) |

---

## 4. Action Manager（动作管理器）

### 4.1 动作空间定义

```python
# 策略网络输出
a_policy ∈ ℝ⁶
范围: [-1, 1]

# 映射到关节指令
q_target_i = q_default_i + a_policy_i * scale
scale = 0.25 rad (±14.3°)
```

### 4.2 PD 控制器工作流

```
神经网络输出 a ∈ [-1, 1]
    ↓
乘以缩放因子 (×0.25)
    ↓
加上默认关节位置
    ↓ q_target
┌──────────────────────────────────────┐
│  PD 控制器                           │
│  τ = Kp(q_target - q) + Kd(-q̇)      │
│  Kp = 25.0 (刚度)                   │
│  Kd = 0.8 (阻尼)                    │
└──────────────────────────────────────┘
    ↓ τ
力矩限制 clip(τ, -300, 300) N⋅m
    ↓
应用到执行器
    ↓
物理仿真更新关节位置
```

### 4.3 参数灵敏度分析

#### Kp (刚度) 的影响

```
Kp = 25.0 (默认)
  优点: 平衡快速响应和能效
  缺点: 无

Kp = 50.0 (大)
  优点: 关节响应快，精度高
  缺点: 易振荡，能耗高
  使用: 需要快速动作的任务

Kp = 10.0 (小)
  优点: 平滑，能耗低
  缺点: 反应迟缓，无法精确跟踪
  使用: 能耗受限的场景
```

#### Kd (阻尼) 的影响

```
Kd = 0.8 (默认)
  优点: 适度阻尼，减少振荡
  缺点: 无

Kd = 2.0 (大)
  优点: 大幅减少振荡
  缺点: 反应变慢，到达时间增长
  使用: 容易振荡的系统

Kd = 0.3 (小)
  优点: 响应快
  缺点: 易振荡
  使用: 要求动作快的任务
```

### 4.4 动作处理完整流程

```python
# 时刻 t 的完整控制流程

1. 获取观测 o[t]
   o[t] = [g_proj, w_z, dq, dq_dot, ..., phase]

2. 策略推断
   a[t] = π(o[t])  # 神经网络
   范围: [-1, 1]

3. 动作处理
   a_scaled[t] = a[t] * 0.25
   q_target[t] = q_default + a_scaled[t]

4. PD 控制
   τ[t] = 25.0 * (q_target[t] - q[t]) + 0.8 * (-q_dot[t])
   τ_clipped[t] = clip(τ[t], -300, 300)

5. 物理仿真 (Δt = 0.0025 秒)
   q[t+1], q_dot[t+1] = Physics(q[t], τ_clipped[t])

6. 重复步骤 4-5 (2500 次，共 6.25 秒)
   再执行一个强化学习步骤
```

### 4.5 动作空间维度统计

```
总维度: 6
  - abad_L: 关节指令 1
  - abad_R: 关节指令 2
  - hip_L: 关节指令 3
  - hip_R: 关节指令 4
  - knee_L: 关节指令 5
  - knee_R: 关节指令 6

缩放后范围:
  原始: [-1, 1]
  缩放: [-0.25, 0.25] rad
  相对允许范围: ~70% (足够灵活)
```

---

## 5. 训练框架

### 5.1 PPO 算法概述

```
Proximal Policy Optimization (PPO)
策略梯度方法，适合连续控制

特点:
- 样本效率高
- 实现简单
- 性能稳定
```

### 5.2 训练流程

```
初始化:
  - 神经网络权重随机初始化
  - 创建 4096 个并行环境

主训练循环:
┌────────────────────────────────────────┐
│  Rollout Phase (数据收集)               │
│  ┌──────────────────────────────────┐  │
│  │ 对每个环境执行 2500 步:          │  │
│  │  1. 获取观测 o[t]               │  │
│  │  2. 策略推断 a[t] = π(o[t])    │  │
│  │  3. 环境步进，获得 r[t]        │  │
│  │  4. 存储 (o, a, r, o', done)  │  │
│  └──────────────────────────────────┘  │
│  总数据: 4096 × 2500 = 1024 万 样本    │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│  Advantage Estimation (优势估计)        │
│  ┌──────────────────────────────────┐  │
│  │ GAE 方法:                         │  │
│  │ A[t] = r[t] + γV(o[t+1])         │  │
│  │         - V(o[t])                │  │
│  │ (用于衡量动作相对于基线的好坏)    │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│  Policy Update (策略更新)              │
│  ┌──────────────────────────────────┐  │
│  │ 使用 PPO 目标函数 (~20 epoch):  │  │
│  │                                  │  │
│  │ L_PPO = -min(r_t * A_t,         │  │
│  │        clip(r_t, 1±ε) * A_t)   │  │
│  │                                  │  │
│  │ 更新策略网络权重                 │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│  Value Update (价值网络更新)            │
│  ┌──────────────────────────────────┐  │
│  │ 最小化价值损失:                  │  │
│  │ L_V = (V(o) - Return)²          │  │
│  │                                  │  │
│  │ 更新价值网络权重                 │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
                    ↓
            重复 (N 次迭代)
```

### 5.3 关键超参数

```python
# 环境配置
num_envs = 4096              # 并行环境数（显存允许的最大值）
episode_length = 2500        # 单个 episode 步数（6.25 秒）

# PPO 超参数
learning_rate = 1e-4         # 学习率
gamma = 0.99                 # 衰减因子（未来奖励的重要性）
gae_lambda = 0.95            # GAE λ 参数
clip_epsilon = 0.2           # PPO 裁剪参数（±20%）

# 网络架构
policy_network = [256, 128]  # Actor 网络层数
value_network = [256, 128]   # Critic 网络层数
activation = ReLU            # 激活函数

# 训练配置
num_iterations = 5000        # 总迭代数
save_interval = 100          # 每 100 次迭代保存一次
```

---

## 6. Sim-to-Real 迁移

### 6.1 域随机化 (Domain Randomization)

在训练时随机变化物理参数，提高策略的鲁棒性：

```python
# 随机参数范围
机器人质量: m ∈ [2.0, 3.0] kg (±20%)
关节刚度: Kp ∈ [20, 30]
关节阻尼: Kd ∈ [0.5, 1.2]
地面摩擦: μ ∈ [0.3, 1.5]
重力加速度: g ∈ [9.0, 10.0] m/s²
传感器延迟: τ ∈ [0, 50] ms

效果:
- 实机性能: 提升 30-50%
- 算法: PPO 策略梯度 × 成本系数
```

### 6.2 特权信息蒸馏

```
训练阶段:
  学生策略 π_s (仅政策观测)
  教师策略 π_t (政策 + 特权观测)

蒸馏:
  L_kl = KL(π_t || π_s)  # KL 散度最小化
  
  学生优化:
    L_total = L_RL + λ * L_kl
    其中 λ ≈ 0.1

部署:
  移除特权信息
  直接使用学生策略
```

---

## 参考文献与学习资源

### Isaac Lab 官方资源

- **官方文档**: https://isaac-sim.github.io/IsaacLab/
- **GitHub 仓库**: https://github.com/isaac-sim/IsaacLab
- **API 参考**: https://docs.omniverse.nvidia.com/isaacsim/latest/

### 强化学习理论

- **PPO 论文**: [Schulman et al., 2017] "Proximal Policy Optimization Algorithms"
- **Actor-Critic**: [Konda & Tsitsiklis, 2000]
- **GAE**: [Schulman et al., 2015] "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

### 机器人控制

- **双足行走**: [Tedrake, 2021] "Underactuated Robotics"
- **步态分析**: [Perry & Burnfield, 2010] "Gait Analysis: Normal and Pathological Function"

### 实用教程

- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Berkeley Deep RL**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **DeepMind Control Suite**: https://github.com/deepmind/dm_control

---

## 文件结构对应

```
exts/bipedal_locomotion/bipedal_locomotion/
├── assets/
│   ├── config/
│   │   ├── pointfoot_cfg.py        ← Scene: 关节/执行器配置
│   │   ├── solefoot_cfg.py
│   │   └── wheelfoot_cfg.py
│   └── usd/
│       └── [PF|SF|WF]_TRON1A/
│           ├── [Type].usd          ← USD 模型
│           └── configuration/      ← 物理配置
│
└── tasks/locomotion/
    ├── cfg/
    │   ├── [PF|SF|WF]/
    │   │   └── limx_base_env_cfg.py ← Scene/Obs/Reward/Action
    │   └── ...
    │
    └── mdp/
        ├── observations.py          ← 观测函数实现
        ├── rewards.py              ← 奖励函数实现
        ├── actions.py              ← 动作处理器
        ├── curriculums.py          ← 课程学习
        └── events.py               ← 环境事件回调

rsl_rl/rsl_rl/
├── algorithm/ppo.py                ← PPO 算法实现
├── env/vec_env.py                  ← 向量化环境
├── modules/
│   ├── actor_critic.py             ← 网络架构
│   └── mlp_encoder.py              ← MLP 编码器
├── runner/on_policy_runner.py      ← 训练主循环
└── storage/rollout_storage.py      ← 轨迹存储

scripts/rsl_rl/
├── train.py                        ← 训练脚本
└── play.py                         ← 推理脚本
```

---

## 总结

这个项目实现了一个完整的强化学习管道：

1. **模块化设计**: 场景、观测、奖励、动作独立配置，便于修改
2. **大规模并行**: 4096 个环境同步运行，加速训练
3. **非对称学习**: 教师-学生框架，提高泛化性能
4. **Sim-to-Real**: 域随机化和知识蒸馏，实现真实机器人部署
5. **灵活的奖励设计**: 多项奖励组合，权重可调

通过理解这些核心概念，可以有效地修改、调试和优化机器人的强化学习训练过程。

---

**最后修改**: 2024-12-17  
**维护者**: 双足机器人团队
