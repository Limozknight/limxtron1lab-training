# 任务修改指南：2.2-2.4 代码实现

> 本文档详细介绍如何修改和实现 2.2-2.4 任务的代码，包括所需的方法、预期挑战和学习资源

---

## Task 2.2: 平地速度跟随 (Flat Ground Velocity Tracking)

### 2.2.1 任务描述

**目标**: 机器人在平整路面上稳定行走，精准响应速度指令

**输入**: 用户指令 `(v_x, v_y, ω_z)`
- `v_x`: 前进速度 (m/s)，范围 [-1.5, 1.5]
- `v_y`: 横向速度 (m/s)，范围 [-0.5, 0.5]  
- `ω_z`: 旋转速度 (rad/s)，范围 [-1.0, 1.0]

**评分标准**:

```
总分 = (1 - MSE权重) * 速度精度 + 姿态稳定性 + 存活率
      = 0.6 * (1 - MSE/MSE_max) + 0.3 * 姿态稳定性 + 0.1 * 存活率

1. 速度追踪误差 (MSE, 权重 60%)
   MSE = (1/T) * Σ(v_actual - v_command)²
   目标: MSE < 0.1

2. 姿态稳定性 (权重 30%)
   roll_std, pitch_std < 0.1 rad
   目标: 机器人不摇晃

3. 存活率 (权重 10%)
   目标: 测试过程中不摔倒 (roll/pitch > 45°)
```

### 2.2.2 实现步骤

#### Step 1: 创建平地环境配置

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_flat_env_cfg.py`

```python
from dataclasses import dataclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF import limx_base_env_cfg as base_cfg
from bipedal_locomotion.tasks.locomotion import mdp

@dataclass
class FlatTerrainEnvCfg(base_cfg.LocomotionEnvCfg):
    """平地行走环境配置"""
    
    def __init__(self):
        super().__init__()
        
        # ===== 1. 修改地形配置 =====
        # 确保使用平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None  # 不使用生成器
        
        # ===== 2. 修改奖励权重 =====
        # 对平地任务的权重调整
        self.rewards.stay_alive.weight = 0.2      # 降低存活奖励（平地易存活）
        self.rewards.base_tracking.weight = 2.0   # 提高速度追踪权重（核心任务）
        self.rewards.action_smoothness.weight = -0.02  # 增加平滑性约束
        
        # 移除对复杂地形的奖励
        if hasattr(self.rewards, "feet_clearance"):
            del self.rewards.feet_clearance
        
        # ===== 3. 修改命令配置 =====
        # 增加命令随机化
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)  # 每 4-8 秒改变一次
        self.commands.base_velocity.heading_command.ranges = [
            [-1.5, 1.5],    # v_x range
            [-0.5, 0.5],    # v_y range
            [-1.0, 1.0],    # w_z range
        ]
```

#### Step 2: 修改奖励函数以支持 2D 速度追踪

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py`

```python
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

class VelocityTrackingReward(ManagerTermBase):
    """
    2D 速度 + 旋转速度追踪奖励
    
    奖励策略:
    - 基于欧氏距离的 2D 速度误差
    - 加权的旋转速度误差
    """
    
    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        )
        # 从配置中读取权重
        self.lin_vel_weight = float(cfg.params.get("lin_vel_weight", 1.0))
        self.ang_vel_weight = float(cfg.params.get("ang_vel_weight", 0.5))
        self.lin_vel_sigma = float(cfg.params.get("lin_vel_sigma", 0.5))
        self.ang_vel_sigma = float(cfg.params.get("ang_vel_sigma", 0.5))
    
    def __call__(self, env) -> torch.Tensor:
        """计算速度追踪奖励"""
        # 获取机器人资产
        asset: Articulation = env.scene[self.asset_cfg.name]
        device = env.device
        
        # ===== 线性速度追踪 =====
        # 获取实际速度 (世界坐标系)
        lin_vel_actual = asset.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
        
        # 获取命令速度
        lin_vel_command = env.command_manager.get_command("base_velocity")[:, :2]
        
        # 计算 2D 误差 (欧氏距离)
        lin_vel_error = torch.norm(lin_vel_actual - lin_vel_command, dim=1)
        
        # 高斯奖励: exp(-error²/σ²)
        lin_vel_reward = torch.exp(-(lin_vel_error ** 2) / (self.lin_vel_sigma ** 2))
        
        # ===== 旋转速度追踪 =====
        # 获取实际角速度
        ang_vel_actual = asset.data.root_ang_vel_w[:, 2]  # Z轴
        
        # 获取命令角速度
        ang_vel_command = env.command_manager.get_command("base_velocity")[:, 2]
        
        # 计算误差
        ang_vel_error = torch.abs(ang_vel_actual - ang_vel_command)
        
        # 高斯奖励
        ang_vel_reward = torch.exp(-(ang_vel_error ** 2) / (self.ang_vel_sigma ** 2))
        
        # ===== 组合奖励 =====
        reward = (
            self.lin_vel_weight * lin_vel_reward +
            self.ang_vel_weight * ang_vel_reward
        ) / (self.lin_vel_weight + self.ang_vel_weight)
        
        return reward


# 在 RewardsCfg 中注册
@dataclass
class RewardsCfg:
    # ... 其他奖励项 ...
    
    base_velocity_tracking = RewTerm(
        func=VelocityTrackingReward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lin_vel_weight": 1.0,       # 线性速度权重
            "ang_vel_weight": 0.5,       # 旋转速度权重
            "lin_vel_sigma": 0.5,        # 线性速度容差
            "ang_vel_sigma": 0.5,        # 旋转速度容差
        }
    )
```

#### Step 3: 验证与调试

**文件**: `scripts/rsl_rl/train.py`

```python
# 在训练脚本中添加诊断代码
import torch

def diagnose_velocity_tracking(env, n_steps=100):
    """诊断速度追踪性能"""
    lin_vel_errors = []
    ang_vel_errors = []
    
    for _ in range(n_steps):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 获取当前速度
            asset = env.scene["robot"]
            lin_vel_actual = asset.data.root_lin_vel_w[:, :2]
            lin_vel_command = env.command_manager.get_command("base_velocity")[:, :2]
            
            lin_vel_error = torch.norm(lin_vel_actual - lin_vel_command, dim=1)
            lin_vel_errors.append(lin_vel_error.mean().item())
            
            # 执行一步
            action = env.action_manager.action  # 获取上一步的动作
            obs, reward, done, _ = env.step(action)
    
    # 计算统计信息
    lin_vel_errors = torch.tensor(lin_vel_errors)
    mse = torch.mean(lin_vel_errors ** 2).item()
    mean_error = torch.mean(lin_vel_errors).item()
    std_error = torch.std(lin_vel_errors).item()
    
    print(f"Linear Velocity Tracking:")
    print(f"  MSE: {mse:.4f} (目标: < 0.1)")
    print(f"  Mean Error: {mean_error:.4f} m/s")
    print(f"  Std Error: {std_error:.4f} m/s")
```

### 2.2.3 预期挑战与解决方案

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 速度响应缓慢 | MSE > 0.2 | 增加 `base_tracking` 权重; 增加 `stiffness` |
| 运动不稳定 | roll/pitch 抖动 | 增加 `damping`; 增加 `action_smoothness` 惩罚 |
| 原地打转 | ω_z 精度低 | 调整 `ang_vel_sigma`; 增加 `ang_vel_weight` |
| 能耗过高 | 关节扭矩 > 200 N·m | 增加 `action_smoothness` 惩罚; 降低 `stiffness` |

### 2.2.4 关键参数表

```python
# 推荐参数配置
stiffness = 30.0              # 提高刚度以加快响应
damping = 1.5                 # 增加阻尼以减少振荡
effort_limit = 300            # 保持最大力矩
action_scale = 0.25           # 动作缩放

reward_weights = {
    "stay_alive": 0.2,        # 低权重（平地易存活）
    "base_tracking": 2.0,     # 高权重（主要目标）
    "action_smoothness": -0.02,
}

lin_vel_sigma = 0.5           # 允许 ±0.5 m/s 的追踪误差
ang_vel_sigma = 0.5           # 允许 ±0.5 rad/s 的角速度误差
```

---

## Task 2.3: 抗干扰鲁棒性 (Disturbance Rejection)

### 2.3.1 任务描述

**目标**: 在平地行走时承受突发推力而不摔倒

**评分标准**:

```
抗扰能力指数 = (最大承受冲量 / 参考冲量) × 恢复速度系数

1. 最大冲量容限 (Primary)
   I_max = max(I) where robot不摔倒
   单位: N·s (牛顿·秒)
   目标: I_max > 100 N·s

2. 恢复速度 (Secondary)
   恢复时间 = 受扰后恢复正常步态的时间
   单位: 秒
   目标: 恢复时间 < 1.5 s

3. 稳定性保留率 (Tertiary)
   保持速度精度的比例
   目标: 受扰后 MSE 增长 < 50%
```

### 2.3.2 实现步骤

#### Step 1: 域随机化配置

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_robust_env_cfg.py`

```python
from dataclasses import dataclass
from isaaclab.sim.schemas.sim import SimulationCfg
from isaaclab.utils.configclass import MISSING
from bipedal_locomotion.tasks.locomotion.cfg.PF import limx_flat_env_cfg

@dataclass
class RobustEnvCfg(limx_flat_env_cfg.FlatTerrainEnvCfg):
    """抗干扰鲁棒性环境配置 - 包含域随机化"""
    
    def __init__(self):
        super().__init__()
        
        # ===== 1. 机器人参数随机化 =====
        # 关节刚度变化 ±20%
        self.scene.robot.actuators["legs"].stiffness_range = (20.0, 30.0)
        
        # 关节阻尼变化 ±25%
        self.scene.robot.actuators["legs"].damping_range = (0.6, 1.2)
        
        # 机器人质量变化 ±15%
        self.scene.robot.mass_range = (2.125, 2.875)  # 2.5 ± 15%
        
        # ===== 2. 地形参数随机化 =====
        # 摩擦系数变化
        self.scene.terrain.physics_material.friction_range = (0.3, 1.5)
        
        # ===== 3. 环境参数随机化 =====
        # 重力变化 ±2%
        self.gravity_range = (9.0, 10.0)
        
        # 风力干扰 (可选)
        self.wind_force_range = (0.0, 5.0)  # 0-5 N
        
        # ===== 4. 传感器噪声增加 =====
        self.observations.policy.projected_gravity.noise.std = 0.05   # 增加
        self.observations.policy.joint_pos.noise.std = 0.02          # 增加
        self.observations.policy.joint_vel.noise.std = 0.02          # 增加
        
        # ===== 5. 修改奖励配置 =====
        # 增加稳定性奖励权重
        self.rewards.feet_regulation.weight = -0.2   # 更严格的足部约束
        self.rewards.base_tracking.weight = 1.5      # 优先恢复速度追踪
```

#### Step 2: 实现推力施加机制

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/events.py`

```python
import torch
import numpy as np
from isaaclab.managers import ManagerTermBase, EventTermCfg
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

class RandomPushEvent(ManagerTermBase):
    """
    随机推力事件 - 模拟外部冲击
    
    参数:
    - frequency: 推力频率 (Hz)
    - impulse_magnitude_range: 冲量范围 (N·s)
    - impulse_direction: 推力方向 ('random', 'forward', 'side')
    """
    
    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        )
        
        # 配置参数
        self.frequency = float(cfg.params.get("frequency", 0.5))      # Hz
        self.impulse_magnitude = float(cfg.params.get(
            "impulse_magnitude", 50.0
        ))  # N·s
        self.direction = cfg.params.get("impulse_direction", "random")
        
        # 计算执行周期
        self.timestep = env.step_dt
        self.apply_period = int(1.0 / (self.frequency * self.timestep))
        
        # 初始化计数器
        self.counter = 0
        
    def __call__(self, env):
        """每步检查是否施加推力"""
        self.counter += 1
        
        if self.counter < self.apply_period:
            return  # 还没到执行时间
        
        self.counter = 0
        
        # 获取机器人资产
        asset: Articulation = env.scene[self.asset_cfg.name]
        device = env.device
        
        # 随机选择哪些环境施加推力
        num_envs = env.num_envs
        push_mask = torch.rand(num_envs, device=device) < 0.3  # 30% 概率
        
        # 生成推力方向和大小
        if self.direction == "random":
            # 随机方向 (水平平面内)
            angles = torch.rand(num_envs, device=device) * 2 * np.pi
            push_force_x = torch.cos(angles)
            push_force_y = torch.sin(angles)
        elif self.direction == "forward":
            push_force_x = torch.ones(num_envs, device=device)
            push_force_y = torch.zeros(num_envs, device=device)
        elif self.direction == "side":
            push_force_x = torch.zeros(num_envs, device=device)
            push_force_y = torch.ones(num_envs, device=device)
        
        # 生成冲量大小 (随机化)
        impulse_magnitudes = (
            torch.rand(num_envs, device=device) * self.impulse_magnitude
        )
        
        # 构造推力 [N·s]
        push_impulse = torch.zeros(num_envs, 3, device=device)
        push_impulse[:, 0] = push_force_x * impulse_magnitudes
        push_impulse[:, 1] = push_force_y * impulse_magnitudes
        
        # 仅对选中的环境施加
        push_impulse = push_impulse * push_mask.unsqueeze(-1)
        
        # 应用到机器人基座
        # 注意: Isaac Lab 中冲量通过增加速度实现
        # v_new = v_old + impulse / mass
        mass = asset.data.body_mass[:, 0]  # 基座质量
        lin_vel_increment = push_impulse / mass.unsqueeze(-1)
        
        asset.data.root_lin_vel_w += lin_vel_increment


# 事件配置
push_event_cfg = EventTermCfg(
    func=RandomPushEvent,
    mode="reset",  # "reset" 或 "start"
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "frequency": 0.5,           # 每 2 秒一次推力
        "impulse_magnitude": 50.0,  # N·s
        "impulse_direction": "random",
    }
)
```

#### Step 3: 阈值逐级增加的测试

**文件**: `scripts/rsl_rl/test_robustness.py` (新建)

```python
import torch
import numpy as np
from collections import deque

class RobustnessEvaluator:
    """逐级增加冲量测试机器人鲁棒性"""
    
    def __init__(self, env, policy_network):
        self.env = env
        self.policy = policy_network
        self.device = env.device
        
        # 冲量阈值列表 (N·s)
        self.impulse_levels = np.linspace(0, 200, 20)  # 0-200 N·s
        self.results = {}
    
    def test_single_impulse_level(self, impulse_magnitude, num_trials=5):
        """
        测试单一冲量级别
        
        返回:
        - success_rate: 不摔倒的比例
        - recovery_time: 平均恢复时间 (秒)
        - mse_increase: MSE 增长幅度
        """
        success_count = 0
        recovery_times = []
        mse_increases = []
        
        for trial in range(num_trials):
            obs, _ = self.env.reset()
            
            # 运行无干扰的基准测试 (3 秒)
            baseline_errors = []
            for _ in range(600):  # 3 秒 @ 200 Hz
                with torch.no_grad():
                    action = self.policy(obs.unsqueeze(0))
                obs, reward, done, _ = self.env.step(action)
                
                if not done:
                    asset = self.env.scene["robot"]
                    lin_vel_actual = asset.data.root_lin_vel_w[:, :2]
                    lin_vel_command = self.env.command_manager.get_command(
                        "base_velocity"
                    )[:, :2]
                    error = torch.norm(lin_vel_actual - lin_vel_command)
                    baseline_errors.append(error.item())
            
            baseline_mse = np.mean([e**2 for e in baseline_errors])
            
            # 施加推力
            self._apply_impulse(impulse_magnitude)
            
            # 检查是否摔倒
            recovery_frames = 0
            for frame in range(1500):  # 恢复期最长 7.5 秒
                with torch.no_grad():
                    action = self.policy(obs.unsqueeze(0))
                obs, reward, done, _ = self.env.step(action)
                
                # 检查姿态
                asset = self.env.scene["robot"]
                base_quat = asset.data.root_quat_w
                roll, pitch = self._quat_to_euler(base_quat)
                
                if abs(roll) > 0.785 or abs(pitch) > 0.785:  # 45°
                    break  # 摔倒了
                
                # 计算恢复指标
                lin_vel_actual = asset.data.root_lin_vel_w[:, :2]
                lin_vel_command = self.env.command_manager.get_command(
                    "base_velocity"
                )[:, :2]
                error = torch.norm(lin_vel_actual - lin_vel_command)
                
                if error < baseline_mse ** 0.5 * 1.2:  # 误差恢复到 120%
                    recovery_frames = frame
                    break
            
            # 统计结果
            if recovery_frames > 0:  # 没有摔倒
                success_count += 1
                recovery_times.append(recovery_frames * self.env.step_dt)
        
        success_rate = success_count / num_trials
        mean_recovery = np.mean(recovery_times) if recovery_times else float('inf')
        
        return {
            "success_rate": success_rate,
            "recovery_time": mean_recovery,
            "impulse_magnitude": impulse_magnitude,
        }
    
    def run_all_levels(self):
        """运行所有冲量级别的测试"""
        max_impulse = 0
        
        for impulse in self.impulse_levels:
            result = self.test_single_impulse_level(impulse, num_trials=3)
            self.results[impulse] = result
            
            print(f"Impulse: {impulse:.1f} N·s | "
                  f"Success: {result['success_rate']*100:.0f}% | "
                  f"Recovery: {result['recovery_time']:.2f}s")
            
            # 找到最大可承受冲量
            if result["success_rate"] > 0.8:
                max_impulse = impulse
        
        return max_impulse
    
    def _apply_impulse(self, magnitude):
        """对机器人施加推力"""
        asset = self.env.scene["robot"]
        
        # 随机方向
        angle = np.random.rand() * 2 * np.pi
        force_x = np.cos(angle) * magnitude
        force_y = np.sin(angle) * magnitude
        
        mass = asset.data.body_mass[0, 0]
        lin_vel_increment = torch.tensor(
            [force_x / mass, force_y / mass, 0.0],
            device=self.device
        )
        
        asset.data.root_lin_vel_w += lin_vel_increment.unsqueeze(0)
    
    def _quat_to_euler(self, quat):
        """四元数转欧拉角 (简化版)"""
        # 只返回 roll 和 pitch
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        sin_roll = 2 * (w * x + y * z)
        cos_roll = 1 - 2 * (x**2 + y**2)
        roll = torch.atan2(sin_roll, cos_roll)
        
        sin_pitch = 2 * (w * y - z * x)
        pitch = torch.asin(sin_pitch)
        
        return roll, pitch
```

### 2.3.3 预期挑战与解决方案

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 推力后摔倒 | 成功率 < 50% | 增加 `feet_regulation` 权重; 增加 `stiffness`/`damping` |
| 恢复缓慢 | 恢复时间 > 2.0s | 增加 `base_tracking` 权重; 调整奖励函数 |
| 参数过度随机化 | 正常行走崩溃 | 缩小随机化范围; 使用课程学习 |

### 2.3.4 推荐参数

```python
# 域随机化范围
stiffness_range = (20.0, 30.0)      # ±20%
damping_range = (0.6, 1.2)          # ±50%
mass_range = (2.0, 3.0)             # ±20%
friction_range = (0.3, 1.5)         # 5倍变化
gravity_range = (9.0, 10.0)         # ±2%

# 推力参数
impulse_frequency = 0.5             # 每 2 秒一次
impulse_magnitude_range = (0, 150)  # N·s
```

---

## Task 2.4: 复杂地形适应 (Terrain Traversal)

### 2.4.1 任务描述

**目标**: 机器人在混合地形上从起点 A 到达终点 B

**地形组合**:

```
起点 A
  ↓ (平地, 10m)
斜坡上升 (15°, 5m)
  ↓
平台 (2m × 2m)
  ↓
台阶下降 (3 阶, 0.2m/阶)
  ↓
离散踏脚石 (5 块)
  ↓
斜坡下降 (20°, 5m)
  ↓
终点 B
```

**评分标准**:

```
总分 = 通过率×50 + 地形适应性×30 + 速度×20

1. 通过率 (Primary, 50%)
   成功到达终点的百分比
   目标: 通过率 > 80%

2. 地形适应性 (Secondary, 30%)
   - 地形切换时的步态平滑度
   - 足部冲击力的均匀性
   目标: 无剧烈的加速度突跳

3. 速度 (Tertiary, 20%)
   - 完成时间
   - 平均速度
   目标: 尽量快速通过
```

### 2.4.2 实现步骤

#### Step 1: 创建混合地形

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_terrain_env_cfg.py`

```python
from dataclasses import dataclass
from isaaclab.sim.schemas.terrain import TerrainGeneratorCfg
from isaaclab.terrains import (
    DiamondSquareCfg,
    HillsCfg,
    PyramidSlopedTerrainCfg,
    TerrainImporterCfg,
)
from bipedal_locomotion.tasks.locomotion.cfg.PF import limx_flat_env_cfg

@dataclass
class TerrainTraversalEnvCfg(limx_flat_env_cfg.FlatTerrainEnvCfg):
    """复杂地形环境配置"""
    
    def __init__(self):
        super().__init__()
        
        # ===== 1. 配置地形生成器 =====
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ComplexTerrainGeneratorCfg(),
            max_init_terrain_level=0,  # 从简单地形开始
            terrain_resampling_interval=100,  # 每 100 steps 重新采样地形
            visual_material=None,
        )
        
        # ===== 2. 调整奖励权重 (复杂地形) =====
        self.rewards.stay_alive.weight = 1.0        # 高权重（易摔倒）
        self.rewards.base_tracking.weight = 0.5     # 低权重（速度不是首要）
        self.rewards.feet_regulation.weight = -0.3  # 重惩罚（足部稳定性重要）
        self.rewards.gait_reward.weight = 2.0       # 高权重（步态规划重要）
        
        # 新增奖励: 地形适应性
        self.rewards.terrain_adaptation = RewardTermCfg(
            func=mdp.TerrainAdaptationReward,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            }
        )
        
        # ===== 3. 高度扫描器配置 (用于环境感知) =====
        self.scene.height_scanner.enabled = True
        self.scene.height_scanner.resolution = 0.05  # 5cm 分辨率
        self.scene.height_scanner.size = [2.0, 2.0]  # 2m × 2m 范围
        self.scene.height_scanner.max_range = 0.5    # 50cm 高度范围
        
        # ===== 4. 课程学习配置 =====
        self.curriculum = CurriculumCfg(
            enable=True,
            stages=[
                {"name": "flat", "duration": 2000, "terrain_level": 0},
                {"name": "gentle_slopes", "duration": 2000, "terrain_level": 1},
                {"name": "moderate_slopes", "duration": 2000, "terrain_level": 2},
                {"name": "stairs", "duration": 2000, "terrain_level": 3},
                {"name": "complex", "duration": 5000, "terrain_level": 4},
            ]
        )


@dataclass
class ComplexTerrainGeneratorCfg(TerrainGeneratorCfg):
    """
    复杂地形生成器配置
    
    地形难度等级:
    0: 平面
    1: 缓坡 (10°)
    2: 陡坡 (20°)
    3: 台阶
    4: 混合 (坡 + 台阶 + 离散块)
    """
    
    def __init__(self, terrain_level=4):
        super().__init__()
        
        self.terrain_level = terrain_level
        
        # 基础配置
        self.size = [200.0, 200.0]              # 地形大小 (200m × 200m)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.vertical_scale = 1.0
        self.horizontal_scale = 1.0
        
        # 地形参数
        if terrain_level >= 1:
            # 斜坡地形
            self.slope_terrain = HillsCfg(
                size=[100.0, 100.0],
                slope_range=[0.0, 0.3],  # 最高 30° 斜坡
                step_height_range=[0.0, 0.2],  # 最高 0.2m 台阶
                platform_size=[2.0, 2.0],
                randomize_start=True,
            )
        
        if terrain_level >= 3:
            # 台阶地形
            self.stairs_terrain = PyramidSlopedTerrainCfg(
                size=[20.0, 20.0],
                step_height_range=[0.1, 0.3],  # 0.1-0.3m 高度
                step_width=0.5,                # 0.5m 宽度
            )
        
        if terrain_level >= 4:
            # 离散地形（随机块）
            self.discrete_terrain = DiamondSquareCfg(
                size=[100.0, 100.0],
                terrain_scale=0.2,
                min_height=-0.1,
                max_height=0.3,
                step=0.02,
            )
```

#### Step 2: 地形适应奖励

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py`

```python
class TerrainAdaptationReward(ManagerTermBase):
    """
    地形适应奖励 - 鼓励平滑的地形过渡
    
    指标:
    1. 足部冲击力的平滑性 (加速度)
    2. 步态规律性
    3. 能耗效率
    """
    
    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        )
        self.sensor_cfg: SceneEntityCfg = cfg.params.get(
            "sensor_cfg", SceneEntityCfg("height_scanner")
        )
        
        # 历史缓冲 (用于计算加速度)
        self.prev_foot_forces = None
        self.prev_prev_foot_forces = None
    
    def __call__(self, env) -> torch.Tensor:
        """计算地形适应奖励"""
        asset: Articulation = env.scene[self.asset_cfg.name]
        device = env.device
        
        # ===== 1. 足部冲击力平滑性 =====
        # 获取足部接触力
        contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        foot_forces = torch.norm(
            contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids],
            dim=-1
        )
        
        # 初始化历史
        if self.prev_foot_forces is None:
            self.prev_foot_forces = foot_forces.clone()
            self.prev_prev_foot_forces = foot_forces.clone()
            force_smoothness = torch.ones(env.num_envs, device=device)
        else:
            # 计算二阶导数 (加速度)
            force_accel = (
                foot_forces - 2 * self.prev_foot_forces + self.prev_prev_foot_forces
            )
            
            # 平滑性得分: 加速度越小越好
            force_smoothness = torch.exp(-(torch.abs(force_accel) ** 2) / 100)
            
            # 更新历史
            self.prev_prev_foot_forces = self.prev_foot_forces
            self.prev_foot_forces = foot_forces
        
        # ===== 2. 高度变化感知 =====
        # 获取前方地形高度
        height_scanner: RayCaster = env.scene.sensors[self.sensor_cfg.name]
        heights = height_scanner.data.ray_hits_w  # [num_envs, num_rays]
        
        # 计算前方地形的标准差 (高度变化剧烈度)
        height_std = torch.std(heights, dim=1)
        
        # 根据地形难度调整响应
        # 地形变化大 → 奖励更高的自适应
        height_awareness = torch.exp(-height_std / 0.2)
        
        # ===== 3. 综合奖励 =====
        reward = (
            force_smoothness.mean(dim=1 if len(force_smoothness.shape) > 1 else -1) * 0.6 +
            height_awareness * 0.4
        )
        
        return reward
```

#### Step 3: 课程学习策略

**文件**: `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py`

```python
from isaaclab.managers import CurriculumTermCfg

class TerrainDifficultyCurriculum:
    """逐步增加地形难度的课程学习"""
    
    def __init__(self, env):
        self.env = env
        self.progress = 0.0  # 0-1
        self.terrain_level = 0
        
        # 课程阶段定义
        self.stages = [
            {"name": "flat", "terrain_level": 0, "duration": 2000},
            {"name": "gentle_slopes", "terrain_level": 1, "duration": 2000},
            {"name": "moderate_slopes", "terrain_level": 2, "duration": 2000},
            {"name": "stairs", "terrain_level": 3, "duration": 2000},
            {"name": "complex", "terrain_level": 4, "duration": 5000},
        ]
        
        self.current_stage = 0
        self.frames_in_stage = 0
    
    def update(self, episode_length_frames):
        """更新课程进度"""
        self.frames_in_stage += 1
        
        # 计算当前阶段的目标帧数
        current_stage_duration = self.stages[self.current_stage]["duration"]
        
        # 检查是否应该进入下一阶段
        if self.frames_in_stage >= current_stage_duration:
            self.current_stage = min(
                self.current_stage + 1, len(self.stages) - 1
            )
            self.frames_in_stage = 0
        
        # 更新地形难度
        self.terrain_level = self.stages[self.current_stage]["terrain_level"]
        
        # 计算总体进度
        total_frames = sum(s["duration"] for s in self.stages)
        self.progress = min(
            (self.current_stage * 2000 + self.frames_in_stage) / total_frames,
            1.0
        )
    
    def get_reward_scale(self):
        """根据课程阶段返回奖励缩放因子"""
        scales = {
            0: {"base_tracking": 2.0, "stay_alive": 0.2},
            1: {"base_tracking": 1.5, "stay_alive": 0.5},
            2: {"base_tracking": 1.0, "stay_alive": 1.0},
            3: {"base_tracking": 0.5, "stay_alive": 2.0},
            4: {"base_tracking": 0.3, "stay_alive": 3.0},
        }
        return scales.get(self.terrain_level, scales[4])


curriculum_cfg = CurriculumTermCfg(
    func=TerrainDifficultyCurriculum,
    mode="episode",
)
```

### 2.4.3 预期挑战与解决方案

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 地形过难导致崩溃 | 摔倒率 > 80% | 使用课程学习，从简单地形开始 |
| 步态在地形切换处震荡 | 高度突跳时速度波动 | 增加 `gait_reward` 权重; 增加高度感知观测 |
| 机器人卡在障碍物 | 无法越过台阶 | 增加 `stay_alive` 权重鼓励尝试 |
| 训练时间过长 | 需要数小时收敛 | 增加 `num_envs` (如显存允许) |

### 2.4.4 关键参数表

```python
# 地形参数
max_slope = 0.3              # 最大斜坡 30°
max_step_height = 0.3        # 最大台阶高度 0.3m
step_width = 0.5             # 台阶宽度 0.5m
discrete_block_size = 0.2    # 离散块大小 0.2m

# 奖励权重（复杂地形）
stay_alive = 1.0             # 高权重（易摔倒）
base_tracking = 0.5          # 低权重（速度非首要）
gait_reward = 2.0            # 高权重（步态重要）
terrain_adaptation = 0.5     # 新增权重

# 课程学习
num_stages = 5
stage_duration = [2000, 2000, 2000, 2000, 5000]
```

---

## 通用学习资源与建议

### 核心算法理论

#### 1. PPO 算法深度理解

**必读论文**:
- [Schulman et al., 2017] "Proximal Policy Optimization Algorithms" 
  - 链接: https://arxiv.org/pdf/1707.06347.pdf
  - 关键概念: PPO 目标函数、截断、优势函数

**推荐教程**:
- OpenAI Spinning Up - PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- 中文讲解: https://www.bilibili.com/video/BV1c7411N7uc

#### 2. Actor-Critic 框架

**论文**:
- [Konda & Tsitsiklis, 2000] "Actor-Critic Algorithms"
- [Schulman et al., 2015] "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

**代码参考**:
- OpenAI Baselines: https://github.com/openai/baselines
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3

### 机器人控制相关

#### 1. 双足行走理论

**教科书**:
- Russ Tedrake (MIT) "Underactuated Robotics" - 免费在线版
  - 链接: http://underactuated.csail.mit.edu/
  - 特别推荐 Chapter 5-7 (双足动力学)

**论文**:
- 步态分析: [Perry & Burnfield, 2010] "Gait Analysis: Normal and Pathological Function"

#### 2. 强化学习在机器人中的应用

**关键资源**:
- DeepMind Control Suite: https://github.com/deepmind/dm_control
- NVIDIA Isaac Lab 官方教程: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Berkeley CS294-112 Deep RL: http://rail.eecs.berkeley.edu/deeprlcourse/

### Isaac Lab 专业知识

#### 1. 官方文档

- **完整 API**: https://isaac-sim.github.io/IsaacLab/
- **教程集合**: https://github.com/isaac-sim/IsaacLab/tree/main/docs
- **示例代码**: https://github.com/isaac-sim/IsaacLab/tree/main/source/examples

#### 2. 关键概念

| 概念 | 资源 | 难度 |
|------|------|------|
| USD 文件格式 | https://graphics.pixar.com/usd/docs/ | 中 |
| 物理仿真 | Isaac 官方教程 | 中 |
| 域随机化 | https://arxiv.org/pdf/1703.06907.pdf | 难 |
| 传感器建模 | Isaac 官方 API 文档 | 中 |

### 实战建议

#### 1. 调试技巧

```python
# 在训练脚本中添加诊断代码
import matplotlib.pyplot as plt

def diagnose_training(env, policy, n_episodes=10):
    """诊断训练过程中的关键指标"""
    
    metrics = {
        "reward": [],
        "lin_vel_error": [],
        "ang_vel_error": [],
        "joint_torque": [],
        "episode_length": [],
    }
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward.mean().item()
            
            # 收集诊断数据
            # ... (代码略)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    for i, (key, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(values)
        plt.title(key)
    plt.tight_layout()
    plt.savefig("diagnosis.png")
```

#### 2. 参数调优工作流

```
Step 1: 基准配置 (默认参数)
  - 运行 1000 iterations
  - 记录 reward 曲线

Step 2: 识别瓶颈
  - 速度不达标? → 增加 w_vel
  - 步态不稳? → 增加 w_gait
  - 能耗过高? → 增加 w_smooth

Step 3: 精细调优
  - 微调权重 ±10-20%
  - A/B 对比测试

Step 4: 验证
  - 在测试地形评估性能
  - 记录最优参数配置
```

#### 3. 常见错误及排查

| 错误 | 表现 | 原因 | 解决方案 |
|------|------|------|---------|
| CUDA OOM | 显存溢出 | `num_envs` 过大 | 减少环境数量 |
| 梯度爆炸 | Loss → NaN | 学习率过高 | 降低 LR 或用梯度裁剪 |
| 收敛停滞 | Reward 平坦 | 奖励信号不足 | 增加奖励权重或改进设计 |
| 物理不稳定 | 机器人乱动 | PD 参数不合理 | 调整 `stiffness`/`damping` |

---

## 总结表：三个任务对比

| 方面 | Task 2.2 | Task 2.3 | Task 2.4 |
|------|----------|----------|----------|
| **难度** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **关键技术** | 速度追踪 | 域随机化 | 课程学习 |
| **主要奖励** | base_tracking | feet_regulation | gait_reward |
| **评估时间** | 1 分钟 | 2-3 分钟 | 5-10 分钟 |
| **实现工作量** | 中 | 中-高 | 高 |
| **学习资源** | PPO/速度控制 | 域随机化 | 课程学习 |

---

**最后修改**: 2024-12-17  
**作者**: 强化学习团队  
**维护者**: 机器人开发小组
