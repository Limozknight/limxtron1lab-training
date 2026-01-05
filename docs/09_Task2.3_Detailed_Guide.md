# 任务 2.3 详细指南：抗干扰鲁棒性测试（Disturbance Rejection）

**目标受众**：想要让机器人变得"抗揍"的你

**预计学习时间**：3-4 小时理论 + 2-3 天实践

**难度等级**：⭐⭐⭐（难度中等，但概念新颖）

---

## 第一部分：什么是抗干扰鲁棒性？

### 生活类比

```
想象你在走路时：

正常情况：
  • 平地上走路，没有人推你
  • 你可以稳定前进
  → 这是任务 2.2 的场景

有人推你：
  • 突然有人从侧面推你一把
  • 你会失衡，但经过调整后恢复平衡
  • 继续前进
  → 这是任务 2.3 的场景

任务 2.3 就是训练机器人：
  "被推了之后，不要摔倒，快速恢复平衡继续走"
```

### 为什么需要抗干扰能力？

```
现实世界中的机器人会遇到：

1. 外力干扰
   ├─ 风吹
   ├─ 人或物体撞击
   ├─ 地面突然滑动
   └─ 负载突然变化（背包掉了一个东西）

2. 模型误差
   ├─ 仿真和现实的差异
   ├─ 执行器的延迟
   └─ 传感器的噪声

3. 不确定性
   ├─ 地面摩擦系数变化
   ├─ 关节磨损
   └─ 电池电量下降

如果机器人没有抗干扰能力：
  • 仿真中走得很好
  • 现实中一碰就倒
  → Sim-to-Real 转移失败 ✗

如果机器人有抗干扰能力：
  • 遇到扰动能自动调整
  • 快速恢复稳定
  → 真实世界中实用 ✓
```

---

## 第二部分：核心概念详解

### 2.1 什么是"推力冲量"（Impulse）？

```
物理定义：
  冲量 = 力 × 时间
  单位：牛顿·秒 (N·s)

例子：
  • 小推力，长时间：10 N × 0.5 s = 5 N·s
  • 大推力，短时间：50 N × 0.1 s = 5 N·s
  → 两者冲量相同，但感觉不同

在任务 2.3 中：
  系统会随机施加瞬时推力
  • 方向：随机（前、后、左、右）
  • 大小：随机（5-100 N）
  • 时间：很短（通常 0.1-0.2 秒）
  
  机器人需要：
  1. 感知到被推了（通过加速度或速度变化）
  2. 快速调整姿态和步态
  3. 恢复原来的行走模式
```

### 2.2 Domain Randomization（域随机化）

```
核心思想：
  "如果训练时见过各种各样的情况，
   部署时遇到新情况也不会慌"

具体做法：
  在训练时，随机改变环境参数：
  
  物理参数随机化：
    • 质量：± 20%
    • 摩擦系数：0.5 - 2.0
    • 关节阻尼：± 50%
    • 执行器延迟：0-10 ms
  
  外力随机化：
    • 方向：360°
    • 大小：0-100 N
    • 频率：随机间隔
  
  观测随机化：
    • 传感器噪声：± 5%
    • 延迟：0-5 ms
  
  结果：
    神经网络学会了"通用策略"
    → 对各种情况都有应对方法
    → 鲁棒性提升
```

### 2.3 恢复速度（Recovery Speed）

```
定义：
  从受到干扰到恢复稳定状态的时间

评价指标：
  
  1. 速度恢复时间
     受扰后，速度回到命令值 ± 10% 的时间
     • < 1 秒：优秀 ✓✓✓
     • 1-2 秒：良好 ✓✓
     • 2-5 秒：及格 ✓
     • > 5 秒：较差 △
  
  2. 姿态恢复时间
     Roll/Pitch 回到 ± 5° 范围的时间
     • < 0.5 秒：优秀
     • 0.5-1 秒：良好
     • > 1 秒：需改进
  
  3. 步态恢复
     恢复正常的步态周期
     通常 2-3 步 (1-2 秒)
```

---

## 第三部分：实现抗干扰的技术方法

### 方法 1：Domain Randomization（推荐） ⭐⭐⭐

**原理**：在训练时随机施加外力，让神经网络学会应对

**优点**：
- ✓ 简单有效
- ✓ 自动学习最优反应
- ✓ 泛化能力强

**实现步骤**：

#### 步骤 1：在配置中启用外力随机化

**文件**：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py`

```python
from isaaclab.managers import EventTermCfg

@configclass
class EventsCfg:
    """环境事件配置"""
    
    # 原有的事件...
    reset_base = EventTerm(...)
    reset_robot_joints = EventTerm(...)
    
    # 新增：随机外力干扰
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",           # 间隔模式
        interval_range_s=(5, 10),  # 每 5-10 秒推一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-2.0, 2.0),   # X 方向速度扰动 (m/s)
                "y": (-2.0, 2.0),   # Y 方向速度扰动
                "z": (0.0, 0.0),    # Z 方向不扰动（不让它飞起来）
                "roll": (-0.5, 0.5),   # Roll 扰动 (rad/s)
                "pitch": (-0.5, 0.5),  # Pitch 扰动
                "yaw": (-1.0, 1.0),    # Yaw 扰动
            },
        },
    )
    
    # 或者：直接施加力
    apply_external_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3, 8),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-100.0, 100.0),    # 力的范围 (N)
            "torque_range": (-50.0, 50.0),     # 力矩范围 (N·m)
        },
    )
```

#### 步骤 2：实现外力施加函数

**文件**：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/events.py`

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

def push_by_setting_velocity(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_range: dict = None,
):
    """通过设置速度来模拟推力"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 生成随机速度扰动
    num_envs = len(env_ids)
    device = asset.data.root_lin_vel_w.device
    
    # 线速度扰动
    vel_x = torch.FloatTensor(num_envs).uniform_(
        velocity_range["x"][0], 
        velocity_range["x"][1]
    ).to(device)
    
    vel_y = torch.FloatTensor(num_envs).uniform_(
        velocity_range["y"][0], 
        velocity_range["y"][1]
    ).to(device)
    
    vel_z = torch.zeros(num_envs, device=device)
    
    # 角速度扰动
    ang_vel_x = torch.FloatTensor(num_envs).uniform_(
        velocity_range["roll"][0], 
        velocity_range["roll"][1]
    ).to(device)
    
    ang_vel_y = torch.FloatTensor(num_envs).uniform_(
        velocity_range["pitch"][0], 
        velocity_range["pitch"][1]
    ).to(device)
    
    ang_vel_z = torch.FloatTensor(num_envs).uniform_(
        velocity_range["yaw"][0], 
        velocity_range["yaw"][1]
    ).to(device)
    
    # 应用扰动（叠加到当前速度上）
    asset.data.root_lin_vel_w[env_ids] += torch.stack([vel_x, vel_y, vel_z], dim=1)
    asset.data.root_ang_vel_w[env_ids] += torch.stack([ang_vel_x, ang_vel_y, ang_vel_z], dim=1)


def apply_external_force_torque(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_range: tuple = (-100.0, 100.0),
    torque_range: tuple = (-50.0, 50.0),
):
    """直接施加外力和力矩"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    num_envs = len(env_ids)
    device = asset.data.root_pos_w.device
    
    # 随机生成力和力矩
    force = torch.FloatTensor(num_envs, 3).uniform_(
        force_range[0], 
        force_range[1]
    ).to(device)
    
    torque = torch.FloatTensor(num_envs, 3).uniform_(
        torque_range[0], 
        torque_range[1]
    ).to(device)
    
    # 施加到机器人基座
    # 注意：Isaac Lab 的具体 API 可能不同，需要查阅文档
    asset.set_external_force_and_torque(
        forces=force,
        torques=torque,
        env_ids=env_ids,
        body_ids=asset_cfg.body_ids,
    )
```

---

### 方法 2：添加"抗干扰"奖励（辅助） ⭐⭐

即使有 Domain Randomization，添加明确的奖励项也能加速学习。

#### 奖励 1：惩罚加速度变化

```python
def acceleration_penalty(env: ManagerBasedRLEnv,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                       ) -> torch.Tensor:
    """惩罚大的加速度（表示不稳定）"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取当前速度
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    
    # 计算加速度（速度的变化率）
    if not hasattr(env, "_prev_lin_vel"):
        env._prev_lin_vel = lin_vel.clone()
        env._prev_ang_vel = ang_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    lin_acc = (lin_vel - env._prev_lin_vel) / env.step_dt
    ang_acc = (ang_vel - env._prev_ang_vel) / env.step_dt
    
    # 更新历史
    env._prev_lin_vel = lin_vel.clone()
    env._prev_ang_vel = ang_vel.clone()
    
    # 惩罚大的加速度
    lin_acc_penalty = torch.sum(torch.square(lin_acc), dim=1)
    ang_acc_penalty = torch.sum(torch.square(ang_acc), dim=1)
    
    return lin_acc_penalty + ang_acc_penalty


def base_stability_reward(env: ManagerBasedRLEnv,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        ) -> torch.Tensor:
    """奖励稳定的基座姿态"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取基座的 Roll 和 Pitch
    quat = asset.data.root_quat_w
    euler = quat_to_euler_xyz(quat)  # 需要实现这个转换
    
    roll = euler[:, 0]
    pitch = euler[:, 1]
    
    # 期望：Roll 和 Pitch 接近 0
    stability_reward = torch.exp(-(roll**2 + pitch**2) / 0.1)
    
    return stability_reward
```

#### 在配置中注册

```python
@configclass
class RewardsCfg:
    """奖励配置"""
    
    # 原有奖励...
    base_tracking = RewTerm(func=mdp.base_tracking, weight=1.0)
    
    # 新增：抗干扰相关奖励
    acceleration_penalty = RewTerm(
        func=mdp.acceleration_penalty,
        weight=-0.05,  # 惩罚加速度变化
    )
    
    base_stability = RewTerm(
        func=mdp.base_stability_reward,
        weight=0.3,  # 奖励稳定姿态
    )
```

---

### 方法 3：增加观测项（让神经网络"感知"干扰）⭐

如果神经网络能"感知"到被推了，它能更好地反应。

```python
def base_linear_acceleration(env: ManagerBasedRLEnv,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                           ) -> torch.Tensor:
    """观测基座的线加速度"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    lin_vel = asset.data.root_lin_vel_w
    
    if not hasattr(env, "_prev_obs_lin_vel"):
        env._prev_obs_lin_vel = lin_vel.clone()
        return torch.zeros_like(lin_vel)
    
    lin_acc = (lin_vel - env._prev_obs_lin_vel) / env.step_dt
    env._prev_obs_lin_vel = lin_vel.clone()
    
    return lin_acc


def base_angular_acceleration(env: ManagerBasedRLEnv,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                            ) -> torch.Tensor:
    """观测基座的角加速度"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    ang_vel = asset.data.root_ang_vel_w
    
    if not hasattr(env, "_prev_obs_ang_vel"):
        env._prev_obs_ang_vel = ang_vel.clone()
        return torch.zeros_like(ang_vel)
    
    ang_acc = (ang_vel - env._prev_obs_ang_vel) / env.step_dt
    env._prev_obs_ang_vel = ang_vel.clone()
    
    return ang_acc
```

在配置中注册观测：

```python
@configclass
class PolicyCfg(ObsGroup):
    """策略观测"""
    
    # 原有观测...
    proj_gravity = ObsTerm(func=mdp.projected_gravity, ...)
    
    # 新增：加速度观测
    base_lin_acc = ObsTerm(
        func=mdp.base_linear_acceleration,
        noise=GaussianNoise(mean=0.0, std=0.1),
        scale=0.1,  # 加速度可能很大，需要缩放
    )
    
    base_ang_acc = ObsTerm(
        func=mdp.base_angular_acceleration,
        noise=GaussianNoise(mean=0.0, std=0.05),
        scale=0.1,
    )
```

---

## 第四部分：逐步实现的完整流程

### 第 1 天：理解和基础设置

```
上午：理解抗干扰概念
  ☐ 读懂什么是冲量
  ☐ 理解 Domain Randomization 的原理
  ☐ 了解评分标准（最大冲量 + 恢复速度）

下午：验证基础训练
  ☐ 确保任务 2.2 的模型能稳定运行
  ☐ 测试在无干扰情况下的性能
  ☐ 记录基准指标（速度误差、姿态稳定性）
```

**验证命令**：

```bash
# 运行任务 2.2 的已训练模型
python scripts/play.py --task=PointFootLocomotion \
    --checkpoint=logs/PointFootLocomotion-*/model.pt \
    --num_envs=256

# 应该看到：
# - 机器人稳定行走
# - 速度误差 < 0.1 m/s
# - 不摔倒
```

---

### 第 2 天：添加外力干扰

```
上午：实现外力函数
  ☐ 在 events.py 中实现 push_by_setting_velocity()
  ☐ 或实现 apply_external_force_torque()
  ☐ 测试函数是否正确（手动调用）

下午：配置事件系统
  ☐ 在 limx_base_env_cfg.py 中添加 EventsCfg
  ☐ 配置外力参数（频率、大小）
  ☐ 启动训练，观察机器人被推的情况

晚上：初步训练
  ☐ 训练 50-100 轮
  ☐ 观察机器人是否学会应对干扰
```

**期望结果**：

```
初期（前 20 轮）：
  • 机器人被推后经常摔倒
  • 奖励显著下降（相比无干扰）
  • 学习曲线波动较大

中期（20-50 轮）：
  • 机器人开始适应干扰
  • 摔倒次数减少
  • 奖励逐渐恢复

后期（50-100 轮）：
  • 机器人能承受小到中等的推力
  • 恢复速度加快
  • 奖励接近无干扰时的水平
```

---

### 第 3 天：优化和增强

```
上午：添加抗干扰奖励
  ☐ 实现 acceleration_penalty()
  ☐ 实现 base_stability_reward()
  ☐ 调整奖励权重

下午：添加加速度观测
  ☐ 实现加速度观测函数
  ☐ 注册到观测配置
  ☐ 重新训练

晚上：测试最大冲量
  ☐ 逐步增加推力大小
  ☐ 找到机器人能承受的最大冲量
  ☐ 记录恢复速度
```

**测试脚本**（可以修改 play.py）：

```python
# 在 play.py 中添加测试逻辑

def test_disturbance_rejection(env, policy, max_impulse=200.0):
    """测试抗干扰能力"""
    
    results = {
        "impulses": [],
        "success": [],
        "recovery_time": [],
    }
    
    # 测试不同大小的冲量
    for impulse in range(10, int(max_impulse), 10):
        # 重置环境
        obs = env.reset()
        
        # 让机器人先稳定行走 2 秒
        for _ in range(400):  # 2s / 0.005s = 400 steps
            action = policy(obs)
            obs, _, _, _ = env.step(action)
        
        # 施加冲量
        force = impulse / 0.1  # 假设作用时间 0.1s
        apply_impulse_to_robot(env, force)
        
        # 观察恢复
        recovery_start = time.time()
        recovered = False
        max_time = 5.0  # 最多观察 5 秒
        
        while time.time() - recovery_start < max_time:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            
            # 检查是否恢复
            if is_stable(env):  # 需要定义稳定性判断
                recovered = True
                recovery_time = time.time() - recovery_start
                break
            
            if done:
                break
        
        # 记录结果
        results["impulses"].append(impulse)
        results["success"].append(recovered)
        results["recovery_time"].append(recovery_time if recovered else max_time)
        
        print(f"Impulse: {impulse} N·s, Success: {recovered}, "
              f"Recovery: {recovery_time:.2f}s")
    
    return results
```

---

## 第五部分：评分标准详解

### 评分指标 1：最大承受冲量

```
测试方法：
  1. 机器人稳定行走
  2. 施加不同大小的冲量（从小到大）
  3. 记录机器人能承受而不摔倒的最大冲量

评分标准：
  • > 150 N·s：100 分 ✓✓✓
  • 100-150 N·s：90 分 ✓✓
  • 50-100 N·s：70 分 ✓
  • 20-50 N·s：50 分 △
  • < 20 N·s：30 分 ✗

参考值：
  人类被推：约 30-50 N·s 会失衡但不倒
  轻度推力：10-30 N·s
  中度推力：30-80 N·s
  重度推力：80-150 N·s
  极端推力：> 150 N·s
```

### 评分指标 2：恢复速度

```
定义：
  从受到干扰到恢复稳定状态的时间

测量方法：
  1. 施加标准冲量（如 50 N·s）
  2. 开始计时
  3. 检查以下条件是否满足：
     ├─ 速度误差 < 0.15 m/s
     ├─ Roll/Pitch < 10°
     └─ 步态周期稳定
  4. 当所有条件满足时停止计时

评分标准：
  • < 1.0 秒：100 分 ✓✓✓
  • 1.0-2.0 秒：80 分 ✓✓
  • 2.0-3.0 秒：60 分 ✓
  • 3.0-5.0 秒：40 分 △
  • > 5.0 秒：20 分 ✗
```

---

## 第六部分：常见问题和解决方案

### ❌ 问题 1：机器人被推后立即摔倒

```
症状：
  • 即使很小的推力也会摔倒
  • 无法学会应对干扰
  • 奖励始终很低

原因分析：
  1. 推力太大太频繁（最可能）
  2. 基础步态不够稳定
  3. PD 参数不适合快速反应

解决方案：

✓ 方案 1：减小推力和频率
  # 在 limx_base_env_cfg.py 中
  push_robot = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      interval_range_s=(10, 15),  # 从 (5,10) 增加到 (10,15)
      params={
          "velocity_range": {
              "x": (-0.5, 0.5),  # 从 (-2.0, 2.0) 减小到 (-0.5, 0.5)
              "y": (-0.5, 0.5),
              ...
          }
      }
  )

✓ 方案 2：使用课程学习
  阶段 1：无干扰（训练 100 轮）
  阶段 2：小干扰（velocity_range = ±0.5, 训练 100 轮）
  阶段 3：中干扰（velocity_range = ±1.0, 训练 100 轮）
  阶段 4：大干扰（velocity_range = ±2.0, 训练 100 轮）

✓ 方案 3：增加 PD 刚度和阻尼
  # 在 pointfoot_cfg.py 中
  stiffness=35.0,  # 从 25 增加到 35
  damping=1.5,     # 从 0.8 增加到 1.5
  
  更硬的控制 → 更快的反应 → 更好的抗干扰
```

### ❌ 问题 2：恢复时间太长

```
症状：
  • 机器人被推后不摔倒
  • 但需要很长时间（> 5 秒）才能恢复
  • 恢复过程中姿态摇晃

原因分析：
  • 神经网络学到了"保守策略"（慢慢恢复）
  • 缺乏"快速恢复"的奖励信号

解决方案：

✓ 添加"恢复速度"奖励
  ```python
  def recovery_speed_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
      """奖励快速恢复到目标速度"""
      asset = env.scene["robot"]
      commands = env.command_manager.get_command("base_velocity")
      
      # 当前速度和目标速度
      current_vel = asset.data.root_lin_vel_w[:, :2]
      target_vel = commands[:, :2]
      
      # 速度误差
      vel_error = torch.norm(current_vel - target_vel, dim=1)
      
      # 奖励与误差成反比（误差小 = 恢复快 = 高奖励）
      reward = torch.exp(-vel_error ** 2 / 0.1)
      
      return reward
  ```
  
  在 RewardsCfg 中：
  ```python
  recovery_speed = RewTerm(
      func=mdp.recovery_speed_reward,
      weight=1.5,  # 高权重，鼓励快速恢复
  )
  ```

✓ 增加训练时的推力频率
  # 更频繁的干扰 → 更多练习机会
  interval_range_s=(3, 5),  # 从 (5,10) 减少到 (3,5)
```

### ❌ 问题 3：模型泛化能力差

```
症状：
  • 在训练时的推力大小下表现很好
  • 测试时用不同大小的推力就失败
  • 只能应对特定方向的推力

原因分析：
  • Domain Randomization 范围太窄
  • 过拟合到训练时的特定扰动

解决方案：

✓ 增加随机化范围
  ```python
  push_robot = EventTermCfg(
      params={
          "velocity_range": {
              "x": (-3.0, 3.0),  # 更大的范围
              "y": (-3.0, 3.0),
              "yaw": (-2.0, 2.0),
          }
      }
  )
  ```

✓ 添加更多类型的扰动
  ```python
  # 除了速度扰动，还加入：
  
  # 1. 质量随机化
  randomize_mass = EventTermCfg(
      func=mdp.randomize_rigid_body_mass,
      mode="startup",  # 每个 episode 开始时随机
      params={
          "mass_range": (0.8, 1.2),  # ±20%
      }
  )
  
  # 2. 摩擦系数随机化
  randomize_friction = EventTermCfg(
      func=mdp.randomize_friction,
      mode="startup",
      params={
          "friction_range": (0.5, 1.5),
      }
  )
  
  # 3. 执行器延迟
  randomize_actuator_lag = EventTermCfg(
      func=mdp.randomize_actuator_lag,
      mode="startup",
      params={
          "lag_range": (0.0, 0.02),  # 0-20ms 延迟
      }
  )
  ```
```

### ❌ 问题 4：训练不收敛

```
症状：
  • 训练 200 轮后奖励还是负数
  • 学习曲线波动剧烈
  • 无法看到明显的进步

原因分析：
  • 任务太难（干扰太强）
  • 奖励函数设计不合理
  • 学习率或其他超参数问题

解决方案：

✓ 检查任务难度
  # 临时禁用干扰，看基础性能
  # 如果无干扰时性能好，说明是干扰太强
  # 如果无干扰时也不好，说明是基础问题

✓ 调整奖励权重
  # 增加稳定性相关的奖励
  stay_alive = RewTerm(weight=2.0)  # 从 0.5 增加
  base_stability = RewTerm(weight=1.0)  # 新增
  
  # 降低可能冲突的奖励
  base_tracking = RewTerm(weight=0.5)  # 从 1.0 降低
  # 因为在应对干扰时，短暂偏离速度是可以接受的

✓ 使用预训练模型
  # 先在无干扰环境训练到收敛
  # 然后加载模型，在有干扰环境继续训练
  python scripts/train.py --task=PointFootLocomotion \
      --checkpoint=logs/.../model_no_disturbance.pt \
      --resume
```

---

## 第七部分：与任务 2.2、2.4 的关系

### 任务依赖关系

```
任务 2.2（平地速度跟随）
  ↓
  ├─ 学会基础的行走
  ├─ 学会速度控制
  └─ 建立稳定的步态
  
任务 2.3（抗干扰）← 你在这里
  ↓
  ├─ 基于 2.2 的能力
  ├─ 增加鲁棒性
  └─ 学会应对扰动
  
任务 2.4（复杂地形）
  ↓
  ├─ 需要 2.2 的行走能力
  ├─ 需要 2.3 的鲁棒性（地形变化类似扰动）
  └─ 学会地形适应
```

### 是否必须按顺序完成？

```
推荐顺序：2.2 → 2.3 → 2.4

理由：
  ✓ 2.2 提供了基础训练和调试经验
  ✓ 2.3 的模型可以直接用于 2.4
    （复杂地形本质上也是一种"扰动"）
  ✓ 循序渐进，避免一次面对太多挑战

可以跳过吗？

  直接做 2.4（跳过 2.2 和 2.3）：
    ✗ 不推荐
    • 难度太大，容易卡住
    • 调试困难，不知道问题出在哪
    • 学习曲线陡峭
  
  先做 2.2，跳过 2.3 直接做 2.4：
    △ 可行但有风险
    • 可能在地形切换时摔倒
    • 需要在 2.4 中额外添加鲁棒性训练
    • 训练时间可能更长
  
  建议：
    至少完成 2.2，熟悉训练流程
    如果时间紧，可以简化 2.3（只训练 50 轮）
    然后在 2.4 中一起训练鲁棒性和地形适应
```

---

## 第八部分：快速开始指南

### 最小化实现（2 小时）

如果你只想快速验证抗干扰功能：

```python
# 第 1 步：在 limx_base_env_cfg.py 中添加（5 分钟）
@configclass
class EventsCfg:
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8, 12),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
            }
        }
    )

# 第 2 步：在 events.py 中实现（20 分钟）
def push_by_setting_velocity(env, env_ids, asset_cfg, velocity_range):
    # ... 见前面的代码 ...
    pass

# 第 3 步：启动训练（1 小时）
python scripts/train.py --task=PointFootLocomotion --headless

# 第 4 步：测试（30 分钟）
python scripts/play.py --task=PointFootLocomotion \
    --checkpoint=logs/.../model.pt

# 观察机器人是否能应对推力
```

---

## 总结：关键要点

### 核心概念 3 点

1. **Domain Randomization**（域随机化）
   - 训练时加入各种随机扰动
   - 让神经网络学会通用策略
   - 提高 Sim-to-Real 的成功率

2. **推力冲量**（Impulse）
   - 力 × 时间的乘积
   - 评价标准：能承受的最大冲量
   - 越大说明机器人越"抗揍"

3. **恢复速度**（Recovery Speed）
   - 从扰动到稳定的时间
   - 评价标准：越快越好
   - 需要明确的奖励引导

### 实现步骤 3 步

1. **添加外力干扰事件**
   - 在 EventsCfg 中配置
   - 实现 push 函数

2. **调整奖励和观测**
   - 添加稳定性奖励
   - 添加加速度观测

3. **训练和测试**
   - 逐步增加难度
   - 测试最大冲量
   - 记录恢复速度

### 与其他任务的关系

```
任务 2.2：基础 → 学会走路
任务 2.3：鲁棒性 → 学会"抗揍"
任务 2.4：适应性 → 学会应对复杂环境

建议顺序：2.2 → 2.3 → 2.4
最小路径：2.2 → 简化版 2.3 → 2.4
```

现在你已经准备好让机器人变得更加鲁棒了！🛡️

