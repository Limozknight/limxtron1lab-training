# 现有任务代码配置审计 / Current Task Configuration Audit

**文件来源**: `limx_pointfoot_env_cfg.py`
**审计时间**: 2026-01-11

该文档列出了针对三个核心任务的**当前代码快照**中的关键策略和参数。请用此对比你的任务书要求，确认是否存在偏差。

---

## 2.2 平地速度跟随 (Flat Ground Velocity Tracking)

**对应配置类**: `PFBaseEnvCfg` (及其衍生类 `PFBlindFlatEnvCfg`)
**核心策略**: 尽可能减少干扰，纯粹考核速度追踪精度。

| 参数项 | 当前代码设定 | 说明 |
| :--- | :--- | :--- |
| **线速度追踪 (Lin Vel Reward)** | `rew_lin_vel_xy_precise.weight = 2.0` | (Task 2.4 V2中提升到了 5.5, Stair中降回 3.0) |
| **角速度追踪 (Ang Vel Reward)** | `rew_ang_vel_z_precise.weight = 1.5` | (Task 2.4 V2中提升到了 3.2, Stair中提升到 5.0) |
| **转动惩罚 (Ang Vel Penalty)** | `pen_ang_vel_xy.weight = -0.05` | 抑制 Roll/Pitch 震荡，提高姿态得分 |
| **地形 (Terrain)** | `MeshPlane` (Flat) | 纯平地 |
| **干扰 (Push)** | **OFF** | 纯速度考核通常不加推力 |

**代码快照 (RewardsCfg)**:
```python
    # [Fix] 权重从 8.0 降至 2.0，防止梯度爆炸
    rew_lin_vel_xy_precise = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.08)}
    )

    rew_ang_vel_z_precise = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.08)}
    )
```

---

## 2.3 抗干扰鲁棒性测试 (Disturbance Rejection)

**对应配置类**: `PFDisturbanceRejectionEnvCfg`
**核心策略**: 给予极高的生存与稳定奖励，使用高频大幅度推力进行对抗训练。

| 参数项 | 当前代码设定 | 说明 |
| :--- | :--- | :--- |
| **推力间隔 (Push Interval)** | `2.0 - 4.0 s` | 高频攻击 |
| **推力大小 (Force Range)** | `x,y: ±150.0 N` | **非常大** (基础只有 ±50 N) |
| **力矩干扰 (Torque Range)** | `x,y: ±25.0 Nm` | 增加旋转干扰 |
| **基座高度惩罚 (Height Pen)** | `weight = -15.0` | 极其严厉，防止被推倒/趴下 |
| **姿态稳定奖励 (Stability)** | `weight = 15.0` | 鼓励被推后迅速改平 |
| **速度修正力度** | `weight = 10.0` | 鼓励被推歪后迅速回到指令速度 |

**代码快照**:
```python
@configclass
class PFDisturbanceRejectionEnvCfg(PFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 1. 增强推力扰动事件
        self.events.push_robot = EventTerm(
            func=mdp.apply_external_force_torque_stochastic,
            mode="interval", 
            interval_range_s=(2.0, 4.0),
            params={
                "force_range": {"x": (-150.0, 150.0), "y": (-150.0, 150.0), ...},
                "probability": 1.0,
            },
        )
        # 2. 调整奖励权重
        self.rewards.pen_base_height.weight = -15.0
        self.rewards.rew_base_stability.weight = 15.0
        self.rewards.rew_lin_vel_xy_precise.weight = 10.0
```

---

## 2.4 复杂地形适应 (Terrain Traversal) - [你现在跑的]

**对应配置类**: `PFStairTrainingEnvCfg`
**核心策略**: 牺牲速度，换取高扭矩输出和绝对的直行能力（抗转圈）。

| 参数项 | 当前代码设定 | 说明 |
| :--- | :--- | :--- |
| **环境数量 (Num Envs)** | `512` | 降低以换取物理计算速度 |
| **地形 (Terrain)** | `STAIRS_TERRAINS_CFG` | 纯楼梯 (80%) + 斜坡 (20%) |
| **角速度追踪 (Ang Vel)** | `weight = 5.0` | **极高**，专治原地转圈 |
| **关节扭矩惩罚 (Torque Pen)** | `weight = -0.00005` | **极低**，允许电机爆发高扭矩爬楼 |
| **Z轴速度惩罚 (Z Vel Pen)** | `weight = -0.5` | 较低，允许抬腿动作 |
| **雷达精度 (Ray Resolution)** | `0.1` | 降低精度以减少计算量 |

**代码快照**:
```python
@configclass
class PFStairTrainingEnvCfg(PFTerrainTraversalEnvCfgV2):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 512
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        
        # 3. 奖励重点调整 / Reward Tuning
        # Allow more torque for climbing
        self.rewards.pen_joint_torque.weight = -0.00005 
        # Allow vertical movement (lifting legs)
        self.rewards.pen_lin_vel_z.weight = -0.5 

        # [Correction] 防止转圈：提高角速度追踪权重，强迫走直线
        self.rewards.rew_ang_vel_z_precise.weight = 5.0 
```
