# Task 2.4 优化版本 V2 - 降低扭矩与增强姿态稳定性

> 基于 V1 训练结果的分析，针对性优化关节扭矩和躯干稳定性

## 📊 V1 训练结果分析回顾

### ✅ V1 的优点
- **收敛良好**: mean_reward ~170，episode_length ~1000
- **速度跟踪精准**: 线速度和角速度误差小
- **平衡优秀**: keep_balance 达到 1.0

### ⚠️ V1 发现的问题

#### 1. 关节扭矩持续增加
```
pen_joint_torque: 从 -0.05 → -0.20（变得更负）
```

**原因分析**：
- 复杂地形要求机器人**快速响应**高度变化
- 策略学会了"用力过猛"来保持速度跟踪
- 在波浪和格子地形，关节需要**强行抬腿**

**影响**：
- ✅ 短期：任务完成度高
- ❌ 长期：电机过热，能耗过大，实际部署时续航短

**症状**：
- 在 TensorBoard 中，`pen_joint_torque` 曲线持续下探
- Play 时观察到机器人动作"生硬"、"急促"

#### 2. 俯仰/滚转角速度惩罚波动
```
pen_ang_vel_xy: 从 -0.02 → -0.06~-0.08（中后期波动）
```

**原因分析**：
- 地形起伏导致躯干晃动增加
- 策略优先保证速度，牺牲了姿态平滑度
- 在粗糙地形，足部着陆点不稳，引发躯干抖动

**影响**：
- ✅ 短期：未严重摔倒
- ❌ 长期：
  - 视觉传感器（如相机）画面抖动，影响 SLAM/避障
  - 机械结构疲劳，关节寿命缩短
  - 乘客舒适度差（如果是载人机器人）

**症状**：
- `pen_ang_vel_xy` 在训练后期未收敛到低位
- Play 时观察到躯干"点头"或"左右摇晃"

---

## 🎯 V2 优化目标

### 核心指标
| 指标 | V1 实际值 | V2 目标值 | 改进幅度 |
|------|-----------|-----------|----------|
| `pen_joint_torque` | -0.20 | **-0.10** | 减半 ✅ |
| `pen_ang_vel_xy` | -0.08 | **-0.04** | 减半 ✅ |
| `mean_reward` | 170 | **>165** | 允许略降 |
| `rew_lin_vel_xy_precise` | 1.5 | **>1.3** | 允许略降 |

**权衡策略**：
- **牺牲 5% 速度跟踪精度**，换取 **50% 扭矩降低** + **50% 姿态稳定性提升**
- 实际部署时，鲁棒性和能耗比绝对速度更重要

---

## 🔧 V2 修改方案

### 修改 1: 增加关节扭矩惩罚权重

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.rewards.pen_joint_torque.weight = -0.01

# V2（新）
self.rewards.pen_joint_torque.weight = -0.025  # 增加 2.5 倍
```

**原理**：
- 惩罚权重增加 → 策略更倾向于**轻柔动作**
- 强制策略在"完成任务"与"省力"之间找到新平衡点

**预期效果**：
- 关节峰值扭矩降低 30-50%
- 动作变得更"流畅"、"温和"
- 训练初期可能略慢（需要更多探索）

---

### 修改 2: 降低动作尺度（Action Scale）

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.actions.joint_pos.scale = 0.25  # 神经网络输出乘以 0.25 后作为关节位置偏移

# V2（新）
self.actions.joint_pos.scale = 0.20  # 减小 20%
```

**原理**：
- 动作尺度 = 神经网络输出到关节控制的"放大倍数"
- 降低尺度 → 单步动作幅度变小 → 关节加速度降低 → 扭矩降低

**预期效果**：
- 关节运动更保守、更平滑
- 减少"猛抬腿"、"猛踹地"等高扭矩动作
- 步幅可能略小，但稳定性提升

**风险**：
- 如果降得太多（如 0.15），可能导致机器人"迈不开步"

---

### 修改 3: 增强躯干姿态稳定奖励

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.rewards.rew_base_stability.weight = 1.0

# V2（新）
self.rewards.rew_base_stability.weight = 2.0  # 翻倍
```

**原理**：
- `rew_base_stability` 奖励躯干保持水平姿态
- 增加权重 → 策略更重视"稳定优先"

**预期效果**：
- 躯干俯仰/滚转角速度降低
- "点头"和"摇晃"减少
- 视觉传感器画面更稳定

---

### 修改 4: 增加俯仰/滚转角速度惩罚

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.rewards.pen_ang_vel_xy.weight = -0.05

# V2（新）
self.rewards.pen_ang_vel_xy.weight = -0.10  # 翻倍
```

**原理**：
- 直接惩罚 XY 平面的角速度（俯仰 pitch、滚转 roll）
- 权重翻倍 → 策略被迫减少躯干晃动

**预期效果**：
- 躯干运动更"稳重"
- 配合姿态奖励，形成"双重约束"

---

### 修改 5: 增强动作平滑惩罚

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.rewards.pen_action_smoothness.weight = -0.08

# V2（新）
self.rewards.pen_action_smoothness.weight = -0.12  # 增加 50%
```

**原理**：
- `pen_action_smoothness` = 惩罚相邻时间步动作的差异
- 增加惩罚 → 策略倾向于"缓慢变化动作"，而非"突变"

**预期效果**：
- 动作变化率降低
- 关节加速度降低 → 扭矩降低
- 整体步态更"流畅"

---

### 修改 6: 微调速度跟踪权重（可选）

**位置**: `limx_pointfoot_env_cfg.py` → `PFTerrainTraversalEnvCfg`

```python
# V1（旧）
self.rewards.rew_lin_vel_xy_precise.weight = 6.0
self.rewards.rew_ang_vel_z_precise.weight = 3.5

# V2（新）- 可选，如果上述修改导致速度过慢
self.rewards.rew_lin_vel_xy_precise.weight = 5.5  # 略降 8%
self.rewards.rew_ang_vel_z_precise.weight = 3.2   # 略降 8%
```

**原理**：
- 适当降低速度跟踪的"紧迫性"
- 给策略更多"缓冲"去优化扭矩和姿态

**预期效果**：
- 速度跟踪误差略增（可接受范围）
- 但整体平滑度和能耗大幅改善

**何时使用**：
- 如果修改 1-5 后，速度跟踪仍过于激进
- 观察到 `pen_joint_torque` 未降到目标值

---

## 📝 完整修改代码（V2）

### 文件: `limx_pointfoot_env_cfg.py`

在 `PFTerrainTraversalEnvCfg` 的 `__post_init__` 中，添加/修改：

```python
@configclass
class PFTerrainTraversalEnvCfgV2(PFBaseEnvCfg):
    """任务2.4 V2优化版：降低扭矩与增强姿态稳定性"""

    def __post_init__(self):
        super().__post_init__()

        # ========== 地形与传感器配置（与 V1 相同）==========
        self.scene.env_spacing = 3.0
        self.scene.num_envs = 2048
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
        self.curriculum.terrain_levels = None

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(0.0, 10.0),
        )

        # ========== V2 修改 1: 降低动作尺度 ==========
        self.actions.joint_pos.scale = 0.20  # V1: 0.25

        # ========== V2 修改 2-6: 奖励权重调整 ==========
        # 速度跟踪（略降，给扭矩/姿态优化让路）
        self.rewards.rew_lin_vel_xy_precise.weight = 5.5   # V1: 6.0
        self.rewards.rew_ang_vel_z_precise.weight = 3.2    # V1: 3.5

        # 姿态稳定（大幅增加）
        self.rewards.rew_base_stability.weight = 2.0       # V1: 1.0

        # 高度惩罚（保持）
        self.rewards.pen_base_height.func = mdp.base_height_rough_l2
        self.rewards.pen_base_height.weight = -8.0
        self.rewards.pen_base_height.params = {
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        }

        # 姿态约束（保持）
        self.rewards.pen_flat_orientation.weight = -3.0
        self.rewards.pen_feet_regulation.weight = -0.2
        self.rewards.foot_landing_vel.weight = -1.0
        self.rewards.pen_undesired_contacts.weight = -1.0

        # **V2 关键修改：扭矩与动作平滑**
        self.rewards.pen_joint_torque.weight = -0.025      # V1: -0.01（增加 2.5 倍）
        self.rewards.pen_action_smoothness.weight = -0.12  # V1: -0.08（增加 50%）
        
        # **V2 关键修改：俯仰/滚转角速度**
        self.rewards.pen_ang_vel_xy.weight = -0.10         # V1: -0.05（翻倍）

        # 禁用外力扰动
        self.events.push_robot = None


@configclass
class PFTerrainTraversalEnvCfgV2_PLAY(PFTerrainTraversalEnvCfgV2):
    """V2 测试配置"""

    def __post_init__(self):
        super().__post_init__()

        # 更少环境用于评估
        self.scene.num_envs = 64

        # 禁用观测腐蚀与随机化
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
```

---

## 🎮 Gym 注册

### 文件: `robots/__init__.py`

```python
# 注册 V2 训练环境
gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfgV2,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

# 注册 V2 评估环境
gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-V2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfgV2_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)
```

---

## 🚀 训练命令

```bash
# 启动 V2 训练
python scripts/rsl_rl/train.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-V2-v0 \
  --headless \
  --num_envs=2048

# 评估 V2 模型
python scripts/rsl_rl/play.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-V2-Play-v0 \
  --checkpoint_path=logs/.../model_3000.pt \
  --video
```

---

## 📊 预期改进对比

### 训练曲线预期变化

| 指标 | V1 曲线 | V2 预期曲线 | 改进说明 |
|------|---------|-------------|----------|
| `pen_joint_torque` | -0.05 → -0.20 ⬇ | **-0.05 → -0.10** ⬇ | 降幅减半，能耗降低 |
| `pen_ang_vel_xy` | -0.02 → -0.08 ⬇ | **-0.02 → -0.04** ⬇ | 姿态抖动减半 |
| `rew_base_stability` | 1.0 → 5.0 ⬆ | **2.0 → 7.0** ⬆ | 稳定性奖励更高 |
| `rew_lin_vel_xy_precise` | 0 → 1.5 ⬆ | **0 → 1.3** ⬆ | 略降可接受 |
| `mean_reward` | 0 → 170 ⬆ | **0 → 165** ⬆ | 略降可接受 |
| `mean_episode_length` | 200 → 1000 ⬆ | **200 → 1000** ⬆ | 保持高稳定性 |

### 实际表现预期

| 维度 | V1 实际表现 | V2 预期表现 |
|------|-------------|-------------|
| **动作风格** | 生硬、急促 | **流畅、温和** ✅ |
| **躯干晃动** | 明显"点头" | **平滑稳定** ✅ |
| **关节声音**（真机） | 电机嗡嗡作响 | **安静** ✅ |
| **电池续航**（真机） | 15 分钟 | **20+ 分钟** ✅ |
| **速度响应** | 快速但抖动 | **略慢但平稳** |

---

## ⚠️ 风险与应对

### 风险 1: 扭矩惩罚过强导致"不敢动"

**症状**：
- 训练初期 `mean_reward` 长期低迷
- 机器人原地踏步或只能缓慢前进

**应对**：
- 将 `pen_joint_torque.weight` 从 -0.025 调回 -0.018（折中）
- 或增加 `rew_lin_vel_xy_precise` 权重到 6.5（加强速度激励）

### 风险 2: 动作尺度过小导致步幅不足

**症状**：
- 机器人"小碎步"，无法跨越格子地形
- `pen_undesired_contacts` 增加（肚子蹭地）

**应对**：
- 将 `action_scale` 从 0.20 调回 0.22（略增）
- 或增加 `foot_clearance` 奖励（鼓励抬腿更高）

### 风险 3: 姿态约束过强导致速度下降

**症状**：
- `rew_lin_vel_xy_precise` 持续低于 1.0
- 机器人为了保持水平而不敢加速

**应对**：
- 将 `rew_base_stability` 从 2.0 调回 1.5（折中）
- 或将 `pen_ang_vel_xy` 从 -0.10 调回 -0.07（略松）

---

## 🎯 训练监控要点

### 前 500 轮
**关注**: `pen_joint_torque` 是否从 -0.05 开始缓慢下降
- ✅ 正常：-0.05 → -0.08（缓降）
- ❌ 异常：-0.05 → -0.15+（过快，说明扭矩惩罚不足）

### 500-1500 轮
**关注**: `pen_ang_vel_xy` 是否收敛到 -0.04 附近
- ✅ 正常：-0.02 → -0.04（平稳）
- ❌ 异常：-0.02 → -0.08+（仍波动，说明姿态约束不足）

### 1500-3000 轮
**关注**: `mean_reward` 是否稳定在 160-170
- ✅ 正常：收敛到 165 左右
- ❌ 异常：<150（说明权衡失败，扭矩/姿态约束过强）

---

## 📈 A/B 测试方案（可选）

如果资源充足，可以同时训练 V1 和 V2：

```bash
# 终端 1：V1 训练（基线）
python scripts/rsl_rl/train.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-v0 \
  --headless

# 终端 2：V2 训练（优化版）
python scripts/rsl_rl/train.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-V2-v0 \
  --headless
```

**每 500 轮对比**：
- TensorBoard 并排查看曲线
- Play 视频并排对比（动作平滑度、躯干稳定性）
- 提取最终性能表（扭矩均值、角速度均值）

---

## ✅ 验收标准

### 必须满足（硬性指标）
- [ ] `pen_joint_torque` 最终值 < -0.12（比 V1 改善 40%+）
- [ ] `pen_ang_vel_xy` 最终值 < -0.05（比 V1 改善 30%+）
- [ ] `mean_episode_length` > 900（保持高稳定性）

### 期望满足（软性指标）
- [ ] `mean_reward` > 160（允许略降）
- [ ] Play 视频中躯干晃动明显减少
- [ ] 动作看起来"更自然"、"更省力"

### 加分项
- [ ] `rew_lin_vel_xy_precise` 仍 > 1.3（速度损失 <15%）
- [ ] 训练收敛速度未明显变慢（仍在 2500 轮内）

---

## 🔄 迭代流程

1. **实施 V2 修改** → 保存为新的配置类
2. **启动训练** → 监控前 500 轮曲线
3. **对比 V1** → 判断改进方向是否正确
4. **微调参数** → 如有风险，按"风险应对"调整
5. **完整训练** → 跑满 3000 轮
6. **Play 评估** → 录制视频，测量关键指标
7. **性能报告** → 对比 V1/V2 表格

---

## 📦 产出物

### 代码
- ✅ `PFTerrainTraversalEnvCfgV2` 类
- ✅ `PFTerrainTraversalEnvCfgV2_PLAY` 类
- ✅ Gym 注册（V2 ID）

### 模型
- ✅ `logs/.../Isaac-Limx-PF-Terrain-Traversal-V2-v0/checkpoints/model_3000.pt`

### 文档
- ✅ 本文档（`Task2.4_Optimization_V2.md`）
- ✅ TensorBoard 曲线截图（V1 vs V2 对比）
- ✅ Play 视频（V1 vs V2 并排）

### 数据表
```markdown
| 指标 | V1 | V2 | 改进 |
|------|----|----|------|
| pen_joint_torque | -0.20 | -0.10 | 50% ⬆ |
| pen_ang_vel_xy | -0.08 | -0.04 | 50% ⬆ |
| mean_reward | 170 | 165 | -3% ⬇ |
```

---

**版本**: V2  
**日期**: 2026-01-06  
**作者**: AI Assistant  
**状态**: 🚧 待实施与测试  
**依赖**: V1 已完成
