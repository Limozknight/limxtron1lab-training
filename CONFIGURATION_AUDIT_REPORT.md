# 配置审计报告 (Configuration Audit Report)
**日期**: 2026-01-10  
**目标**: 全面检查Task 2+3和Task 2+3+4的环境和训练配置  
**关键问题**: Task 2.2奖励权重错误导致模型学习"站立不动"策略

---

## 📋 检查范围

本审计检查了以下**3个关键文件**：

1. **limx_pointfoot_env_cfg.py** - 环境配置（关键文件）
2. **limx_base_env_cfg.py** - 基础奖励定义
3. **limx_rsl_rl_ppo_cfg.py** (在agents目录下) - 训练器配置
4. **robots/__init__.py** - 任务注册

---

## 🔴 **关键发现：Task 2+3 奖励权重CRITICAL BUG**

### 问题位置
**文件**: `limx_pointfoot_env_cfg.py`  
**类**: `PFTask2And3EnvCfg`  
**行数**: 646-653

### ❌ 当前（错误）配置
```python
class PFTask2And3EnvCfg(PFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 错误的权重设置（来自队友的代码）
        self.rewards.rew_lin_vel_xy_precise.weight = 3.0     # ❌ 错误
        self.rewards.rew_ang_vel_z_precise.weight = 2.0      # ❌ 错误
        self.rewards.pen_base_height.weight = -5.0           # ❌ 错误
```

### ✅ 正确配置（应该是）
```python
class PFTask2And3EnvCfg(PFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 正确的权重设置（来自成功的PFTerrainTraversalEnvCfgV2）
        self.rewards.rew_lin_vel_xy_precise.weight = 5.5     # ✅ 正确
        self.rewards.rew_ang_vel_z_precise.weight = 3.2      # ✅ 正确
        self.rewards.pen_base_height.weight = -1.0           # ✅ 正确
```

### 🔬 权重差异分析

| 参数 | 当前值 | 正确值 | 差异 | 影响 |
|------|--------|--------|------|------|
| `rew_lin_vel_xy_precise.weight` | 3.0 | 5.5 | -45% | 线速度追踪奖励不足，机器人倾向于站立 |
| `rew_ang_vel_z_precise.weight` | 2.0 | 3.2 | -37.5% | 角速度追踪不足 |
| `pen_base_height.weight` | -5.0 | -1.0 | **+400%** (更严厉) | 高度惩罚过度，机器人为避免惩罚宁愿倒地也不追踪速度 |

### 📊 症状表现
- ✅ 训练奖励: 61 points (因为base_height惩罚不活跃在平地)
- ❌ Play阶段: 26-27 points (奖励权重生效，导致保守行为)
- 🤖 实际行为: 机器人站立不动或低速晃动，不追踪命令速度

---

## 📝 完整检查清单

### 1️⃣ **limx_pointfoot_env_cfg.py** - 环境配置 (CRITICAL)

#### 检查项A: Task 2+3 奖励配置
✅ **位置**: 行 622-654  
⚠️ **类**: `PFTask2And3EnvCfg`

**检查的参数**:
- ❌ `rew_lin_vel_xy_precise.weight = 3.0` → 应该是 **5.5**
- ❌ `rew_ang_vel_z_precise.weight = 2.0` → 应该是 **3.2**
- ❌ `pen_base_height.weight = -5.0` → 应该是 **-1.0**
- ✅ `push_robot` 事件: interval (3.0-5.0s), force 80N ✓ 正确
- ✅ `events.push_robot` 已启用 ✓ 正确

**Action**: 需要修改这三个权重值

---

#### 检查项B: Task 2+3 Play环境配置
✅ **位置**: 行 657-665  
✅ **类**: `PFTask2And3EnvCfg_PLAY`

**检查的参数**:
- ✅ `enable_corruption = False` ✓ 正确（不添加噪声）
- ✅ `num_envs = 32` ✓ 正确（Play用）
- ✅ 继承自 `PFTask2And3EnvCfg` ✓ 正确
- ✅ Push interval **保持与训练一致** (3.0-5.0s) ✓ 正确

**分析**: Play环境配置是**正确的**。问题不在Play配置，而在训练配置的权重。

---

#### 检查项C: Task 2+3+4 统一环境配置
✅ **位置**: 行 667-720  
✅ **类**: `PFUnifiedEnvCfg`

**检查的参数**:
- ✅ 继承自 `PFTerrainTraversalEnvCfgV2` ✓ 正确
- ✅ `rew_lin_vel_xy_precise.weight = 5.0` ✓ 合理（地形上略低于5.5）
- ✅ `rew_ang_vel_z_precise.weight = 3.2` ✓ 继承自V2，正确
- ✅ `pen_base_height.weight = -1.0` ✓ 正确（V2已修复）
- ✅ `push_robot` 事件已启用 (3-6s, 80N) ✓ 正确
- ✅ 课程学习已启用 ✓ 正确

**分析**: Task 2+3+4配置**基本正确**。

---

### 2️⃣ **limx_base_env_cfg.py** - 基础奖励定义

#### 检查项D: 基础奖励权重定义
✅ **位置**: 行 474-550  
✅ **类**: `RewardsCfg`

**检查的默认参数** (这些是父类的defaults):
```python
rew_lin_vel_xy_precise = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=2.0,                              # 基础值
    params={"command_name": "base_velocity", "std": 0.5}
)

rew_ang_vel_z_precise = RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=1.5,                              # 基础值
    params={"command_name": "base_velocity", "std": 0.5}
)

pen_base_height = RewTerm(
    func=mdp.base_com_height,
    params={"target_height": 0.78},
    weight=-2.0                              # 基础值
)
```

**注意**: 这些是`PFBlindFlatEnvCfg`的基础值。子类`PFTask2And3EnvCfg`会**覆盖**这些值。

**分析**: 基础定义是合理的，但被子类覆盖为错误值。

---

#### 检查项E: 其他奖励项
✅ **位置**: 行 474-550  
✅ **检查的其他项目**:
- ✅ `keep_balance.weight = 1.0` ✓ 正确（存活奖励）
- ✅ `rew_base_stability.weight = 1.0` ✓ 正确
- ✅ `pen_lin_vel_z.weight = -0.5` ✓ 正确（禁止Z方向运动）
- ✅ `pen_ang_vel_xy.weight = -0.05` ✓ 正确（禁止X/Y转动）
- ✅ `pen_joint_torque.weight = -0.00008` ✓ 正确（微小惩罚，防止爆炸）
- ✅ `pen_action_smoothness.weight = -0.04` ✓ 正确
- ✅ `pen_flat_orientation.weight = -2.0` ✓ 正确
- ✅ `foot_landing_vel.weight = -0.5` ✓ 正确（软着陆）

**分析**: 其他奖励项都配置合理。

---

### 3️⃣ **limx_rsl_rl_ppo_cfg.py** - 训练器配置

#### 检查项F: Task 2+3 PPO运行器配置
✅ **位置**: agents/limx_rsl_rl_ppo_cfg.py, 行 89-130  
✅ **类**: `PF_Task2And3PPORunnerCfg`

**检查的参数**:
```python
experiment_name = "pf_task2_3_flat"
num_steps_per_env = 24
max_iterations = 3000
save_interval = 200
```

- ✅ `experiment_name = "pf_task2_3_flat"` ✓ 正确（清晰的任务标识）
- ✅ `num_steps_per_env = 24` ✓ 正确
- ✅ `max_iterations = 3000` ✓ 正确（Task 2+3训练迭代次数）
- ✅ 使用 `RslRlPpoAlgorithmMlpCfg` ✓ 正确（支持MLP编码器）
- ✅ `obs_history_len = 10` ✓ 正确（与观测配置对齐）

**分析**: PPO配置正确。

---

#### 检查项G: Task 2+3+4 PPO运行器配置
✅ **位置**: agents/limx_rsl_rl_ppo_cfg.py, 行 130-170  
✅ **类**: `PF_Task2And3And4PPORunnerCfg`

**检查的参数**:
```python
experiment_name = "pf_task2_3_4_unified"
num_steps_per_env = 24
max_iterations = 4000
save_interval = 200
```

- ✅ `experiment_name = "pf_task2_3_4_unified"` ✓ 正确（清晰标识）
- ✅ 与Task 2+3共用相同PPO配置 ✓ 正确
- ✅ `max_iterations = 4000` ✓ 合理（地形任务需要更多迭代）

**分析**: PPO配置正确。

---

### 4️⃣ **robots/__init__.py** - 任务注册

#### 检查项H: Task 2+3 任务注册
✅ **位置**: robots/__init__.py, 行 75-88

**检查的注册**:
```python
gym.register(
    id="Isaac-Limx-PF-Task2-3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTask2And3EnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_task2_3_runner_cfg,  # ✓ 使用正确的runner
    },
)

gym.register(
    id="Isaac-Limx-PF-Task2-3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTask2And3EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_task2_3_runner_cfg,  # ✓ Play使用相同runner配置
    },
)
```

- ✅ 训练环境映射: `PFTask2And3EnvCfg` → `limx_pf_task2_3_runner_cfg` ✓ 正确
- ✅ Play环境映射: `PFTask2And3EnvCfg_PLAY` → `limx_pf_task2_3_runner_cfg` ✓ 正确
- ✅ 使用 `experiment_name = "pf_task2_3_flat"` ✓ 日志分离正确

**分析**: 任务注册正确。

---

#### 检查项I: Task 2+3+4 任务注册
✅ **位置**: robots/__init__.py, 行 89-110

**检查的注册**:
```python
gym.register(
    id="Isaac-Limx-PF-Unified-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFUnifiedEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_task2_3_4_runner_cfg,  # ✓ 使用不同runner
    },
)
```

- ✅ 任务环境映射正确 ✓
- ✅ 使用 `experiment_name = "pf_task2_3_4_unified"` ✓ 日志分离正确

**分析**: 任务注册正确。

---

## 🎬 Play环境配置位置

### 问题回答: "运行play的环境配置是在哪里？"

**答案**: Play环境配置在**两个地方**定义:

#### 1️⃣ **环境配置** (更重要)
**文件**: `limx_pointfoot_env_cfg.py`  
**类**: `PFTask2And3EnvCfg_PLAY`  
**行数**: 657-665

```python
@configclass
class PFTask2And3EnvCfg_PLAY(PFTask2And3EnvCfg):
    """Play version of Task 2+3 - same config as training, just disable observation corruption."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 禁用观测噪声，其他配置保持与训练一致
        self.observations.policy.enable_corruption = False
```

**关键点**:
- 继承自 `PFTask2And3EnvCfg` (训练配置)
- 只修改: 环境数量(32), 禁用噪声
- **不修改**: 奖励权重（保持与训练一致，因为Play不需要奖励，但配置必须相同）

#### 2️⃣ **运行器配置** (提供算法参数)
**文件**: `robots/__init__.py`  
**行数**: 81-88

```python
gym.register(
    id="Isaac-Limx-PF-Task2-3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTask2And3EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_task2_3_runner_cfg,
    },
)
```

**Play命令示例**:
```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-PF-Task2-3-Play-v0 \
  --num_envs 32 \
  --load_run 2026-01-10_13-53-08_Task2-3_Baseline_v1 \
  --checkpoint model_3000.pt
```

---

## 📊 检查结果汇总表

| # | 检查项目 | 文件 | 类/函数 | 状态 | 备注 |
|----|---------|------|--------|------|------|
| A | Task 2+3 线速度权重 | limx_pointfoot_env_cfg.py | PFTask2And3EnvCfg | ❌ | 3.0应为5.5 |
| B | Task 2+3 角速度权重 | limx_pointfoot_env_cfg.py | PFTask2And3EnvCfg | ❌ | 2.0应为3.2 |
| C | Task 2+3 高度惩罚权重 | limx_pointfoot_env_cfg.py | PFTask2And3EnvCfg | ❌ | -5.0应为-1.0 |
| D | Task 2+3 推力配置 | limx_pointfoot_env_cfg.py | PFTask2And3EnvCfg | ✅ | 3-5s, 80N正确 |
| E | Task 2+3 Play配置 | limx_pointfoot_env_cfg.py | PFTask2And3EnvCfg_PLAY | ✅ | 正确 |
| F | Task 2+3+4配置 | limx_pointfoot_env_cfg.py | PFUnifiedEnvCfg | ✅ | 权重正确 |
| G | 基础奖励定义 | limx_base_env_cfg.py | RewardsCfg | ✅ | 基础值合理 |
| H | 其他奖励项 | limx_base_env_cfg.py | RewardsCfg | ✅ | 全部正确 |
| I | Task 2+3 PPO配置 | limx_rsl_rl_ppo_cfg.py | PF_Task2And3PPORunnerCfg | ✅ | experiment_name正确 |
| J | Task 2+3+4 PPO配置 | limx_rsl_rl_ppo_cfg.py | PF_Task2And3And4PPORunnerCfg | ✅ | experiment_name正确 |
| K | Task 2+3 任务注册 | robots/__init__.py | gym.register() | ✅ | 映射正确 |
| L | Task 2+3+4 任务注册 | robots/__init__.py | gym.register() | ✅ | 映射正确 |

---

## 🔧 需要修复的项目

### 立即行动 (URGENT)

**修改文件**: `limx_pointfoot_env_cfg.py`  
**修改类**: `PFTask2And3EnvCfg`  
**修改位置**: 行 646-653

```diff
  @configclass
  class PFTask2And3EnvCfg(PFBlindFlatEnvCfg):
      def __post_init__(self):
          super().__post_init__()
          
          # --- Task 2: 高精度速度追踪 ---
-         self.rewards.rew_lin_vel_xy_precise.weight = 3.0
+         self.rewards.rew_lin_vel_xy_precise.weight = 5.5
-         self.rewards.rew_ang_vel_z_precise.weight = 2.0
+         self.rewards.rew_ang_vel_z_precise.weight = 3.2
          
          # --- Task 3: 姿态恢复 ---
          self.rewards.rew_base_stability.weight = 2.0
          
          # 加大摔倒惩罚
-         self.rewards.pen_base_height.weight = -5.0
+         self.rewards.pen_base_height.weight = -1.0
```

---

## 📈 修改后的预期结果

**修改前 (当前错误状态)**:
- 训练奖励: ~61 points (虚高，因为惩罚权重在平地几乎无效)
- Play奖励: 26-27 points (保守行为，站立不动)
- 实际速度: 0-0.2 m/s

**修改后 (预期正确状态)**:
- 训练奖励: ~65-70 points (更高，因为奖励权重更高)
- Play奖励: 60-65 points (与训练一致)
- 实际速度: 0.8-1.2 m/s (主动追踪命令速度)

---

## ✅ 验证清单 (修改后)

- [ ] 修改 `PFTask2And3EnvCfg` 的三个权重
- [ ] 保存文件
- [ ] 运行训练命令验证
- [ ] 检查训练日志中的奖励权重是否正确
- [ ] 等待模型训练完成 (~1-2小时)
- [ ] 运行Play验证机器人是否主动追踪速度
- [ ] 记录新的Play奖励分数

---

## 📌 关键要点总结

1. **问题根源**: `PFTask2And3EnvCfg` 中的奖励权重被错误覆盖（来自队友代码）
2. **影响范围**: 仅限Task 2+3训练，Task 2+3+4不受影响
3. **修复方式**: 修改3个权重值到正确值
4. **验证方法**: 重新训练并检查Play行为
5. **时间成本**: 修复+重新训练需要 1-2 小时GPU时间
6. **其他配置**: 都是正确的，无需修改

---

**审计完成时间**: 2026-01-10 14:30  
**审计者**: GitHub Copilot  
**状态**: 准备实施修复 ✅
