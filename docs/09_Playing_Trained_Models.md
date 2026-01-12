# 运行训练好的模型详细指南（Playing Trained Models）

> 用 Play 来验证训练的模型效果。包括运行命令、环境配置架构、如何针对新任务（如 Task 2.4）配置。

## 快速命令

```bash
# Task 2.2 - 平地速度跟随
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 \
  --checkpoint_path=logs/rsl_rl/ppo_pf_blind_flat/Isaac-Limx-PF-Blind-Flat-v0/checkpoints/model_5000.pt

# Task 2.4 - 复杂地形
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
  --checkpoint_path=logs/rsl_rl/ppo_pf_terrain_traversal/Isaac-Limx-PF-Terrain-Traversal-v0/checkpoints/model_5000.pt

# 加可选参数
python scripts/rsl_rl/play.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
  --checkpoint_path=path/to/checkpoint.pt \
  --num_envs=64 \
  --headless \
  --video \
  --video_length=500
```

## 环境配置架构（你需要理解这个）

### 1️⃣ 核心流程

```
play.py 的执行流程
    │
    ├─→ parse_env_cfg(task_name="Isaac-Limx-PF-Terrain-Traversal-Play-v0", ...)
    │    │
    │    ├─→ gym.make("Isaac-Limx-PF-Terrain-Traversal-Play-v0", cfg=env_cfg)
    │    │
    │    └─→ 查找该 task 在 gym 中的注册信息
    │
    ├─→ robots/__init__.py 中的 gym.register()
    │    │
    │    ├─ id: "Isaac-Limx-PF-Terrain-Traversal-Play-v0"
    │    │
    │    └─ kwargs["env_cfg_entry_point"]: 指向配置类
    │        └─→ limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfg_PLAY
    │
    ├─→ 加载环境配置类
    │    │
    │    └─→ limx_pointfoot_env_cfg.py 中定义的 PFTerrainTraversalEnvCfg_PLAY
    │        ├─ scene 配置（USD 资产、地形、传感器）
    │        ├─ observations 配置
    │        ├─ actions 配置
    │        ├─ rewards 配置
    │        └─ termination conditions
    │
    └─→ 创建环境实例并加载检查点
         └─→ ppo_runner.load(checkpoint_path)
             └─→ 策略推理
```

### 2️⃣ 文件对应关系（关键点！）

| 任务 | Gym ID（train） | Gym ID（play） | 配置类（train） | 配置类（play） |
|------|-----------------|-----------------|-----------------|-----------------|
| 2.2 | Isaac-Limx-PF-Blind-Flat-v0 | Isaac-Limx-PF-Blind-Flat-Play-v0 | `PFBlindFlatEnvCfg` | `PFBlindFlatEnvCfg_PLAY` |
| 2.4 | Isaac-Limx-PF-Terrain-Traversal-v0 | Isaac-Limx-PF-Terrain-Traversal-Play-v0 | `PFTerrainTraversalEnvCfg` | `PFTerrainTraversalEnvCfg_PLAY` |

**所有配置都定义在**：
- 📁 [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_pointfoot_env_cfg.py](../exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_pointfoot_env_cfg.py)

**所有 Gym 注册都在**：
- 📁 [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py](../exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py)

### 3️⃣ 为什么分 train 和 play 两套配置？

```python
# PFBlindFlatEnvCfg（训练用）
class PFBlindFlatEnvCfg(ManagerBasedRLEnvCfg):
    num_envs = 2048          # ← 大并行数，快速采集
    env_spacing = 5.0
    domain_randomization = True  # ← 随机化，增加多样性
    # ...

# PFBlindFlatEnvCfg_PLAY（评估用）
class PFBlindFlatEnvCfg_PLAY(PFBlindFlatEnvCfg):
    num_envs = 64            # ← 小并行数，便于观察
    env_spacing = 25.0
    domain_randomization = False  # ← 关闭随机化，得到"干净"的表现
    # ...
```

**主要差异**：
| 方面 | 训练版本 | 评估版本 |
|------|---------|---------|
| 并行环境数 | 2048 | 64 |
| 域随机化 | ✅ 开启 | ❌ 关闭 |
| 推力干扰 | ✅ 开启 | ❌ 关闭 |
| 地形随机性 | ✅ 有 | ❌ 固定 |
| 目标 | 学习鲁棒策略 | 观看成果、录制视频 |

## 针对 Task 2.4 的完整步骤

### Step 1: 确认配置已添加

打开 [limx_pointfoot_env_cfg.py](../exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_pointfoot_env_cfg.py)，检查是否有：

```python
class PFTerrainTraversalEnvCfg(ManagerBasedRLEnvCfg):
    """任务 2.4 训练配置"""
    # ...

class PFTerrainTraversalEnvCfg_PLAY(PFTerrainTraversalEnvCfg):
    """任务 2.4 评估配置"""
    # ...
```

✅ 已在之前配置好。

### Step 2: 确认 Gym 注册

打开 [robots/__init__.py](../exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py)，检查是否有：

```python
gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)
```

✅ 已注册。

### Step 3: 获取检查点路径

训练完成后，模型保存在：
```
logs/
└── rsl_rl/
    └── ppo_pf_terrain_traversal/
        └── Isaac-Limx-PF-Terrain-Traversal-v0/
            └── checkpoints/
                ├── model_1000.pt
                ├── model_2000.pt
                └── model_5000.pt  ← 通常选这个（最后一个）
```

或更早的检查点：
```
logs/
└── rsl_rl/
    └── ppo_pf_terrain_traversal/
        └── Isaac-Limx-PF-Terrain-Traversal-v0/
            └── checkpoints/
                └── model_latest.pt
```

### Step 4: 运行 Play

```bash
python scripts/rsl_rl/play.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
  --checkpoint_path=logs/rsl_rl/ppo_pf_terrain_traversal/Isaac-Limx-PF-Terrain-Traversal-v0/checkpoints/model_5000.pt \
  --num_envs=64 \
  --video \
  --video_length=500
```

检查输出：
```
[INFO]: Loading model checkpoint from: logs/...
[INFO]: Loaded model checkpoint from: logs/...
[INFO] Opening the visualization window...
```

✅ 运行成功。

## 一般参数说明

| 参数 | 含义 | 示例 |
|------|------|------|
| `--task` | Gym 环境 ID | `Isaac-Limx-PF-Terrain-Traversal-Play-v0` |
| `--checkpoint_path` | 模型文件路径 | `logs/.../model_5000.pt` |
| `--num_envs` | 并行环境数 | 默认取自配置；可覆盖（`--num_envs=32`） |
| `--headless` | 无头模式 | 不显示窗口，速度快 |
| `--video` | 录制视频 | 保存到 `log_dir/videos/play/` |
| `--video_length` | 每段视频步数 | 默认 200；增大如 500 |
| `--seed` | 随机种子 | 重现结果 |

## 常见问题

### Q1: 我找不到检查点文件怎么办？

运行：
```bash
ls logs/rsl_rl/*/Isaac-Limx-PF-*/checkpoints/
```

如果空的，说明训练还没完成或没有保存。检查 train.py 的输出。

### Q2: 运行时说 "Task not found" 怎么办？

检查：
1. 拼写是否正确（区分大小写）。
2. 是否已导入 bipedal_locomotion（play.py 有 `import bipedal_locomotion`）。
3. 是否已安装：`pip install -e exts/bipedal_locomotion`。

### Q3: 运行时环境和训练时不一样怎么办？

确认使用的 Play 配置与 Train 配置**对应**：
- Train: `Isaac-Limx-PF-Terrain-Traversal-v0` → Train 配置类
- Play:  `Isaac-Limx-PF-Terrain-Traversal-Play-v0` → Play 配置类

Play 配置通常关闭随机化，但**物理场景、观测、动作空间必须完全相同**。

### Q4: 能否用 Train 配置来 Play？

不建议。Train 配置有大量随机化（地形、风、推力等），看不到"真实"的模型性能。用 Play 配置才能看到清晰的效果。

### Q5: 如何录制视频？

```bash
python scripts/rsl_rl/play.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
  --checkpoint_path=path/to/checkpoint.pt \
  --video \
  --video_length=500 \
  --headless
```

视频保存在：
```
logs/rsl_rl/.../Isaac-Limx-PF-Terrain-Traversal-v0/videos/play/
```

### Q6: 我想只在 1 个环境中运行看细节怎么办？

```bash
python scripts/rsl_rl/play.py \
  --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
  --checkpoint_path=... \
  --num_envs=1
```

## 工作流对比

### 完整训练 → 评估流程

```
1. 训练阶段
   python scripts/rsl_rl/train.py \
     --task=Isaac-Limx-PF-Terrain-Traversal-v0 \
     --headless
   
   ✓ 输出 logs/.../checkpoints/model_5000.pt

2. 评估阶段
   python scripts/rsl_rl/play.py \
     --task=Isaac-Limx-PF-Terrain-Traversal-Play-v0 \
     --checkpoint_path=logs/.../model_5000.pt \
     --video

3. 分析结果
   • 查看视频（logs/.../videos/play/）
   • 检查输出日志中的性能数据（速度误差、地形通过率等）
```

## 一句话总结

**不需要额外配置——所有环境配置都已在 `limx_pointfoot_env_cfg.py` 中定义，所有 Gym 注册都在 `robots/__init__.py` 中。只需指定正确的 task ID 和检查点路径，run play.py 即可。**
