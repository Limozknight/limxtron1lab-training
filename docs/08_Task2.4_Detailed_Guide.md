# 任务 2.4 详细指南：复杂地形适应（Terrain Traversal）

**目标受众**：对强化学习完全陌生，但要完成地形适应任务的新手

**预计学习时间**：4-6 小时理论 + 3-5 天实践

**难度等级**：⭐⭐⭐⭐（比任务 2.2 难）

---

## 第一部分：为什么任务 2.4 比 2.2 难？

### 问题对比

```
任务 2.2（平地速度跟随）：
  ✓ 环境简单：完全平坦
  ✓ 问题简单：只需学会走路
  ✓ 奖励简单：速度对不对就完事了
  ✓ 失败容易预测：摔倒 = 失败
  
  结果：容易训练，快速收敛

任务 2.4（复杂地形）：
  ✗ 环境复杂：斜坡、台阶、离散路面多重变化
  ✗ 问题复杂：需要根据地形改变步态
  ✗ 奖励复杂：既要速度快，又要稳定，还要适应地形
  ✗ 失败难预测：什么情况会摔倒不确定
  
  结果：难以训练，收敛缓慢
```

### 为什么会这样？

```
平地行走：
  机器人只需学会一种"标准步态"
  无论什么时候，同一个步态都有效
  神经网络：记住这个步态 → 完成任务

地形适应：
  不同地形需要不同的步态
  • 上坡：步幅小，更频繁
  • 下坡：小心谨慎，更稳定
  • 台阶：大步幅跨越
  • 离散路面：频繁调整平衡
  
  神经网络：必须学会"如何感知地形" + "如何根据地形调整"
  这多出了一层复杂性！
```

---

## 第二部分：你需要学习的核心概念（按优先级）

### 1️⃣ **必学**：什么是 TerrainGenerator？

```
TerrainGenerator = "地形生成器"
作用：自动创建各种不同的地形（不需要你手工建模）

传统方法：
  • 美工手工建模每一块地形
  • 时间长、工作量大、无法灵活改变

Isaac Lab 的方法：
  • 告诉 TerrainGenerator 参数（如"30 个斜坡"、"宽度 2m"）
  • TerrainGenerator 自动生成
  • 可以随时改参数重新生成
  • 效率极高，适合强化学习实验

工作流程：
  1. 定义地形类型参数（比如"高度差 0.3m"）
  2. TerrainGenerator 根据参数生成地形
  3. 机器人在这些地形上训练
  4. 强化学习修改神经网络权重
```

### 2️⃣ **必学**：Isaac Lab 中的地形类型

```
Isaac Lab 支持的基础地形：

1. PLANE（平面）
   ├─ 描述：完全平坦，像操场
   ├─ 难度：⭐（最简单）
   ├─ 用途：基础训练
   └─ 参数：friction（摩擦系数）

2. WAVE（波形）
   ├─ 描述：起伏不平，像波浪
   ├─ 难度：⭐⭐
   ├─ 用途：轻微不平地形
   └─ 参数：amplitude（振幅）, frequency（频率）

3. STAIRS（台阶）
   ├─ 描述：规则的阶梯状
   ├─ 难度：⭐⭐⭐
   ├─ 用途：离散高度变化
   └─ 参数：step_height（阶高）, step_width（阶宽）

4. ROUGH（粗糙地面）
   ├─ 描述：随机粗糙表面
   ├─ 难度：⭐⭐⭐
   ├─ 用途：不规则表面
   └─ 参数：roughness（粗糙度）, size（块大小）

5. SLOPE（斜坡）
   ├─ 描述：均匀的倾斜面
   ├─ 难度：⭐⭐
   ├─ 用途：上下坡行走
   └─ 参数：angle（斜率角度）

混合地形（SubterraniumTerrain）：
   ├─ 描述：上面几种的随机组合
   ├─ 难度：⭐⭐⭐⭐
   ├─ 用途：真实场景模拟
   └─ 参数：各类型地形的比例和参数
```

### 3️⃣ **必学**：地形配置的参数

在 `limx_base_env_cfg.py` 中，地形配置大概长这样：

```python
# 简化版本
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",          # 使用生成器
    terrain_generator=SubterraniumTerrainCfg(
        # 地形生成参数
        size=(100, 100),               # 地形的宽度和长度（m）
        num_rows=20,                   # 有多少行地形
        num_cols=20,                   # 有多少列地形
        horizontal_scale=0.25,         # 地形网格的水平分辨率
        
        # 各地形的配置
        terrain_proportions=[
            0.2,  # 20% PLANE
            0.2,  # 20% WAVE
            0.2,  # 20% STAIRS
            0.2,  # 20% ROUGH
            0.2,  # 20% SLOPE
        ],
        
        # 各地形的参数
        plane_cfg=PlaneTerrainCfg(friction=1.0),
        wave_cfg=WaveTerrainCfg(amplitude=0.2, frequency=0.1),
        stairs_cfg=StairsTerrainCfg(step_height=0.3, step_width=0.4),
        rough_cfg=RoughTerrainCfg(step_height=0.1, step_width=0.1),
        slope_cfg=SlopeTerrainCfg(slope_range=(-0.5, 0.5)),
    ),
)
```

**关键参数解释**：

```
size = (100, 100)
  ├─ 地形有 100m × 100m 那么大
  ├─ 机器人在这个空间里训练
  └─ 足够大让机器人走够远

num_rows = 20, num_cols = 20
  ├─ 把地形分成 20 × 20 = 400 个小块
  ├─ 每个小块是一种地形类型（比如这块是斜坡，那块是台阶）
  ├─ 多样性 → 机器人学会适应变化
  └─ 分块数越多 → 变化越频繁 → 难度越高

horizontal_scale = 0.25
  ├─ 地形网格的分辨率
  ├─ 0.25 意味着每 0.25m 有一个网格点
  ├─ 数值越小 → 地形越光滑
  ├─ 数值越大 → 地形越粗糙（模拟真实凹凸）
  └─ 通常 0.1-0.5 之间

step_height = 0.3
  ├─ 台阶的高度（米）
  ├─ 0.3m = 30cm（差不多膝盖高度）
  ├─ 数值太大 → 机器人无法跨越
  ├─ 数值太小 → 没有挑战
  └─ 通常 0.1-0.5m

slope_range = (-0.5, 0.5)
  ├─ 斜率的范围（弧度或度数）
  ├─ -0.5 表示下坡最陡 30°
  ├─ +0.5 表示上坡最陡 30°
  └─ 范围越大难度越高
```

### 4️⃣ **关键概念**：课程学习（Curriculum Learning）

这是任务 2.4 成功的关键！

```
课程学习的思想（类比上学）：

没有课程学习：
  • 小学一年级小朋友：直接给高等微积分题
  • 结果：完全看不懂，放弃
  
有课程学习：
  • 一年级：学加减法
  • 二年级：学乘除法
  • 三年级：学分数
  • ...
  • 高中：学微积分
  • 结果：循序渐进，最终掌握

强化学习中的课程学习：

阶段 1（初级）：
  • 只有平坦地形
  • 机器人学会基本的行走
  • 奖励容易获得 → 快速学习

阶段 2（中级）：
  • 加入 20% 的波形地形
  • 机器人学会应对轻微起伏
  • 难度略微提升

阶段 3（高级）：
  • 加入 40% 的台阶和斜坡
  • 机器人需要更好的平衡能力
  • 难度显著提升

阶段 4（超级）：
  • 所有地形都有，混合随机出现
  • 机器人需要最大的适应能力
  • 最高难度

实现方式：
  用代码检查"当前阶段是否学得足够好"
  如果足够好 → 升级到下一阶段
  下一阶段的地形更复杂 → 难度提升
```

---

## 第三部分：完整的任务 2.4 工作流程

### 从"什么都不懂"到"完成任务"的步骤

```
第 1 步：理解任务需求
┌──────────────────────────────────────────────┐
│ 目标：机器人从起点 A 走到终点 B              │
│ 路径：经过混合地形（平地→斜坡→台阶→...）  │
│ 评分：能否到达 + 步态平滑度                  │
└──────────────────────────────────────────────┘

第 2 步：设计地形配置
┌──────────────────────────────────────────────┐
│ • 确定地形类型（平面/斜坡/台阶/粗糙）      │
│ • 设置难度等级（参数范围）                  │
│ • 配置课程学习阶段                          │
│   - 阶段 1：只有平面
│   - 阶段 2：平面 + 波形
│   - 阶段 3：加入斜坡
│   - 阶段 4：加入台阶和粗糙
│   - 阶段 5：所有混合
└──────────────────────────────────────────────┘

第 3 步：修改奖励函数
┌──────────────────────────────────────────────┐
│ 新增奖励项：                                 │
│ • 目标位置奖励：靠近终点得奖励              │
│ • 地形适应奖励：在复杂地形上稳定得奖励     │
│ • 存活奖励：不摔倒得奖励                    │
│ • 速度奖励：保持速度得奖励                  │
│                                              │
│ 调整权重以平衡多个目标                      │
└──────────────────────────────────────────────┘

第 4 步：添加观测项（感知地形）
┌──────────────────────────────────────────────┐
│ 新增观测：                                   │
│ • 目标方向：指向终点的向量                  │
│ • 目标距离：到终点还有多远                  │
│ • 地形高度：前方地形的高度信息              │
│ • 坡度信息：当前地形的倾斜角度              │
│                                              │
│ 这些信息帮助神经网络"看到"地形特征         │
└──────────────────────────────────────────────┘

第 5 步：训练和优化
┌──────────────────────────────────────────────┐
│ 初期（1-100 轮）：                          │
│   机器人学会在简单地形上走                  │
│   奖励：-1.0 → 0.5                         │
│                                              │
│ 中期（100-500 轮）：                        │
│   机器人学会适应多种地形                    │
│   奖励：0.5 → 0.8                          │
│                                              │
│ 后期（500-2000 轮）：                       │
│   性能稳定，微调细节                        │
│   奖励：0.8 → 0.95                         │
└──────────────────────────────────────────────┘

第 6 步：评估和测试
┌──────────────────────────────────────────────┐
│ 测试指标：                                   │
│ • 通过率：能走到终点的环境比例              │
│ • 平均时间：走到终点需要多长                │
│ • 步态平滑性：是否有不必要的抖动            │
│ • 适应性：在看不见的地形上表现如何          │
└──────────────────────────────────────────────┘
```

---

## 第四部分：详细的技术实现

### 4.1 修改地形配置

**文件**：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py`

**当前配置**（平地）：
```python
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",  # 只有平面
    max_init_terrain_level=0,
    ...
)
```

**修改为混合地形**：
```python
from isaaclab.terrains import SubterraniumTerrainCfg
from isaaclab.terrains.config import TERRAINS_CFGS

@configclass
class ComplexTerrainCfg(SubterraniumTerrainCfg):
    """复杂地形配置"""
    
    # 基本参数
    size = (200.0, 200.0)              # 200m × 200m 地形
    horizontal_scale = 0.2             # 网格分辨率
    vertical_scale = 0.005             # 高度分辨率
    num_rows = 50                      # 50 行地形
    num_cols = 50                      # 50 列地形
    
    # 地形比例（总和必须 = 1.0）
    terrain_proportions = [
        0.3,  # 30% 平面
        0.2,  # 20% 波形
        0.2,  # 20% 台阶
        0.15, # 15% 粗糙
        0.15, # 15% 斜坡
    ]
    
    # 各地形的具体参数
    plane_cfg = PlaneTerrainCfg(
        friction=1.0,
    )
    
    wave_cfg = WaveTerrainCfg(
        amplitude=0.3,          # 波形幅度 30cm
        frequency=0.1,          # 波频率
    )
    
    stairs_cfg = StairsTerrainCfg(
        step_height=0.3,        # 台阶 30cm 高
        step_width=0.4,         # 台阶 40cm 宽
    )
    
    rough_cfg = RoughTerrainCfg(
        step_height=0.1,        # 粗糙面凹凸 10cm
        step_width=0.1,
        platform_size=2.0,
    )
    
    slope_cfg = SlopeTerrainCfg(
        slope_range=(-0.3, 0.3), # -30° 到 +30°
    )
    
    # 地形边界的物理参数
    border_kwargs = {
        "friction": 1.0,
        "restitution": 0.0,
    }

# 在 SceneCfg 中使用
@configclass
class PFSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ComplexTerrainCfg(),
        max_init_terrain_level=0,  # 初始时用第 0 级难度
    )
```

---

### 4.2 添加目标位置观测和奖励

**第一步**：添加新的观测项

在 `observations.py` 中添加：

```python
def target_relative_position(env: ManagerBasedRLEnv,
                            target_pos: torch.Tensor = None,
                            ) -> torch.Tensor:
    """相对目标位置"""
    asset = env.scene["robot"]
    base_pos = asset.data.root_pos_w  # 当前位置 (x, y, z)
    
    # 获取目标位置（从环境中）
    if not hasattr(env, "target_position"):
        env.target_position = torch.tensor(
            [100.0, 0.0, 0.0], device=env.device
        )  # 默认目标在 100m 远处
    
    # 计算相对位置
    rel_pos = env.target_position - base_pos
    
    # 只考虑 x, y 平面
    rel_pos_xy = rel_pos[:, :2]
    
    return rel_pos_xy

def distance_to_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """到目标的距离"""
    rel_pos = target_relative_position(env)
    distance = torch.norm(rel_pos, dim=1)
    return distance
```

**第二步**：在配置中注册新观测

在 `limx_base_env_cfg.py` 的 `ObservationsCfg` 中：

```python
@configclass
class PolicyCfg(ObsGroup):
    """策略观测"""
    
    # 原有的观测...
    proj_gravity = ObsTerm(func=mdp.projected_gravity, ...)
    # ... 其他观测 ...
    
    # 新增：目标相关观测
    target_rel_pos = ObsTerm(
        func=mdp.target_relative_position,
        scale=0.1,  # 缩放因子（因为距离可能很大）
    )
    
    distance_to_target = ObsTerm(
        func=mdp.distance_to_target,
        scale=0.01,  # 更大的缩放（距离通常是 m）
    )
```

**第三步**：添加目标奖励

在 `rewards.py` 中：

```python
def target_reaching(env: ManagerBasedRLEnv) -> torch.Tensor:
    """到达目标的奖励"""
    asset = env.scene["robot"]
    base_pos = asset.data.root_pos_w[:, :2]  # 当前位置 (x, y)
    
    if not hasattr(env, "target_position"):
        env.target_position = torch.tensor(
            [100.0, 0.0], device=env.device
        ).unsqueeze(0)
    
    # 到目标的距离
    distance = torch.norm(base_pos - env.target_position, dim=1)
    
    # 奖励与距离成反比（用高斯函数）
    reward = torch.exp(-distance ** 2 / (10.0 ** 2))  # 10m 范围
    
    return reward

def goal_reached_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """到达目标的额外奖励"""
    asset = env.scene["robot"]
    base_pos = asset.data.root_pos_w[:, :2]
    
    if not hasattr(env, "target_position"):
        env.target_position = torch.tensor(
            [100.0, 0.0], device=env.device
        ).unsqueeze(0)
    
    distance = torch.norm(base_pos - env.target_position, dim=1)
    
    # 如果距离 < 2m，给大奖励
    reward = torch.where(
        distance < 2.0,
        torch.ones_like(distance) * 10.0,  # 到达目标：+10
        torch.zeros_like(distance)          # 没到达：0
    )
    
    return reward
```

在 `limx_base_env_cfg.py` 的 `RewardsCfg` 中注册：

```python
@configclass
class RewardsCfg:
    """奖励配置"""
    
    # 原有的奖励...
    base_tracking = RewTerm(func=mdp.base_tracking, weight=1.0)
    # ... 其他奖励 ...
    
    # 新增：目标相关奖励
    target_reaching = RewTerm(
        func=mdp.target_reaching,
        weight=0.5,  # 比速度追踪权重小，平衡两个目标
    )
    
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=1.0,  # 到达目标很重要，权重较大
    )
```

---

### 4.3 实现课程学习

**文件**：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py`

```python
@configclass
class CurriculumCfg:
    """课程学习配置"""
    
    @configclass
    class TerrainLevelCfg:
        """地形难度等级"""
        name: str = "level_0"
        
        # 地形参数在这一等级的值
        terrain_proportions: List[float] = field(default_factory=list)
        
        # 触发升级的条件
        min_avg_reward: float = 0.7     # 平均奖励达到 0.7
        min_episodes: int = 100         # 至少训练 100 轮
        
        # 地形特定参数（可选）
        step_height_range: tuple = (0.1, 0.3)
        slope_range: tuple = (-0.2, 0.2)
    
    # 定义各个难度等级
    level_0 = TerrainLevelCfg(
        name="flat_only",
        terrain_proportions=[1.0, 0.0, 0.0, 0.0, 0.0],  # 100% 平面
        min_avg_reward=0.3,
        min_episodes=50,
    )
    
    level_1 = TerrainLevelCfg(
        name="flat_wave",
        terrain_proportions=[0.7, 0.3, 0.0, 0.0, 0.0],  # 70% 平面 + 30% 波形
        min_avg_reward=0.5,
        min_episodes=100,
        slope_range=(-0.1, 0.1),
    )
    
    level_2 = TerrainLevelCfg(
        name="with_stairs",
        terrain_proportions=[0.4, 0.2, 0.2, 0.1, 0.1],  # 混合加台阶
        min_avg_reward=0.6,
        min_episodes=150,
        step_height_range=(0.2, 0.3),
    )
    
    level_3 = TerrainLevelCfg(
        name="with_slopes",
        terrain_proportions=[0.3, 0.2, 0.2, 0.15, 0.15],  # 加入斜坡
        min_avg_reward=0.65,
        min_episodes=200,
        slope_range=(-0.25, 0.25),
    )
    
    level_4 = TerrainLevelCfg(
        name="full_mixed",
        terrain_proportions=[0.2, 0.2, 0.2, 0.2, 0.2],  # 全混合
        min_avg_reward=0.7,
        min_episodes=300,
        step_height_range=(0.1, 0.3),
        slope_range=(-0.3, 0.3),
    )
    
    levels = [level_0, level_1, level_2, level_3, level_4]


# 课程学习的核心逻辑
class CurriculumManager:
    """管理课程学习的进度"""
    
    def __init__(self, cfg: CurriculumCfg, env):
        self.cfg = cfg
        self.env = env
        self.current_level = 0
        self.episode_count = 0
        self.reward_history = []
    
    def update(self, episode_reward):
        """每轮更新一次"""
        self.episode_count += 1
        self.reward_history.append(episode_reward)
        
        # 计算最近 N 轮的平均奖励
        recent_episodes = min(50, len(self.reward_history))
        avg_reward = sum(self.reward_history[-recent_episodes:]) / recent_episodes
        
        # 检查是否可以升级
        current_level_cfg = self.cfg.levels[self.current_level]
        
        if (avg_reward >= current_level_cfg.min_avg_reward and
            self.episode_count >= current_level_cfg.min_episodes and
            self.current_level < len(self.cfg.levels) - 1):
            
            # 升级到下一等级
            self.current_level += 1
            self.episode_count = 0
            
            # 应用新的地形配置
            next_level_cfg = self.cfg.levels[self.current_level]
            print(f"升级到难度等级 {self.current_level}: {next_level_cfg.name}")
            self._apply_level_config(next_level_cfg)
    
    def _apply_level_config(self, level_cfg: CurriculumCfg.TerrainLevelCfg):
        """应用新的难度等级配置"""
        # 这需要动态修改地形生成参数
        # 具体实现取决于 Isaac Lab 的 API
        pass
    
    def get_current_level_name(self):
        """获取当前难度等级名称"""
        return self.cfg.levels[self.current_level].name
```

---

### 4.4 修改终止条件（检测到达目标）

在 `limx_base_env_cfg.py` 中的 `TerminationsCfg`：

```python
@configclass
class TerminationsCfg:
    """环境终止条件"""
    
    # 原有的终止条件...
    time_out = DoneTerm(
        func=mdp.time_limit,
        time_limit=12.5,  # 12.5 秒 = 2500 步
    )
    
    # 新增：到达目标时终止（成功）
    reached_goal = DoneTerm(
        func=mdp.goal_reached,  # 需要实现这个函数
    )

# 在 rewards.py 中实现
def goal_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """检查是否到达目标"""
    asset = env.scene["robot"]
    base_pos = asset.data.root_pos_w[:, :2]
    
    if not hasattr(env, "target_position"):
        env.target_position = torch.tensor(
            [100.0, 0.0], device=env.device
        ).unsqueeze(0)
    
    distance = torch.norm(base_pos - env.target_position, dim=1)
    
    # 距离 < 1m 时，认为到达目标
    return distance < 1.0
```

---

## 第五部分：逐步优化的完整计划

### 第 1 天：理论和验证

```
上午：
  ☐ 读懂 TerrainGenerator 的概念
  ☐ 了解 Isaac Lab 支持的各种地形
  ☐ 理解课程学习的原理
  
下午：
  ☐ 在现有代码中找到地形配置
  ☐ 尝试运行平地训练（确保基础工作）
  ☐ 观察学习曲线，记录基准性能
  
晚上：
  ☐ 规划要添加的地形类型
  ☐ 准备修改代码所需的文件
```

**验证步骤**：

```bash
# 1. 确认当前在平地上能训练
python scripts/train.py --task=PointFootLocomotion --headless

# 2. 运行 100 轮，记录奖励（应该是 0.7+ 的稳定值）
# 3. 用这个作为基准，之后对比

# 4. 保存这个初始模型
#    logs/PointFootLocomotion-PointFootLocomotionPPO/*/model.pt
```

---

### 第 2 天：配置地形和观测

```
上午：
  ☐ 修改 limx_base_env_cfg.py 中的地形配置
  ☐ 改为混合地形（从简单开始，比如 80% 平面 + 20% 波形）
  ☐ 测试地形生成是否正确
  
下午：
  ☐ 添加目标位置观测
  ☐ 在 observations.py 中实现 target_relative_position
  ☐ 注册到 ObservationsCfg 中
  
晚上：
  ☐ 开始训练新配置
  ☐ 观察机器人是否能适应新地形
```

**期望结果**：
```
训练 50 轮后，应该看到：
  • 奖励从负数逐渐增加
  • 机器人能在大部分地形上走
  • 偶尔摔倒（正常，因为难度增加了）
  
如果摔倒次数太多 (> 30%)，说明难度过高：
  • 增加平面地形比例
  • 减小地形参数（如阶高）
  • 增加平衡相关的奖励
```

---

### 第 3 天：添加奖励和目标导向

```
上午：
  ☐ 实现目标相关的奖励函数
  ☐ 在 rewards.py 中添加 target_reaching 和 goal_reached_bonus
  ☐ 注册到 RewardsCfg 中
  
下午：
  ☐ 调整奖励权重
  ☐ 平衡：速度追踪 vs 目标追踪
  ☐ 推荐：base_tracking 权重保持 1.0，target_reaching 权重 0.5
  
晚上：
  ☐ 开始训练带目标的任务
  ☐ 机器人应该开始朝目标行走
```

**期望结果**：
```
训练 100 轮后，应该看到：
  • 机器人能在混合地形上相对稳定地走
  • 有"目标意识"（虽然可能还是会走偏）
  • 奖励值应该 > 0（如果是负的，说明有问题）

可能的问题：
  问题：机器人忽视目标，只关心速度
  原因：target_reaching 权重太小
  解决：增加权重到 1.0 或更高
  
  问题：机器人一直朝目标冲但会摔倒
  原因：对地形适应不足
  解决：增加课程学习的难度过渡
```

---

### 第 4 天：实现课程学习

```
上午：
  ☐ 在 curriculums.py 中实现 CurriculumManager
  ☐ 定义 5 个难度等级
  ☐ 设置每个等级的升级条件
  
下午：
  ☐ 集成课程学习到训练循环
  ☐ 修改 train.py，使其每轮调用 curriculum.update()
  ☐ 添加日志输出显示当前难度等级
  
晚上：
  ☐ 启动课程学习训练
  ☐ 监测难度等级的升级过程
```

**期望结果**：
```
训练 500 轮后，应该看到：
  
  轮次 0-100：Level 0（平面）→ 快速学习，奖励快速增长
  轮次 100-200：Level 1（平面+波形）→ 学习变慢，适应新地形
  轮次 200-300：Level 2（加台阶）→ 学习更慢
  轮次 300-400：Level 3（加斜坡）→ 性能稳定
  轮次 400-500：Level 4（全混合）→ 最终形态
  
最终奖励应该稳定在 0.75-0.9 左右
```

---

### 第 5 天：微调和优化

```
基于前 4 天的结果，进行微调：

如果机器人经常摔倒：
  ☐ 增加 stay_alive 权重
  ☐ 增加 feet_regulation 权重
  ☐ 调整 PD 参数（增加刚度）
  ☐ 减小地形参数（小的阶高、低的坡度）

如果机器人速度太慢：
  ☐ 增加 base_tracking 权重
  ☐ 减少平衡相关惩罚

如果机器人不朝目标走：
  ☐ 增加 target_reaching 权重
  ☐ 检查目标位置观测是否正确注册

如果难度升级太快/太慢：
  ☐ 调整 CurriculumCfg 中的 min_avg_reward
  ☐ 调整 min_episodes
```

---

## 第六部分：评估和测试

### 评分标准详解

```
1. 通过率（Pass Rate）
   ┌─────────────────────────────────────────┐
   │ 定义：在给定时间内到达目标的环境比例   │
   │                                         │
   │ 100% → 100 分 ✓✓✓
   │ 80-99% → 90 分
   │ 50-79% → 70 分
   │ 20-49% → 50 分
   │ < 20% → 20 分
   │                                         │
   │ 评估时间：通常给 2-3 倍的标准时间     │
   │ 比如训练时用 12.5 秒，测试给 30 秒    │
   └─────────────────────────────────────────┘

2. 地形适应性（Terrain Adaptability）
   ┌─────────────────────────────────────────┐
   │ 定义：步态平滑度和稳定性                │
   │                                         │
   │ 测量指标：
   │ • Base 的加速度幅度（越小越好）
   │ • 姿态的摇晃幅度（Roll/Pitch）
   │ • 足部接触的规律性
   │                                         │
   │ 评分：
   │ 加速度 < 1 m/s² → 100 分
   │ 加速度 1-2 → 80 分
   │ 加速度 2-5 → 50 分
   │ 加速度 > 5 → 20 分
   └─────────────────────────────────────────┘
```

### 测试流程

```bash
# 1. 保存训练好的模型
CHECKPOINT="logs/PointFootLocomotion-PointFootLocomotionPPO/*/model.pt"

# 2. 在测试环境中运行
python scripts/play.py --task=PointFootLocomotion \
    --checkpoint=$CHECKPOINT \
    --num_envs=256 \
    --test_episodes=100

# 3. 收集统计数据
# 脚本应该输出：
#   - 通过率
#   - 平均完成时间
#   - 平均加速度
#   - 摔倒次数

# 4. 分析结果
# 对比和平地训练的差异
```

---

## 第七部分：常见问题和解决方案

### ❌ 问题 1：机器人无法在地形上行走

```
症状：
  • 训练开始后立即摔倒
  • 奖励始终为负
  • 机器人几乎没有前进

原因分析：
  1. 地形太难（最可能）
  2. 奖励函数有问题
  3. PD 参数不适应新地形

解决方案：
  
  ✓ 立即尝试：减少地形复杂度
    # 在 limx_base_env_cfg.py 中
    terrain_proportions=[0.9, 0.1, 0.0, 0.0, 0.0]  # 90% 平面 + 10% 波形
    
    重新训练 50 轮，观察是否改善
  
  ✓ 其次：调整 PD 参数
    # 在 pointfoot_cfg.py 中
    stiffness=30.0,  # 从 25 增加到 30
    damping=1.2,     # 从 0.8 增加到 1.2
    
    更硬的关节 → 更容易在不平地形上稳定
  
  ✓ 再者：增加稳定性奖励
    # 在 limx_base_env_cfg.py 中
    stay_alive = RewTerm(weight=2.0)  # 从 0.5 增加到 2.0
    
    更容易活下来 → 更多尝试 → 更快学习
```

### ❌ 问题 2：课程学习不升级

```
症状：
  • 训练很久还停留在 Level 0
  • 日志显示"升级条件不满足"

原因分析：
  • 升级条件设置过严格
  • min_avg_reward 设置太高
  • min_episodes 设置太多

解决方案：
  
  # 在 curriculums.py 中放松条件
  
  level_0 = TerrainLevelCfg(
      ...
      min_avg_reward=0.2,      # 从 0.3 降到 0.2
      min_episodes=30,         # 从 50 降到 30
  )
  
  level_1 = TerrainLevelCfg(
      ...
      min_avg_reward=0.4,      # 从 0.5 降到 0.4
      min_episodes=60,         # 从 100 降到 60
  )
  
  # 也可以暂时禁用课程学习，用固定难度训练
  # 看看是否能达到预期奖励
```

### ❌ 问题 3：目标位置观测不起作用

```
症状：
  • 机器人不朝目标走
  • 完全随机行走
  • 观测中有目标信息，但被忽视

原因分析：
  • 目标奖励权重太小
  • 速度奖励权重太大（支配学习）
  • 目标位置设置有问题

解决方案：
  
  # 临时禁用其他奖励，只保留目标奖励
  target_reaching = RewTerm(
      func=mdp.target_reaching,
      weight=2.0,  # 增加权重
  )
  
  goal_reached_bonus = RewTerm(
      func=mdp.goal_reached_bonus,
      weight=2.0,
  )
  
  # 禁用或降低速度奖励
  base_tracking = RewTerm(
      func=mdp.base_tracking,
      weight=0.1,  # 大幅降低
  )
  
  # 训练 50 轮，看机器人是否朝目标走
  # 如果行为正确，再增加速度奖励权重
  
  # 同时检查目标位置是否正确设置
  if not hasattr(env, "target_position"):
      print("WARNING: 目标位置未设置！")
```

### ❌ 问题 4：性能下降（灾难遗忘）

```
症状：
  • 在简单地形上的性能下降
  • 升级到困难地形后，连平面都走不好了
  
原因分析：
  • 课程升级太快
  • 难地形的奖励信号压倒了简单地形的学习

解决方案：
  
  # 增加难地形中简单地形的比例
  level_4 = TerrainLevelCfg(
      terrain_proportions=[0.4, 0.2, 0.2, 0.1, 0.1],  # 增加平面比例
  )
  
  # 减缓课程升级
  level_3 = TerrainLevelCfg(
      min_episodes=500,  # 增加训练轮数
  )
  
  # 或者使用"混合课程"
  # 每一等级都包含前面等级的一部分
  # 这样可以防止遗忘简单技能
```

---

## 第八部分：关键文件改动总结

### 你需要修改的文件列表

```
优先级 1️⃣（必改）：
  1. limx_base_env_cfg.py
     ├─ 修改地形配置为混合地形
     ├─ 添加目标相关观测
     ├─ 添加目标相关奖励
     └─ 修改终止条件
  
  2. observations.py
     ├─ 添加 target_relative_position()
     └─ 添加 distance_to_target()
  
  3. rewards.py
     ├─ 添加 target_reaching()
     ├─ 添加 goal_reached_bonus()
     └─ 添加 goal_reached() 用于终止条件

优先级 2️⃣（可选但推荐）：
  1. curriculums.py
     ├─ 实现 CurriculumManager
     └─ 定义难度等级
  
  2. train.py
     └─ 集成课程学习逻辑
  
  3. pointfoot_cfg.py
     ├─ 调整 PD 参数（如果需要）
     └─ 调整执行器参数

优先级 3️⃣（高级）：
  1. 自定义终止回调（events.py）
  2. 地形特定的奖励（如"在坡上稳定"）
  3. 可视化脚本（观察机器人在地形上的行为）
```

---

## 总结：一句话的学习路径

```
不了解 → 理解 TerrainGenerator 工作原理
         → 理解课程学习的必要性
         → 学会配置混合地形
         → 添加目标导向奖励
         → 实现课程学习
         → 调试和优化
         → 成功完成任务

预计总时间：
  理论学习：4-6 小时
  代码实现：3-5 小时
  调试优化：5-10 小时
  总计：12-21 小时（3-5 天集中工作）
```

---

## 快速参考：最关键的 3 个概念

### 1️⃣ TerrainGenerator（必懂）
```
= 自动生成各种地形的工具
通过修改参数（如高度、坡度）快速改变环境难度
避免手工建模的麻烦
```

### 2️⃣ 课程学习（关键）
```
= 从简到难的训练策略
Level 0：平面（容易）
Level 1-4：逐步加入复杂地形
好处：快速收敛，避免卡住
```

### 3️⃣ 目标导向（目的）
```
= 加入"到达目标"的奖励
让神经网络知道"方向"
结合速度奖励 → 既要快，还要朝对方向
```

现在你已经有了完整的学习地图！开始实战吧 🚀

