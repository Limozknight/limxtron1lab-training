# 模块详解一：场景与机器人配置 (Scene Configuration)

> **目标**：理解如何把机器人放进虚拟世界里，并给它定义好身体结构。

## 1. 虚拟世界的搭建 (`PFSceneCfg`)

在 `limx_base_env_cfg.py` 文件中，`PFSceneCfg` 类负责定义整个“舞台”。

### 1.1 地形 (Terrain)
代码如下：
```python
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",  # <---哪怕是纯小白也看得懂，这里选了"平面"
    ...
)
```
*   **`terrain_type="plane"`**: 这意味着我们在一个无限大的平地上训练。对于初学者和基础步态训练，这是最简单的开始。如果以后要走楼梯，这里会改成其他类型。
*   **`primary_path`**: 这是地形在 USD（通用场景描述）文件结构中的“地址”。

### 1.2 机器人资产 (Robot Asset)
这是最关键的部分。代码并没有直接在这里写机器人的长相，而是引用了一个 USD 文件。

在 `pointfoot_cfg.py` 文件中：
```python
usd_path = os.path.join(current_dir, "../usd/PF_TRON1A/PF_TRON1A.usd")
```
User Universal Scene Description (USD) 是 Pixar 公司发明的一种文件格式，你可以把它理解为 **3D 世界的 PDF**。它里面包含了：
*   机器人的外观（模型网格）
*   机器人的连杆（大腿、小腿、脚）
*   关节的连接关系

### 1.3 关节定义 (Joints Breakdown)

双足机器人（PointFoot）的身体结构在代码中被精确定义。它有两条腿，每条腿有 3 个关节，一共 6 个**自由度 (DOF)**。

```python
joint_names=[
    "abad_L_Joint", "abad_R_Joint", # 髋关节-侧向（外展/内收）：负责两腿张开/合并
    "hip_L_Joint",  "hip_R_Joint",  # 髋关节-前后（屈曲/伸展）：负责大腿前后摆动
    "knee_L_Joint", "knee_R_Joint"  # 膝关节（屈曲/伸展）：负责小腿伸缩
]
```

*   **L** 代表 Left (左)，**R** 代表 Right (右)。
*   这 6 个关节就是我们要控制的核心对象。神经网络最后输出的 6 个数字，就是给这 6 个关节下达的命令。

### 1.4 物理属性 (Physics Properties)
我们还定义了机器人是不是会“穿模”（穿过自己）。

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True, # <--- 开启自碰撞检测
    ...
)
```
**为什么这很重要？**
如果不开这个选项，机器人的左脚可能会直接穿过右脚，这在现实物理中是不可能的。开启它，如果两脚相撞，物理引擎（PhysX）就会模拟出碰撞反弹的效果，逼迫机器人学会走路不要“绊脚”。

## 2. 初始状态 (Initial State)

当训练开始的第一帧，机器人是什么姿势？
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),   # 把它悬空放在 1米 高的地方
    joint_pos={...: 0.0},  # 所有关节归零（通常是直立或自然下垂姿态）
)
```
我们把它放在 1 米高，是因为在仿真开始的瞬间，机器人会自然下落触地，这个过程可以测试它的着陆稳定性。
