# limx_base_env_cfg.py 详细问答

本文以问答形式解释 `limx_base_env_cfg.py` 中的各个配置块，并指出任务 2.2 / 2.3 / 2.4 需要重点调整的参数。

---

## 总览

**Q: 这个文件的作用是什么？**  
A: 定义训练环境的完整配置：场景、命令、动作、观测、事件（域随机化）、奖励、终止条件、课程学习，以及顶层环境参数。

**Q: 三个任务分别主要改哪里？**  
- 任务 2.2（平地速度跟随）：改奖励权重、观测噪声、命令范围。  
- 任务 2.3（抗干扰）：改/加事件中的外力推扰；可增减惩罚和稳定性奖励。  
- 任务 2.4（复杂地形）：改场景地形为生成器；可能增加课程学习；调整奖励与观测。

---

## 场景配置 `PFSceneCfg`

**Q: 地形现在是什么？怎么改？**  
A: 目前 `terrain_type="plane"`，平地。要切换到地形生成器：
```python
terrain_type="generator"
terrain_generator=...  # 选用地形生成器配置
```
这是任务 2.4 的关键改动。

**Q: 光照和材质有什么用？**  
A: 视觉效果；不影响训练物理，可保持默认。

**Q: 传感器有哪些？**  
A: `height_scanner`（待子类定义）、`contact_forces`（记录接触与空中时间）。接触传感器用于奖励和惩罚。

**Q: `env_spacing`、`num_envs` 在哪？**  
A: 在顶层 `PFEnvCfg.scene = PFSceneCfg(num_envs=4096, env_spacing=2.5)`。4096 环境并行，env 间距 2.5m。

---

## 命令配置 `CommandCfg`

**Q: 控制的命令是什么？**  
A: 速度命令 (`base_velocity`) 和步态命令 (`gait_command`)。

**Q: 速度命令范围在哪里？**  
A: `self.base_velocity.ranges`：
- `lin_vel_x=(-1.5, 1.5)` 前后速度
- `lin_vel_y=(-1.0, 1.0)` 侧向速度
- `ang_vel_z=(-0.5, 0.5)` 角速度

**Q: 任务 2.2 要改吗？**  
A: 一般无需改，可根据目标速度范围适当放宽/收紧。若想更聚焦前向速度，可收窄 `lin_vel_y`。

**Q: 步态命令有什么用？**  
A: 随机步态参数（频率、相位、接触时长、摆动高度）。保持即可，除非要固定特定步态。

---

## 动作配置 `ActionsCfg`

**Q: 动作是什么？**  
A: 关节位置目标（6 个关节），scale=0.25，使用默认偏移。

**Q: 任务需要改吗？**  
A: 一般不改。若动作过激，可减小 scale；若动作响应太小，可增大 scale（谨慎）。

---

## 观测配置 `ObservarionsCfg`

**Q: 包含哪些观测组？**  
- `policy`：给策略网络的 59 维观测（含噪声）。
- `obsHistory`：历史观测（长度 10，不展平）。
- `critic`：特权观测（无噪声，含力矩、惯量等）。
- `commands`：命令观测。

**Q: 噪声在哪里调？**  
A: `GaussianNoise(std=...)`。任务 2.2 可适当减小噪声帮助收敛；任务 2.3/2.4 可保留或略增以增强鲁棒性。

**Q: 如果要加入目标相关观测（任务 2.4）？**  
A: 在 `PolicyCfg` 和/或 `CriticCfg` 增加新的 `ObsTerm`，如 `target_relative_position`，并实现对应函数于 `mdp/observations.py`。

**Q: 历史观测的作用？**  
A: 提供时间信息（10 步），对平滑控制和鲁棒性有帮助，一般不改。

---

## 事件配置 `EventsCfg`（域随机化）

**Q: 有哪些事件类型？**  
- startup：质量、惯量、材质、执行器刚度阻尼、重心。
- reset：根状态、关节状态重置。
- interval：外力推扰 `push_robot`。

**Q: 任务 2.3 需要怎么改？**  
A: 强化外力扰动。现在 `push_robot` 用 `apply_external_force_torque_stochastic`，力范围 ±500N，概率 0.002。可以：
- 增大 `force_range` 或 `probability`；
- 改为基于速度的推扰 `push_by_setting_velocity`（需要在 `mdp/events.py` 实现）。

**Q: 任务 2.2 要改吗？**  
A: 可先降低或关闭推扰（把概率设小）以便收敛。

**Q: 任务 2.4 呢？**  
A: 可结合地形生成器，保持随机化以增强泛化。

---

## 奖励配置 `RewardsCfg`

**Q: 核心跟踪奖励？**  
- `rew_lin_vel_xy` (weight=3.0) 跟踪平面线速度
- `rew_ang_vel_z` (weight=1.5) 跟踪角速度

**Q: 主要惩罚项？**  
- `pen_base_height` (-20.0)
- 关节扭矩/加速度/速度/功率惩罚
- 动作速率和平滑度惩罚
- 不期望接触、足距、足部调节、着陆速度、姿态平坦性

**Q: 步态奖励？**  
`test_gait_reward`（weight=1.0）基于接触/速度概率模型。

**Q: 任务 2.2 调参建议**  
- 提高 `rew_lin_vel_xy` 到 4~6，`rew_ang_vel_z` 到 2~3，促进跟踪。
- 适度减小平滑度惩罚（绝对值不要过小防止抖动）。
- 若摔倒多，降低 `pen_base_height` 负值幅度或增加平衡相关奖励。

**Q: 任务 2.3 调参建议**  
- 增加姿态/稳定性相关奖励（可在 `rewards.py` 添加新函数，如加速度惩罚、姿态稳定奖励）。
- 适度提高动作平滑与能量惩罚，避免抗扰时乱蹬。

**Q: 任务 2.4 调参建议**  
- 对地形/目标相关的奖励（如到达目标、通过障碍）需新增 `RewTerm`。
- 可能降低平地特定惩罚，增加通过性奖励。

---

## 终止条件 `TerminationsCfg`

**Q: 有哪些终止？**  
- `time_out`：到达 episode 长度
- `base_contact`：基座接触（摔倒）

**Q: 要改吗？**  
A: 一般不改。可调 `threshold` 使判定更宽/严。

---

## 课程学习 `CurriculumCfg`

**Q: 现在有什么？**  
A: `terrain_levels = mdp.terrain_levels_vel`，用于基于速度/地形难度的课程调整。

**Q: 任务 2.4 需要吗？**  
A: 推荐结合地形生成器使用，逐步提升难度。

---

## 顶层环境配置 `PFEnvCfg`

**Q: 关键参数有哪些？**  
- `num_envs=4096`：并行环境数
- `env_spacing=2.5`：环境间距
- `decimation=4`：控制频率 1/(dt*decimation) ≈ 50Hz/4 ≈ 12.5Hz
- `sim.dt=0.005`：物理步长 5ms
- `episode_length_s=20.0`：每个 episode 20 秒
- `seed=42`：随机种子

**Q: 要调哪些来加速/稳定？**  
- 训练速度不够：可减小 `episode_length_s` 做快速迭代；或减小 `num_envs`（但会降低收敛速度）。
- 控制更频繁：减小 `decimation`（会增算力消耗）。

---

## 任务导向修改清单

**任务 2.2（平地速度跟随）**
- 奖励：提高 `rew_lin_vel_xy`/`rew_ang_vel_z`；视情况减小动作惩罚；保持平衡相关惩罚适中。
- 观测噪声：可略减小帮助收敛。
- 事件：可降低 `push_robot` 概率/力度。

**任务 2.3（抗干扰）**
- 事件：增强 `push_robot` 力度或概率，或改为速度推扰。
- 奖励：增加稳定性/加速度惩罚；保持平滑惩罚。
- 观测：可加入线加速度等（在 `observations.py` 添加）。

**任务 2.4（复杂地形）**
- 场景：改 `terrain_type="generator"` 并提供生成器配置。
- 奖励：添加通过性或目标奖励。
- 观测：添加目标/地形相关观测。
- 课程：启用/强化 `terrain_levels`。

---

## 参考修改位置速查

- 场景与地形：`PFSceneCfg.terrain`
- 命令范围：`CommandCfg.__post_init__` 中的 `base_velocity.ranges`
- 动作缩放：`ActionsCfg.joint_pos.scale`
- 观测噪声：`ObservarionsCfg.PolicyCfg` 等 `GaussianNoise(std=...)`
- 外力事件：`EventsCfg.push_robot`
- 奖励权重：`RewardsCfg` 中各 `RewTerm(weight=...)`
- 终止阈值：`TerminationsCfg.base_contact.params["threshold"]`
- 课程：`CurriculumCfg.terrain_levels`

---

## 如果要新增功能，放哪？

- 新奖励函数：`bipedal_locomotion/tasks/locomotion/mdp/rewards.py`，然后在本文件 `RewardsCfg` 挂上 `RewTerm`。
- 新观测函数：`mdp/observations.py`，再在 `PolicyCfg`/`CriticCfg` 加 `ObsTerm`。
- 新事件（外力/参数随机化）：`mdp/events.py`，再在 `EventsCfg` 加 `EventTerm`。
- 新课程逻辑：`mdp/curriculums.py`，再在 `CurriculumCfg` 引用。

---

## 最小改动示例（任务 2.2）

1) 提升速度跟踪奖励：
```python
rew_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)})
rew_ang_vel_z  = RewTerm(func=mdp.track_ang_vel_z_exp,  weight=2.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)})
```
2) 减小推扰：
```python
push_robot.params["probability"] = 0.0005
```
3) 略减噪声（可选）：
```python
proj_gravity.noise.std = 0.015
joint_pos.noise.std    = 0.008
joint_vel.noise.std    = 0.008
```

---

## 最小改动示例（任务 2.3）

1) 增强推扰：
```python
push_robot.params["probability"] = 0.01
push_robot.params["force_range"] = {"x": (-800, 800), "y": (-800, 800), "z": (0, 0)}
```
2) 在 `rewards.py` 添加加速度惩罚并挂载：
```python
pen_base_acc = RewTerm(func=mdp.base_acc_l2, weight=-0.5)
```
3) 观测中可加入线加速度（需在 observations.py 实现）。

---

## 最小改动示例（任务 2.4）

1) 改地形：
```python
terrain_type="generator"
terrain_generator=YourTerrainGeneratorCfg(...)
```
2) 新奖励：在 `RewardsCfg` 添加目标/通过性奖励，并在 `rewards.py` 实现。
3) 新观测：在 `PolicyCfg` 添加目标位置、地形信息观测（在 `observations.py` 实现）。
4) 课程：确保 `terrain_levels` 与生成器联动。

---

## 结论

`limx_base_env_cfg.py` 是所有任务的核心配置。按任务需求重点调整：
- 2.2：奖励权重、噪声、推扰弱化
- 2.3：推扰增强、稳定性相关奖励/观测
- 2.4：地形生成、课程学习、目标/地形相关奖励与观测

修改前建议备份；每次只改少量参数，跑 50~100 轮快速验证，再做下一步调整。
