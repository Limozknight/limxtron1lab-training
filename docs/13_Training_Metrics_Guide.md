# 训练可视化指标解读指南（以 Task 2.2 输出为例）

> 帮你快速判断训练是否健康、是否收敛，以及问题出在哪。示例基于 2.2 的曲线，但同样适用于 2.3/2.4。

## 快速检查清单（先看这 7 个）
- Train/mean_reward：整体任务完成度；应平滑上升并趋于平台。
- Train/mean_episode_length：稳定性；趋近最大步数说明存活久。
- Episode/keep_balance & Episode/rew_base_stability：平衡与姿态质量；需持续上升或持平。
- Episode/rew_lin_vel_xy_precise & Episode/rew_ang_vel_z_precise：速度跟踪精度；上升后波动变小为好。
- Episode/pen_base_height：地形/高度控制；数值小且不再变糟（更负）为好。
- Loss/value_function：估计器稳定性；应缓慢下降到 0.x-1.x 并保持低噪声。
- Policy/mean_kl：策略更新幅度；应低且平稳，尖峰少。

## 各分组怎么看
### Train 分组
- mean_reward / mean_reward/time：核心 KPI。上升 → 收敛；长平台期 → 可能学习率过小或奖励权重不敏感。
- mean_episode_length：接近最大步数，说明少摔倒；若突然下降，通常是探测到难例或策略发散。

### Episode/Reward（正向）
- keep_balance：平衡能力。阶梯式上升常见；后期持平即可。
- rew_lin_vel_xy / rew_lin_vel_xy_precise：速度跟踪。先粗后精，曲线抬升且抖动收敛表明控制变稳。
- rew_ang_vel_z / rew_ang_vel_z_precise：转向跟踪。与线速度类似，关注是否高波动（可能动作噪声大）。
- rew_base_stability：躯干姿态。持续上升或持平即可。

### Episode/Reward（惩罚项）
目标：绝对值小且稳定，不要持续变得更负。
- pen_base_height：地形/高度约束。突然变负 → 可能踩空、台阶过高、或传感噪声。
- pen_lin_vel_z：竖直速度抖动。下探过深或弹跳会让其变更负。
- pen_ang_vel_xy：俯仰/翻滚角速度。更负 → 身体晃动大，可调增姿态稳定奖励或减动作噪声。
- pen_joint_torque / pen_action_rate：能耗与动作平滑。持续变负 → 关节控制过猛，可调低动作尺度或增平滑惩罚。

### Loss 分组
- value_function：开始有尖峰正常，数百轮后应 <~2 且波动小。持续高或抖动大 → 可能优势估计噪声或价值网络欠拟合。
- encoder（如果用特征编码器）：应快速降到接近 0，若停在高位说明特征难学或 lr 偏低。
- surrogate：PPO 损失；应围绕 0 小幅波动，长时间正/负偏移说明剪切或优势分布不平衡。
- learning_rate：确认调度是否按预期衰减。

### Policy 分组
- mean_noise_std：动作探索强度。初期下降后略回升是正常；若持续上升导致控制抖动，可下调初始噪声或加平滑惩罚。
- mean_kl：策略步幅。应保持低且平稳；频繁尖峰 → 可能需要调低 clip 或增 mini-batch 数。

### Perf 分组
- total_fps / collection_time / learning_time：吞吐与算力健康度。骤降 → 可能被 IO/显存/CPU 限制或调试日志过多。

## 典型模式与处理
- 平台期很长（奖励不上升）：尝试略增学习率，或放大奖励差异（如提升主奖励权重）。
- 惩罚变得更负：检查对应物理量（高度、角速度、扭矩）；降低动作尺度、增平滑、或调整终止条件。
- value_loss 长期高波动：调小价值损失系数，或增加 GAE 平滑（更小的 gae_lambda）。
- mean_kl 尖峰：减小学习率或 PPO clip；必要时增 mini-batch。
- mean_noise_std 回升过大：减低初始噪声或动作尺度；若任务需要探索，可配合更强的平滑惩罚。

## 如何用这些图判断“训得好不好”
1) 先看 Train/mean_reward 与 mean_episode_length 是否同步上升并持平。
2) 再看主奖励（速度/姿态）是否上升、惩罚是否稳定不恶化。
3) 确认 value_loss、mean_kl、mean_noise_std 没有持续抬升或尖峰连发。
4) 性能图是否稳定；若 FPS 波动但奖励正常，可忽略。

## 对当前 2.2 曲线的快速点评（示例）
- 主奖励（速度、角速度、姿态）阶梯式上升后平台，符合收敛预期。
- 惩罚项在早期变负后趋稳，属正常；如需更平滑，可微调动作尺度和能耗惩罚。
- value_loss 已降到低位且平稳，mean_kl 低且无尖峰，策略更新健康。
- mean_reward 与 episode_length 均接近平台，说明策略已基本收敛。

## 记录/对比的最小图表集合（建议常看）
- Train: mean_reward, mean_episode_length
- Episode: keep_balance, rew_lin_vel_xy_precise, rew_ang_vel_z_precise, rew_base_stability
- Episode penalties: pen_base_height, pen_ang_vel_xy, pen_action_rate
- Loss: value_function, surrogate
- Policy: mean_noise_std, mean_kl
- Perf: total_fps

将这些图表加入收藏夹或仪表板，日常只看这一套即可快速发现问题。
