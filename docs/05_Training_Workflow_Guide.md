# 完整训练流程和启动指南

**适用于**：完全不懂也想开始训练的你

**预计阅读时间**：20 分钟

---

## 第一部分：如何启动训练？

### 最简单的启动方式（一行命令）

```bash
# 进入项目根目录
# 启动训练（Point Foot 机器人）
python scripts/rsl_rl/train.py --task Isaac-Limx-PointFoot-v0 --headless

# 或者使用简写
python scripts/rsl_rl/train.py --task=Isaac-Limx-PointFoot-v0 --headless
```

**参数解释**：
```
--task：指定任务名称
  • Isaac-Limx-PointFoot-v0：Point Foot 机器人
  • Isaac-Limx-SoleFoot-v0：Sole Foot 机器人
  • Isaac-Limx-WheelFoot-v0：Wheel Foot 机器人

--headless：无头模式（不显示可视化界面）
  • 优点：训练更快（节省渲染时间）
  • 缺点：看不到机器人在干什么
  • 推荐：训练时用 --headless，测试时不用
```

### 完整的命令行参数

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Limx-PointFoot-v0 \      # 任务名称
    --headless \                           # 无头模式
    --num_envs 4096 \                      # 并行环境数（默认 4096）
    --max_iterations 1000 \                # 最大训练轮数
    --seed 42 \                            # 随机种子
    --device cuda:0                        # 使用的设备（GPU）

# 恢复训练（从检查点继续）
python scripts/rsl_rl/train.py \
    --task Isaac-Limx-PointFoot-v0 \
    --headless \
    --resume \                             # 恢复训练
    --load_run path/to/run/folder          # 指定运行文件夹
```

### 运行推理（测试已训练模型）

```bash
python scripts/rsl_rl/play.py \
    --task Isaac-Limx-PointFoot-v0 \
    --checkpoint logs/rsl_rl/pointfoot_locomotion/2024-01-01_12-00-00/model_1000.pt \
    --num_envs 16                          # 测试时用少量环境即可
```

---

## 第二部分：训练流程详解

### 整体流程图

```
┌─────────────────────────────────────────────────────┐
│ 阶段 0：准备工作                                    │
│   • 安装依赖（Isaac Sim, Isaac Lab, PyTorch）     │
│   • 克隆项目代码                                    │
│   • 熟悉项目结构                                    │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 1：初始化（第一次运行）                        │
│   1. 加载配置文件                                   │
│      ├─ limx_base_env_cfg.py                       │
│      ├─ pointfoot_cfg.py                           │
│      └─ limx_rsl_rl_ppo_cfg.py                     │
│                                                     │
│   2. 创建环境                                       │
│      ├─ 初始化 Isaac Sim                           │
│      ├─ 加载机器人 USD 模型                        │
│      ├─ 创建 4096 个并行环境                       │
│      └─ 初始化观测、奖励、动作管理器               │
│                                                     │
│   3. 创建神经网络                                   │
│      ├─ Actor 网络（策略）                         │
│      ├─ Critic 网络（价值函数）                    │
│      └─ 随机初始化权重                             │
│                                                     │
│   4. 创建优化器                                     │
│      └─ Adam 优化器（学习率 5e-4）                 │
│                                                     │
│   ⏱️ 耗时：首次运行约 3-5 分钟                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 2：训练循环（重复 1000 次）                    │
│                                                     │
│   ┌─────────────────────────────────────────────┐  │
│   │ 第 N 轮迭代 (Iteration)                     │  │
│   │                                             │  │
│   │ 步骤 1：数据收集 (Rollout)                  │  │
│   │   ├─ 用当前策略在 4096 个环境中运行         │  │
│   │   ├─ 每个环境运行 N 步（通常 24 步）        │  │
│   │   ├─ 收集 (观测, 动作, 奖励) 元组          │  │
│   │   └─ 总数据量：4096 × 24 ≈ 98,304 个样本   │  │
│   │   ⏱️ 耗时：每轮约 5-10 秒                  │  │
│   │                                             │  │
│   │ 步骤 2：优势计算 (GAE)                      │  │
│   │   ├─ 计算每个动作的优势函数                 │  │
│   │   ├─ 优势 = 实际回报 - 价值预测            │  │
│   │   └─ 用于指导策略更新                       │  │
│   │   ⏱️ 耗时：< 1 秒                          │  │
│   │                                             │  │
│   │ 步骤 3：策略更新 (PPO Update)               │  │
│   │   ├─ 分成多个 mini-batch（通常 4 个）      │  │
│   │   ├─ 每个 mini-batch 训练多遍（通常 5 遍） │  │
│   │   ├─ 计算 PPO 损失                         │  │
│   │   │   ├─ Actor 损失（策略梯度）            │  │
│   │   │   ├─ Critic 损失（价值函数）           │  │
│   │   │   └─ 熵损失（鼓励探索）                │  │
│   │   ├─ 反向传播                               │  │
│   │   └─ 更新网络参数                           │  │
│   │   ⏱️ 耗时：每轮约 10-20 秒                 │  │
│   │                                             │  │
│   │ 步骤 4：日志记录                            │  │
│   │   ├─ 平均奖励                               │  │
│   │   ├─ 平均 episode 长度                     │  │
│   │   ├─ 策略损失                               │  │
│   │   ├─ 价值损失                               │  │
│   │   └─ 其他统计信息                           │  │
│   │   ⏱️ 耗时：< 1 秒                          │  │
│   │                                             │  │
│   │ 步骤 5：保存检查点（每 N 轮）               │  │
│   │   ├─ 保存网络权重                           │  │
│   │   ├─ 保存优化器状态                         │  │
│   │   └─ 保存训练统计                           │  │
│   │   ⏱️ 耗时：< 1 秒                          │  │
│   └─────────────────────────────────────────────┘  │
│                                                     │
│   总耗时：每轮约 15-30 秒                          │
│   1000 轮总计：约 4-8 小时                         │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 3：训练完成                                    │
│   • 保存最终模型                                    │
│   • 生成训练报告                                    │
│   • 绘制学习曲线                                    │
└─────────────────────────────────────────────────────┘
```

---

## 第三部分：训练过程中的输出解读

### 终端输出示例

```
============================================
Training: Isaac-Limx-PointFoot-v0
============================================
Initializing environment...
  - Number of environments: 4096
  - Physics timestep: 0.005 s
  - Decimation: 4
  - Control timestep: 0.02 s
  
Loading robot from: .../PF_TRON1A.usd
Initializing observation manager...
  - Policy observations: 59 dimensions
  - Critic observations: 59 dimensions
  
Initializing reward manager...
  - Number of reward terms: 7
  - Total reward weight: 2.95
  
Initializing action manager...
  - Action space: 6 dimensions
  - Action scale: 0.25
  
Creating PPO network...
  - Actor hidden layers: [512, 512]
  - Critic hidden layers: [512, 512]
  - Total parameters: 1.2M
  
Starting training...
============================================

Iteration 1/1000
--------------------------------------------
Episode: 0-10
  Mean reward: -5.23
  Mean episode length: 156 steps
  FPS: 12450
  Time: 15.2s

Policy update:
  Actor loss: 0.234
  Critic loss: 0.189
  Entropy: 2.456
  Learning rate: 5.0e-4

Checkpoint saved: logs/rsl_rl/pointfoot_locomotion/2024-01-01_12-00-00/model_1.pt

Iteration 2/1000
--------------------------------------------
Episode: 10-20
  Mean reward: -3.45
  Mean episode length: 234 steps
  FPS: 13200
  Time: 14.8s

Policy update:
  Actor loss: 0.198
  Critic loss: 0.156
  Entropy: 2.401

... (后续轮次)

Iteration 100/1000
--------------------------------------------
Episode: 990-1000
  Mean reward: 0.82          ← 奖励显著提升！
  Mean episode length: 2450  ← episode 更长（没摔倒）
  FPS: 14500
  Time: 13.5s

Iteration 1000/1000
--------------------------------------------
Training completed!
Total time: 4h 23m 15s
Final checkpoint: logs/.../model_1000.pt
```

### 关键指标解释

```
Mean reward（平均奖励）：
  • 所有环境的平均总奖励
  • 初期：负数（机器人还不会走）
  • 中期：0 附近（开始学会了）
  • 后期：正数（走得很好）
  • 目标：> 0.8

Mean episode length（平均 episode 长度）：
  • 机器人在摔倒前能走多少步
  • 初期：100-300 步（经常摔倒）
  • 中期：500-1000 步（偶尔摔倒）
  • 后期：2000-2500 步（很少摔倒）
  • 最大值：2500 步（时间限制）

FPS（每秒帧数）：
  • 仿真速度
  • GPU 好：15000+
  • GPU 一般：10000-15000
  • GPU 差：< 10000
  • 影响因素：环境数、GPU 型号、系统负载

Actor loss（策略损失）：
  • 策略网络的损失
  • 初期：较大（0.3+）
  • 后期：较小（0.05-0.1）
  • 如果一直很大或突然增大 → 可能有问题

Critic loss（价值损失）：
  • 价值网络的损失
  • 类似 Actor loss
  • 应该逐渐减小

Entropy（熵）：
  • 策略的随机性
  • 初期：高（3.0+）→ 探索多
  • 后期：低（1.0-2.0）→ 确定性强
  • 如果降到接近 0 → 可能过拟合
```

---

## 第四部分：文件结构和输出

### 训练输出的文件结构

```
logs/
└── rsl_rl/
    └── pointfoot_locomotion/              # 任务名称
        └── 2024-01-01_12-00-00/           # 运行时间戳
            ├── config.json                 # 训练配置
            ├── model_50.pt                 # 第 50 轮检查点
            ├── model_100.pt                # 第 100 轮检查点
            ├── model_200.pt                # ...
            ├── model_500.pt
            ├── model_1000.pt               # 最终模型
            │
            ├── summaries/                  # TensorBoard 日志
            │   └── events.out.tfevents...
            │
            └── videos/                     # 录制的视频（如果启用）
                ├── episode_0.mp4
                └── episode_100.mp4
```

### 如何查看训练进度（TensorBoard）

```bash
# 安装 TensorBoard（如果没有）
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir logs/rsl_rl

# 然后在浏览器打开：http://localhost:6006

# 可以看到：
#   • 奖励曲线
#   • 损失曲线
#   • episode 长度
#   • 学习率变化
#   • 等等
```

---

## 第五部分：任务依赖关系详解

### 任务依赖图

```
                    任务 2.2
                (平地速度跟随)
                      │
            ┌─────────┼─────────┐
            ↓                   ↓
        任务 2.3             任务 2.4
    (抗干扰鲁棒性)        (复杂地形适应)
            │                   │
            └─────────┬─────────┘
                      ↓
                任务 2.5（可选）
              (综合表现 + 开源)
```

### 详细依赖关系

#### 任务 2.2 → 任务 2.3

```
任务 2.2 提供：
  ✓ 基础的行走能力
  ✓ 速度控制能力
  ✓ 稳定的步态
  ✓ 已训练的模型（可作为 2.3 的起点）

任务 2.3 需要：
  ✓ 2.2 的所有能力
  ✓ 额外：应对外力干扰的能力

依赖程度：⭐⭐⭐⭐（强依赖）
  • 如果 2.2 没完成好，2.3 会很困难
  • 建议：2.2 达到奖励 > 0.7 再做 2.3

可以跳过吗？
  ✗ 不推荐
  • 直接做 2.3 相当于同时学走路和抗干扰
  • 难度指数增长
  • 训练时间可能 10 倍以上
```

#### 任务 2.2 → 任务 2.4

```
任务 2.2 提供：
  ✓ 基础行走能力
  ✓ 速度控制
  ✓ 平坦地形上的稳定性

任务 2.4 需要：
  ✓ 2.2 的所有能力
  ✓ 额外：地形感知和适应能力
  ✓ 额外：目标导向能力

依赖程度：⭐⭐⭐（中等依赖）
  • 可以在 2.4 中从头训练
  • 但有 2.2 的基础会快很多

可以跳过吗？
  △ 可以但不建议
  • 可以直接训练 2.4，但需要：
    ├─ 使用课程学习（从平地开始）
    ├─ 更长的训练时间（2-3 倍）
    └─ 更多的调试工作
```

#### 任务 2.3 → 任务 2.4

```
任务 2.3 提供：
  ✓ 鲁棒的策略
  ✓ 应对扰动的能力
  ✓ 快速恢复平衡的能力

任务 2.4 需要：
  ✓ 地形适应（类似于"连续的扰动"）
  ✓ 2.3 的鲁棒性会很有帮助

依赖程度：⭐⭐（弱依赖）
  • 2.3 不是 2.4 的必须
  • 但有 2.3 的能力，2.4 会更容易

关系：
  • 地形变化 ≈ 连续的小扰动
  • 2.3 的抗干扰能力在 2.4 中很有用
  • 特别是在地形切换时（如平地→斜坡）
```

### 推荐的完成路径

#### 路径 1：完整路径（最稳妥）⭐⭐⭐⭐⭐

```
第 1 周：任务 2.2（平地速度跟随）
  • 目标：奖励 > 0.8，速度误差 < 0.1 m/s
  • 训练时间：500-1000 轮
  • 预计耗时：2-3 天

第 2 周：任务 2.3（抗干扰）
  • 基于 2.2 的模型继续训练
  • 目标：承受冲量 > 100 N·s，恢复时间 < 2s
  • 训练时间：300-500 轮
  • 预计耗时：1-2 天

第 3 周：任务 2.4（复杂地形）
  • 基于 2.3 的模型继续训练
  • 使用课程学习
  • 目标：通过率 > 80%
  • 训练时间：1000-2000 轮
  • 预计耗时：3-5 天

总计：6-10 天
优点：稳扎稳打，每个任务都做好
缺点：时间较长
```

#### 路径 2：快速路径（时间紧）⭐⭐⭐

```
第 1 周：任务 2.2（平地速度跟随）
  • 快速训练到基本能走（奖励 > 0.5）
  • 训练时间：200-300 轮
  • 预计耗时：1 天

第 2 周：任务 2.4（直接跳到复杂地形）
  • 从 2.2 的模型开始
  • 同时训练地形适应和鲁棒性
  • 使用课程学习（从简单地形开始）
  • 训练时间：1500-2000 轮
  • 预计耗时：4-5 天

跳过：任务 2.3

总计：5-6 天
优点：节省时间
缺点：难度较大，可能需要更多调试
```

#### 路径 3：极速路径（极限赶工）⭐⭐

```
直接做任务 2.4，但使用激进的课程学习：

阶段 1：平地（模拟任务 2.2）
  • 100% 平地，训练 100 轮
  • 学会基础行走

阶段 2：平地 + 小扰动（模拟任务 2.3）
  • 80% 平地 + 20% 波形
  • 加入小的外力扰动
  • 训练 100 轮

阶段 3：混合地形
  • 逐步增加地形复杂度
  • 训练 800-1000 轮

总计：3-4 天
优点：最快
缺点：风险最大，可能卡住
推荐：只在时间极度紧张时使用
```

---

## 第六部分：常见启动问题

### ❌ 问题 1：找不到任务

```
错误信息：
  ValueError: Task 'Isaac-Limx-PointFoot-v0' not found

原因：
  • 任务名称拼写错误
  • 环境未正确注册

解决方案：
  # 检查可用的任务
  python scripts/rsl_rl/train.py --help
  
  # 或者查看注册文件
  # exts/bipedal_locomotion/bipedal_locomotion/__init__.py
```

### ❌ 问题 2：CUDA 内存不足

```
错误信息：
  RuntimeError: CUDA out of memory

原因：
  • GPU 内存不够
  • 环境数太多

解决方案：
  # 减少并行环境数
  python scripts/rsl_rl/train.py \
      --task Isaac-Limx-PointFoot-v0 \
      --headless \
      --num_envs 2048  # 从 4096 减少到 2048
  
  # 或者 1024
  --num_envs 1024
```

### ❌ 问题 3：训练速度很慢

```
症状：
  • FPS < 5000
  • 每轮需要 > 1 分钟

原因：
  • 没有使用 GPU
  • GPU 驱动问题
  • 环境数设置不当

解决方案：
  # 检查 GPU 是否可用
  python -c "import torch; print(torch.cuda.is_available())"
  
  # 如果返回 False，检查：
  # 1. CUDA 是否安装
  # 2. PyTorch 是否支持 CUDA
  # 3. 驱动是否最新
  
  # 如果 GPU 不可用，减少环境数
  --num_envs 256  # 在 CPU 上运行
```

---

## 总结：快速启动清单

### 第一次运行（15 分钟设置）

```
☐ 确认环境已安装
  - Isaac Sim 4.5.0
  - Isaac Lab 2.1.0
  - Python 3.10+
  - PyTorch with CUDA

☐ 克隆项目代码
  git clone <repository-url>

☐ 进入项目目录
  cd limxtron1lab-main

☐ 启动第一次训练（测试）
  python scripts/rsl_rl/train.py \
      --task Isaac-Limx-PointFoot-v0 \
      --headless \
      --max_iterations 10  # 只训练 10 轮测试

☐ 观察输出
  - 是否正常启动
  - FPS 是否合理（> 10000）
  - 奖励是否有变化
```

### 正式训练（每个任务）

```
☐ 任务 2.2（平地速度跟随）
  python scripts/rsl_rl/train.py \
      --task Isaac-Limx-PointFoot-v0 \
      --headless \
      --max_iterations 1000

☐ 任务 2.3（抗干扰）
  1. 修改配置（添加外力干扰）
  2. 从 2.2 的模型继续训练（可选）
  3. python scripts/rsl_rl/train.py ...

☐ 任务 2.4（复杂地形）
  1. 修改地形配置
  2. 添加课程学习
  3. python scripts/rsl_rl/train.py ...
```

### 测试和评估

```
☐ 运行推理
  python scripts/rsl_rl/play.py \
      --task Isaac-Limx-PointFoot-v0 \
      --checkpoint logs/.../model_1000.pt

☐ 记录性能指标
  - 速度追踪误差
  - 姿态稳定性
  - 存活率
  - （2.3）最大冲量
  - （2.4）通过率

☐ 保存最佳模型
  cp logs/.../model_best.pt final_models/
```

---

现在你完全知道如何启动训练了！🚀

关键命令回顾：
```bash
# 训练
python scripts/rsl_rl/train.py --task Isaac-Limx-PointFoot-v0 --headless

# 测试
python scripts/rsl_rl/play.py --task Isaac-Limx-PointFoot-v0 --checkpoint logs/.../model.pt

# 查看进度
tensorboard --logdir logs/rsl_rl
```

