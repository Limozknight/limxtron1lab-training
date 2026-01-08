import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm



######################
# 双足机器人基础环境 / Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFEnvCfg):
    """双足机器人基础环境配置 - 所有变体的共同基础 / Base environment configuration for pointfoot robot - common foundation for all variants"""
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }
        # 调整基座质量随机化参数 / Adjust base mass randomization parameters
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # 设置基座接触终止条件 / Set base contact termination condition
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # 更新视口相机设置 / Update viewport camera settings
        self.viewer.origin_type = "env"  # 相机跟随环境 / Camera follows environment


@configclass
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
    """双足机器人基础测试环境配置 - 用于策略评估 / Base play environment configuration - for policy evaluation"""
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# 双足机器人盲视平地环境 / Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# 双足机器人盲视粗糙环境 / Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class PFTerrainTraversalEnvCfg(PFBaseEnvCfg):
    """任务2.4用：混合粗糙/坡度地形遍历配置 / Task 2.4 terrain traversal configuration."""

    def __post_init__(self):
        super().__post_init__()

        # 更粗的环境间距，减少相邻干扰 / Increase spacing to reduce inter-env collisions
        self.scene.env_spacing = 3.0
        self.scene.num_envs = 2048

        # 使用地形生成器混合粗糙地形 / Enable mixed rough terrains via generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
        # 由生成器管理难度，关闭平地课程条目 / Let generator handle difficulty; drop flat curriculum term
        self.curriculum.terrain_levels = None

        # 开启高度射线用于粗糙地形高度估计 / Ray-based height scanner for uneven terrain
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 把高度观测送入策略与价值网络 / Feed height scans to policy/critic
        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # 缩放因子（降低以稳定训练）/ Scaling factor (reduced for stability)
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(0.0, 10.0),
        )

        # 历史观测加入高度扫描，保证历史维度与策略维度一致
        self.observations.obsHistory.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # 缩放因子（降低以稳定训练）/ Scaling factor (reduced for stability)
        )

        # 奖励针对粗糙地形的调整 / Reward tweaks for terrain traversal
        self.rewards.rew_lin_vel_xy_precise.weight = 6.0
        self.rewards.rew_ang_vel_z_precise.weight = 3.5

        # 高度惩罚使用射线相对高度 / Use terrain-aware height penalty
        self.rewards.pen_base_height.func = mdp.base_height_rough_l2
        self.rewards.pen_base_height.weight = -8.0
        self.rewards.pen_base_height.params = {
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        }

        # 提高足部与姿态相关约束 / Stronger foot and posture regulation
        self.rewards.pen_flat_orientation.weight = -3.0
        self.rewards.pen_feet_regulation.weight = -0.2
        self.rewards.foot_landing_vel.weight = -1.0
        self.rewards.pen_undesired_contacts.weight = -1.0
        self.rewards.pen_action_smoothness.weight = -0.08

        # 粗糙地形下保持轻扰动：关闭外力事件 / Disable random pushes for stability on rough terrain
        self.events.push_robot = None


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


@configclass
class PFTerrainTraversalEnvCfg_PLAY(PFTerrainTraversalEnvCfg):
    """任务2.4测试版：较少环境，关闭扰动 / Play config for task 2.4."""

    def __post_init__(self):
        super().__post_init__()

        # 更小并行数用于评估 / Fewer envs for evaluation
        self.scene.num_envs = 64

        # 限制速度命令范围仅向前 / Restrict velocity to forward only
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)  # 只向前 / Forward only
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # 最小横向 / Minimal lateral
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)  # 小转向 / Small turning

        # 禁用观测腐蚀与随机化 / Disable corruption/random pushes
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


#############################
# 任务2.4 V2优化版：降低扭矩与增强姿态稳定性
# Task 2.4 V2: Reduced Torque & Enhanced Stability
#############################

@configclass
class PFTerrainTraversalEnvCfgV2(PFBaseEnvCfg):
    """任务2.4 V2优化版：降低扭矩与增强姿态稳定性 / Task 2.4 V2: optimized for lower torque and better stability."""

    def __post_init__(self):
        super().__post_init__()

        # ========== 地形与传感器配置（与 V1 相同）==========
        self.scene.env_spacing = 3.0
        self.scene.num_envs = 2048
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
        self.curriculum.terrain_levels = None

        # 高度扫描传感器 / Height scanner
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 高度观测 / Height observations
        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # 缩放因子（降低以稳定训练）/ Scaling factor (reduced for stability)
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(0.0, 10.0),
        )

        # 历史观测加入高度扫描，保证历史维度与策略维度一致
        self.observations.obsHistory.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # 缩放因子（降低以稳定训练）/ Scaling factor (reduced for stability)
        )

        # ========== V2 修改 1: 降低动作尺度 ==========
        self.actions.joint_pos.scale = 0.20  # V1: 0.25

        # ========== V2 修改 2-6: 奖励权重调整 ==========
        # 速度跟踪（略降，给扭矩/姿态优化让路）/ Velocity tracking (slightly reduced)
        self.rewards.rew_lin_vel_xy_precise.weight = 5.5   # V1: 6.0
        self.rewards.rew_ang_vel_z_precise.weight = 3.2    # V1: 3.5

        # 姿态稳定（大幅增加）/ Base stability (significantly increased)
        self.rewards.rew_base_stability.weight = 2.0       # V1: 1.0

        # 高度惩罚（保持）/ Height penalty (maintained)
        self.rewards.pen_base_height.func = mdp.base_height_rough_l2
        self.rewards.pen_base_height.weight = -8.0
        self.rewards.pen_base_height.params = {
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        }

        # 姿态约束（保持）/ Posture constraints (maintained)
        self.rewards.pen_flat_orientation.weight = -3.0
        self.rewards.pen_feet_regulation.weight = -0.2
        self.rewards.foot_landing_vel.weight = -1.0
        self.rewards.pen_undesired_contacts.weight = -1.0

        # **V2 关键修改：扭矩与动作平滑** / Key V2 changes: torque and smoothness
        self.rewards.pen_joint_torque.weight = -0.025      # V1: -0.01（增加 2.5 倍）
        self.rewards.pen_action_smoothness.weight = -0.12  # V1: -0.08（增加 50%）
        
        # **V2 关键修改：俯仰/滚转角速度** / Key V2 change: pitch/roll angular velocity
        self.rewards.pen_ang_vel_xy.weight = -0.10         # V1: -0.05（翻倍）

        # 禁用外力扰动 / Disable random pushes
        self.events.push_robot = None


@configclass
class PFTerrainTraversalEnvCfgV2_PLAY(PFTerrainTraversalEnvCfgV2):
    """任务2.4 V2测试版 / Task 2.4 V2 play config."""

    def __post_init__(self):
        super().__post_init__()

        # 更小并行数用于评估 / Fewer envs for evaluation
        self.scene.num_envs = 64

        # 禁用观测腐蚀与随机化 / Disable corruption/random pushes
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


##############################
# 任务2.4：台阶遍历环境 / Task 2.4: Stair Traversal Environment
##############################

@configclass
class PFStairTraversalEnvCfg(PFBaseEnvCfg):
    """Task 2.4 stair traversal training configuration with height scanner for terrain adaptation"""

    def __post_init__(self):
        super().__post_init__()

        # Environment configuration
        self.scene.env_spacing = 3.0
        self.scene.num_envs = 2048

        # Use stairs terrain generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.curriculum.terrain_levels = None

        # Height scanner for stair detection
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Height observations for policy and critic
        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # Scaling factor (reduced for stability)
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(0.0, 10.0),
        )

        # Height observations in history for temporal context
        self.observations.obsHistory.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,  # Scaling factor (reduced for stability)
        )

        # Adjust velocity command ranges for stairs (more conservative)
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # Reward adjustments for stair climbing
        self.rewards.rew_lin_vel_xy_precise.weight = 5.0
        self.rewards.rew_ang_vel_z_precise.weight = 2.5

        # Height penalty with terrain awareness
        self.rewards.pen_base_height.func = mdp.base_height_rough_l2
        self.rewards.pen_base_height.weight = -10.0
        self.rewards.pen_base_height.params = {
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        }

        # Increased posture constraints for stairs
        self.rewards.pen_flat_orientation.weight = -4.0
        self.rewards.pen_feet_regulation.weight = -0.3
        self.rewards.foot_landing_vel.weight = -1.5
        self.rewards.pen_undesired_contacts.weight = -1.5
        self.rewards.pen_action_smoothness.weight = -0.15

        # Disable random pushes for stable stair climbing
        self.events.push_robot = None


@configclass
class PFStairTraversalEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    """Task 2.4 stair traversal testing configuration with reduced environments"""

    def __post_init__(self):
        super().__post_init__()

        # Environment configuration
        self.scene.env_spacing = 3.0
        self.scene.num_envs = 64

        # Use stairs terrain generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG
        self.curriculum.terrain_levels = None

        # Height scanner for stair detection
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Height observations
        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(0.0, 10.0),
        )

        # History observations
        self.observations.obsHistory.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 10.0),
            scale=0.1,
        )

        # Restrict velocity to forward only for testing / 限制速度仅向前用于测试
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)  # Forward only / 仅向前
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # Minimal lateral / 最小横向
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)  # Small turning / 小转向

        # Disable randomization for evaluation
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


##############################
# 双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    """盲视楼梯环境配置 - 专门训练爬楼梯能力 / Blind stairs environment configuration - specialized for stair climbing training"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯训练环境 / Post-initialization - configure stairs training environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)      # 前进速度：0.5-1.0 m/s / Forward velocity: 0.5-1.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     # 横向速度：0（仅直行）/ Lateral velocity: 0 (straight only)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)  # 转向：±30度 / Turning: ±30 degrees

        # 调整奖励权重以适应楼梯爬升 / Adjust reward weights for stair climbing
        self.rewards.rew_lin_vel_xy.weight = 2.0          # 增加线速度跟踪奖励 / Increase linear velocity tracking reward
        self.rewards.rew_ang_vel_z.weight = 1.5           # 增加角速度跟踪奖励 / Increase angular velocity tracking reward
        self.rewards.pen_lin_vel_z.weight = -1.0          # 增加Z方向速度惩罚 / Increase Z velocity penalty
        self.rewards.pen_ang_vel_xy.weight = -0.05        # XY角速度惩罚 / XY angular velocity penalty
        self.rewards.pen_action_rate.weight = -0.01       # 动作变化率惩罚 / Action rate penalty
        self.rewards.pen_flat_orientation.weight = -2.5   # 姿态保持惩罚 / Orientation keeping penalty
        self.rewards.pen_undesired_contacts.weight = -1.0 # 不期望接触惩罚 / Undesired contact penalty

        # 设置楼梯地形 / Set up stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    """盲视楼梯测试环境配置 / Blind stairs play environment configuration"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯测试环境 / Post-initialization - configure stairs testing environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 设置测试专用的速度命令 / Set testing-specific velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 固定前进速度范围 / Fixed forward velocity range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)   # 无横向移动 / No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)   # 无转向 / No turning

        # 固定重置姿态（无偏航角变化）/ Fixed reset pose (no yaw variation)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # 设置测试楼梯地形 / Set up testing stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # 设置中等难度的楼梯测试环境 / Set medium difficulty stairs testing environment
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# 带高度扫描的双足机器人楼梯环境 / Pointfoot Stairs Environment with Height Scanning
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# Task 2.5: 双足跳（Pronk）环境配置 / Pronk Environment
#############################

@configclass
class PFPronkEnvCfg(PFBlindFlatEnvCfg):
    """双足跳跃环境配置 / Pronk environment configuration.
    
    基于平地环境，修改奖励函数以鼓励跳跃。
    Based on flat terrain environment, modifies rewards to encourage jumping.
    """
    def __post_init__(self):
        super().__post_init__()
        
        # 1. 修改命令范围：双足跳通常不需要大范围的水平移动，或者只是直线跳
        # 这里我们限制为主要是X方向的移动，Y方向和旋转设为0
        self.commands.ranges.base_velocity.ranges = {
            "lin_vel_x": (0.0, 1.0),   # 允许向前跳 / Allow forward jump
            "lin_vel_y": (0.0, 0.0),   # 禁止侧向移动 / No lateral movement
            "ang_vel_z": (0.0, 0.0),   # 禁止旋转 / No rotation
            "heading": (0.0, 0.0),
        }
        
        # 2. 调整奖励函数 / Adjust rewards
        # 移除/禁用不利于跳跃的平稳行走奖励
        self.rewards.rew_lin_vel_xy_precise = None
        self.rewards.rew_ang_vel_z_precise = None
        self.rewards.no_fly = None     # 必须移除！否则腾空会被惩罚 / Must remove!
        self.rewards.stand_still = None
        # self.rewards.feet_air_time = None # 如果有这个的话也要移除
        
        # 添加 Pronk 专属奖励
        # A. 强制双脚同步 (权重很大)
        self.rewards.feet_sync = RewTerm(
            func=mdp.feet_synchronization,
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link")}
        )
        
        # B. 鼓励双脚同时腾空 (权重很大)
        self.rewards.pronk_air_time = RewTerm(
            func=mdp.pronk_air_time,
            weight=5.0, # 给予很大的奖励鼓励起飞
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link")}
        )
        
        # C. 简单的向上速度奖励 (辅助起跳)
        self.rewards.jump_vel = RewTerm(
            func=mdp.jump_vertical_velocity,
            weight=0.5
        )
        
        # D. 保持一定的X方向速度 (如果想让它边跳边走)
        self.rewards.track_lin_vel_x = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5}
        )
        
        # E. 姿态稳定性：对于跳跃，允许 Pitch 震荡，但 Roll 应该要小
        self.rewards.orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-0.5, # 较小的负权重
        )

@configclass
class PFPronkEnvCfg_PLAY(PFPronkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 测试时的配置 / Play configuration
        self.commands.ranges.base_velocity.ranges = {
            "lin_vel_x": (0.5, 0.5),   # 固定速度跳
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "heading": (0.0, 0.0),
        }
