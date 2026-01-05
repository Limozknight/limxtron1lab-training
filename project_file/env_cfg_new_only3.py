import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

from bipedal_locomotion.tasks.locomotion import mdp

import torch
from isaaclab.managers import SceneEntityCfg as SceneEntityCfgType


##################
# 场景定义 / Scene Definition
##################


@configclass
class PFSceneCfg(InteractiveSceneCfg):
    """测试场景配置类 / Configuration for the test scene"""

    # 地形配置 / Terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在场景中的路径 / Terrain path in scene
        terrain_type="plane",  # 地形类型：平面 / Terrain type: plane
        terrain_generator=None,  # 不使用地形生成器 / No terrain generator used
        max_init_terrain_level=0,  # 最大初始地形难度等级 / Maximum initial terrain difficulty level
        collision_group=-1,  # 碰撞组ID / Collision group ID

        # 物理材质属性 / Physics material properties
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦力结合模式：乘法 / Friction combine mode: multiply
            restitution_combine_mode="multiply",  # 恢复系数结合模式：乘法 / Restitution combine mode: multiply
            static_friction=1.0,  # 静摩擦系数 / Static friction coefficient
            dynamic_friction=1.0,  # 动摩擦系数 / Dynamic friction coefficient
            restitution=1.0,  # 恢复系数 / Restitution coefficient
        ),

        # 视觉材质配置 / Visual material configuration
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                     + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # 大理石纹理材质路径 / Marble texture material path
            project_uvw=True,  # 启用UV投影 / Enable UV projection
            texture_scale=(0.25, 0.25),  # 纹理缩放比例 / Texture scaling factor
        ),
        debug_vis=False,  # 不显示调试可视化 / Don't show debug visualization
    )

    # 天空光照配置 / Sky lighting configuration
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # pointfoot robot
    robot: ArticulationCfg = MISSING

    # 高度扫描传感器 (将在子类中定义) / Height scanner sensor (to be defined in subclasses)
    height_scanner: RayCasterCfg = MISSING

    # 接触力传感器配置 / Contact force sensor configuration
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 传感器安装路径 / Sensor attachment path
        history_length=4,  # 历史数据长度 / History data length
        track_air_time=True,  # 跟踪空中时间 / Track air time
        update_period=0.0,  # 更新周期 (0表示每帧更新) / Update period (0 means every frame)
    )


##############
# MDP设置 / MDP Settings
##############


@configclass
class CommandCfg(BaseCommandsCfg):
    """命令配置类 - 专注于抗干扰测试的速度命令 / Command configuration class - focus on velocity commands for disturbance rejection test"""
    """后初始化配置 / Post-initialization configuration"""

    def __post_init__(self):
        self.base_velocity.asset_name = "robot"  # 关联的机器人资产名称 / Associated robot asset name
        self.base_velocity.heading_command = True  # 启用航向命令 / Enable heading commands
        self.base_velocity.debug_vis = True  # 启用调试可视化 / Enable debug visualization
        self.base_velocity.heading_control_stiffness = 1.0  # 航向控制刚度 / Heading control stiffness
        self.base_velocity.resampling_time_range = (10.0, 10.0)  # 延长命令重采样时间，减少变化频率
        self.base_velocity.rel_standing_envs = 0.1  # 减少站立环境比例
        self.base_velocity.rel_heading_envs = 0.0  # 航向环境比例 / Heading environments ratio
        # 速度命令范围设置 - 专注于低速稳定性测试 / Velocity command ranges - focus on low-speed stability test
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),  # 减小前进速度范围，专注于稳定性
            lin_vel_y=(-0.3, 0.3),  # 减小横向速度范围
            ang_vel_z=(-0.3, 0.3),  # 减小转向角速度范围
            heading=(-math.pi, math.pi)  # 航向角范围 [rad]
        )

        # 移除步态命令配置，专注于抗干扰
        self.gait_command = None


@configclass
class ActionsCfg:
    """动作规范配置类 / Action specifications configuration class"""

    # 关节位置动作配置 / Joint position action configuration
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",  # 目标资产名称 / Target asset name
        # 控制的关节名称列表 / List of controlled joint names
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint",
                     "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,  # 动作缩放因子 / Action scaling factor
        use_default_offset=True,  # 使用默认偏移量 / Use default offset
    )


@configclass
class ObservarionsCfg:
    """观测规范配置类 - 专注于抗干扰测试 / Observation specifications configuration class - focus on disturbance rejection test"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组配置 - 抗干扰专用 / Policy network observation group configuration - for disturbance rejection"""

        # 机器人基座测量 / Robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,  # 基座线速度
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.5,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,  # 基座角速度函数
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 投影重力函数
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 机器人关节测量 / Robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # 上一步动作 / Last action
        last_action = ObsTerm(func=mdp.last_action)

        # ============= 2.3 关键：抗干扰相关观测 =============
        # 姿态稳定性观测（抗干扰关键指标）
        orientation_stability = ObsTerm(
            func=mdp.base_orientation_stability_metric,  # 使用自定义的姿态稳定性函数
            params={"window_size": 5},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(0.0, 1.0),
            scale=5.0,
        )

        # 恢复进度观测
        recovery_progress = ObsTerm(
            func=mdp.disturbance_recovery_progress,  # 使用自定义的恢复进度函数
            params={},
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(0.0, 1.0),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = True  # 启用观测损坏
            self.concatenate_terms = True  # 连接所有观测项

    @configclass
    class HistoryObsCfg(ObsGroup):
        """历史观测组配置 - 用于存储观测历史 / History observation group - for storing observation history"""

        # robot base measurements
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0),
                               scale=0.5, )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0),
                               scale=0.25, )
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),
                               clip=(-100.0, 100.0), scale=1.0, )

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0),
                            scale=1.0, )
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0),
                            scale=0.05, )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # 抗干扰相关观测
        orientation_stability = ObsTerm(func=mdp.base_orientation_stability_metric, params={"window_size": 5})
        recovery_progress = ObsTerm(func=mdp.disturbance_recovery_progress, params={})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 11  # 历史长度为11步
            self.flatten_history_dim = False  # 不展平历史维度

    @configclass
    class CriticCfg(ObsGroup):
        """评价网络观测组配置 - 包含特权信息，专注于抗干扰 / Critic network observation group - includes privileged information, focus on disturbance rejection"""

        # 策略观测 (与智能体相同)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        last_action = ObsTerm(func=mdp.last_action)

        # 抗干扰相关观测
        orientation_stability = ObsTerm(func=mdp.base_orientation_stability_metric, params={"window_size": 5})
        recovery_progress = ObsTerm(func=mdp.disturbance_recovery_progress, params={})

        heights = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

        # 特权观测 (仅评价网络可见) - 专注于物理参数
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            },
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)

        # 外部干扰状态观测
        external_force_state = ObsTerm(
            func=mdp.base_lin_vel,  # 使用基座线速度作为干扰状态代理
            params={}
        )

        def __post_init__(self):
            self.enable_corruption = False  # 不对特权信息添加噪声
            self.concatenate_terms = True  # 连接所有观测项

    @configclass
    class CommandsObsCfg(ObsGroup):
        """命令观测配置 / Commands observation configuration"""
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventsCfg:
    """事件配置类 - 专注于抗干扰随机化事件 / Events configuration class - focus on disturbance randomization events"""
    # 即域随机化 / i.e. domain randomization

    # 启动时事件 - 增强物理随机化以提高鲁棒性
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-2.0, 5.0),  # 扩大质量分布范围
            "operation": "add",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.5, 1.5),  # 扩大质量缩放范围
            "operation": "scale",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.5, 1.5),  # 扩大惯量分布范围
            "operation": "scale",
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),  # 扩大静摩擦系数范围
            "dynamic_friction_range": (0.5, 1.2),  # 扩大动摩擦系数范围
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (20, 60),  # 扩大刚度分布范围
            "damping_distribution_params": (1.0, 4.0),  # 扩大阻尼分布范围
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # 扩大重心偏移范围
            "com_distribution_params": ((-0.1, 0.1), (-0.08, 0.08), (-0.08, 0.08)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # 重置时事件
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 姿态范围
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 速度范围
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # ============= 2.3 核心：抗干扰推力事件 =============
    # 主要干扰事件 - 随机施加推力
    push_robot_main = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(1.5, 3.0),  # 每1.5-3秒施加一次推力
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # 推力范围 [N] - 专注于水平方向推力
            "force_range": {
                "x": (-1000.0, 1000.0),  # 前后方向
                "y": (-800.0, 800.0),  # 左右方向
                "z": (-300.0, 300.0),  # 垂直方向（较小）
            },
            # 力矩范围 [N⋅m]
            "torque_range": {
                "x": (-150.0, 150.0),  # 俯仰力矩
                "y": (-150.0, 150.0),  # 滚转力矩
                "z": (-80.0, 80.0)  # 偏航力矩
            },
            "probability": 0.5,  # 增加发生概率
        },
        is_global_time=False,
        min_step_count_between_reset=5,  # 重置后至少5步才施加推力
    )

    # 次要干扰事件 - 更频繁但较小的推力
    push_robot_minor = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.5, 1.5),  # 每0.5-1.5秒施加一次推力
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # 较小的推力范围
            "force_range": {
                "x": (-300.0, 300.0),
                "y": (-200.0, 200.0),
                "z": (-100.0, 100.0),
            },
            # 较小的力矩范围
            "torque_range": {
                "x": (-50.0, 50.0),
                "y": (-50.0, 50.0),
                "z": (-30.0, 30.0)
            },
            "probability": 0.3,  # 中等发生概率
        },
        is_global_time=False,
        min_step_count_between_reset=3,
    )


@configclass
class RewardsCfg:
    """奖励项配置类 - 专注于抗干扰鲁棒性 / Reward terms configuration class - focus on disturbance rejection robustness"""

    # 基本生存奖励
    keep_balance = RewTerm(
        func=mdp.stay_alive,
        weight=2.0  # 增加权重，鼓励存活
    )

    # ============= 2.3 核心：抗干扰恢复能力奖励 =============
    rew_disturbance_recovery = RewTerm(
        func=mdp.disturbance_recovery_reward,
        weight=10.0,  # 增加权重，强调恢复能力
        params={
            "recovery_time_window": 3.0,  # 延长恢复时间窗口
            "velocity_tracking_tolerance": 0.3,  # 放宽速度追踪容忍度
            "orientation_tolerance": 0.4,  # 放宽姿态容忍度
            "max_reward": 1.0
        }
    )

    # ============= 2.3 核心：抗干扰鲁棒性奖励 =============
    rew_disturbance_robustness = RewTerm(
        func=mdp.disturbance_robustness_reward,
        weight=15.0,  # 显著增加权重，强调鲁棒性
        params={
            "stability_weight": 1.0
        }
    )

    # 基础速度追踪奖励（保持基本运动能力）
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,  # 降低权重，专注于抗干扰
        params={"command_name": "base_velocity", "std": math.sqrt(0.3)}  # 增加std，降低精度要求
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,  # 降低权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.3)}
    )

    # 姿态稳定性奖励 - 抗干扰关键
    rew_base_stability = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-8.0,  # 适当惩罚姿态不稳定
        params={}
    )

    # 调节相关奖励
    pen_base_height = RewTerm(
        func=mdp.base_com_height,
        params={"target_height": 0.78},
        weight=-10.0,  # 降低惩罚权重
    )

    # 关节相关惩罚
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)  # 降低惩罚
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)  # 降低惩罚
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00005)  # 降低惩罚
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-07)  # 降低惩罚
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # 降低惩罚
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)  # 降低惩罚
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-04)  # 降低惩罚
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2e-04)  # 降低惩罚

    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,  # 降低惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,
        },
    )

    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,
        weight=-0.02  # 降低惩罚
    )
    pen_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0  # 降低惩罚
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-50,  # 降低惩罚
        params={
            "min_feet_distance": 0.115,
            "feet_links_name": ["foot_[RL]_Link"]
        }
    )

    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,  # 降低惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "base_height_target": 0.65,
            "foot_radius": 0.03
        },
    )

    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,
        weight=-0.2,  # 降低惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_[RL]_Link"]),
            "foot_radius": 0.03,
            "about_landing_threshold": 0.08
        },
    )


@configclass
class TerminationsCfg:
    """终止条件配置类 - 专注于抗干扰测试 / Termination conditions configuration class - focus on disturbance rejection test"""

    # 时间超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 基座接触终止 (机器人倒下)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 1.0
        },
    )

    # ============= 2.3 修改：剧烈干扰后的终止条件 =============
    excessive_disturbance = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 0.5  # 降低阈值，更容易触发
        }
    )

    # 过度倾斜终止
    excessive_tilt = DoneTerm(
        func=mdp.flat_orientation_l2,
        params={
            "threshold": 0.8  # Roll/Pitch平方和超过0.8时终止
        }
    )


# ============= 平面地形抗干扰课程学习函数 =============
def flat_terrain_disturbance_levels(
        env,
        env_ids: torch.Tensor | None = None,
        asset_cfg: SceneEntityCfgType = SceneEntityCfgType("robot"),
        step_size: float = 0.15,
        max_level: int = 5,
        **kwargs
) -> int:
    """平面地形抗干扰课程学习函数 - 基于抗干扰能力渐进增加难度"""

    # 对于抗干扰测试，可以从较低难度开始
    level = 1

    return level


@configclass
class CurriculumCfg:
    """课程学习配置类 - 专注于抗干扰 / Curriculum learning configuration class - focus on disturbance rejection"""

    # ============= 2.3 核心：抗干扰强度课程 =============
    disturbance_intensity = CurrTerm(
        func=flat_terrain_disturbance_levels,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "step_size": 0.2,  # 增加步长，更快提高难度
            "max_level": 5,
        }
    )

    # 抗干扰频率课程
    disturbance_frequency = CurrTerm(
        func=flat_terrain_disturbance_levels,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "step_size": 0.1,
            "max_level": 10,
        }
    )


########################
# 环境定义 / Environment Definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """测试环境配置类 - 专注于抗干扰鲁棒性测试 / Test environment configuration class - focus on disturbance rejection robustness test"""

    # 场景设置
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基本设置
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    # MDP设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化配置"""
        self.decimation = 4
        self.episode_length_s = 25.0  # 延长episode长度，提供更多抗干扰测试时间
        self.sim.render_interval = 2 * self.decimation

        # 仿真设置
        self.sim.dt = 0.005
        self.seed = 42

        # 更新传感器更新周期
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt