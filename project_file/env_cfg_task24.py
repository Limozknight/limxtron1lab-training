import math

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import BLIND_ROUGH_TERRAINS_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_pointfoot_env_cfg import PFBaseEnvCfg


@configclass
class PFTerrainTraversalEnvCfg(PFBaseEnvCfg):
    """任务2.4用：混合粗糙/坡度地形遍历配置 / Task 2.4 terrain traversal configuration."""

    def __post_init__(self):
        super().__post_init__()

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

        self.rewards.rew_lin_vel_xy_precise.weight = 6.0
        self.rewards.rew_ang_vel_z_precise.weight = 3.5
        self.rewards.pen_base_height.func = mdp.base_height_rough_l2
        self.rewards.pen_base_height.weight = -8.0
        self.rewards.pen_base_height.params = {
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.pen_flat_orientation.weight = -3.0
        self.rewards.pen_feet_regulation.weight = -0.2
        self.rewards.foot_landing_vel.weight = -1.0
        self.rewards.pen_undesired_contacts.weight = -1.0
        self.rewards.pen_action_smoothness.weight = -0.08

        self.events.push_robot = None


@configclass
class PFTerrainTraversalEnvCfg_PLAY(PFTerrainTraversalEnvCfg):
    """任务2.4测试版：较少环境，关闭扰动 / Play config for task 2.4."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
