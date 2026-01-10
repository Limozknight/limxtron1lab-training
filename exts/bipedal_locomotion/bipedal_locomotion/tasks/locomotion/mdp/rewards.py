"""此子模块包含可用于LimX Point Foot运动任务的奖励函数 / This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

这些函数可以传递给:class:`isaaclab.managers.RewardTermCfg`对象来指定奖励函数及其参数。
The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional, Dict

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """保持存活奖励 - 给予机器人基本的存在奖励 / Reward for staying alive - gives robot basic existence reward."""
    return torch.ones(env.num_envs, device=env.device)


def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """惩罚高足部着陆速度 - 鼓励轻柔着陆 / Penalize high foot landing velocities - encourages soft landing"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取足部Z方向速度 / Get foot Z-direction velocities
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    # 检测接触状态 / Detect contact state
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    # 计算足部高度（相对于地面）/ Calculate foot height (relative to ground)
    foot_heights = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: 改为相对于地形垂直投影的高度 / TODO: change to height relative to terrain vertical projection

    # 检测即将着陆状态：低高度 + 无接触 + 下降速度 / Detect about-to-land state: low height + no contact + downward velocity
    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)

    # 计算着陆速度惩罚 / Calculate landing velocity penalty
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward


def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L1核惩罚关节功率 - 鼓励能效 / Penalize joint powers using L1-kernel - encourages energy efficiency"""
    # 提取使用的数量（启用类型提示）/ Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # 功率 = 力矩 × 角速度 / Power = Torque × Angular Velocity
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str] = ["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0, ) -> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:, feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward


def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                          asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / std ** 2)
    return reward


def leg_symmetry(env: ManagerBasedRLEnv,
                 std: float,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std ** 2)


def same_feet_x_position(env: ManagerBasedRLEnv,
                         asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance


def keep_ankle_pitch_zero_in_air(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
        force_threshold: float = 2.0,
        pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.

    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, 3]) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, 7]) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)


def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
        env, lin_threshold: float = 0.05, ang_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


def feet_regulation(env: ManagerBasedRLEnv,
                    asset_cfg: SceneEntityCfg,
                    foot_radius: float,
                    base_height_target: float,
                    ) -> torch.Tensor:
    """足部调节奖励 - 惩罚足部不当运动 / Foot regulation reward - penalizes improper foot movement"""
    asset: RigidObject = env.scene[asset_cfg.name]

    # 计算足部高度 / Calculate foot height
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: 改为相对于地形垂直投影的高度 / TODO: change to height relative to terrain vertical projection

    # 获取足部XY方向速度 / Get foot XY-direction velocities
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # 高度缩放：足部越低，惩罚权重越大 / Height scaling: lower foot height, higher penalty weight
    height_scale = torch.exp(-feet_height / base_height_target)

    # 计算速度平方惩罚，按高度加权 / Calculate velocity squared penalty, weighted by height
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward


def base_height_rough_l2(
        env: ManagerBasedRLEnv,
        target_height: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(height.mean(dim=1) - target_height)


def base_com_height(
        env: ManagerBasedRLEnv,
        target_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """惩罚基座高度偏离目标 - 维持期望的站立高度 / Penalize base height deviation from target - maintain desired standing height.

    注意：对于平坦地形，目标高度在世界坐标系中。对于粗糙地形，
    传感器读数可以调整目标高度以适应地形。
    Note: For flat terrain, target height is in the world frame. For rough terrain,
    sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


# ============= 2.2 新增：精确速度追踪函数 =============
def track_lin_vel_xy_exp_precise(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """精确的线速度XY追踪奖励 - 使用指数核 / Precise linear velocity XY tracking reward - using exponential kernel.

    用于2.2任务的更精确速度追踪 / Used for more precise velocity tracking in task 2.2.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取命令和实际速度 / Get commands and actual velocities
    commands = env.command_manager.get_command(command_name)
    lin_vel_error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]

    # 使用指数核计算奖励 / Compute reward using exponential kernel
    error_norm = torch.norm(lin_vel_error, dim=1)
    return torch.exp(-error_norm ** 2 / std ** 2)


def track_ang_vel_z_exp_precise(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """精确的角速度Z追踪奖励 - 使用指数核 / Precise angular velocity Z tracking reward - using exponential kernel.

    用于2.2任务的更精确角速度追踪 / Used for more precise angular velocity tracking in task 2.2.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取命令和实际角速度 / Get commands and actual angular velocities
    commands = env.command_manager.get_command(command_name)
    ang_vel_error = commands[:, 2] - asset.data.root_ang_vel_w[:, 2]

    # 使用指数核计算奖励 / Compute reward using exponential kernel
    return torch.exp(-ang_vel_error ** 2 / std ** 2)


def base_orientation_stability_reward(
        env: ManagerBasedRLEnv,
        max_roll: float,
        max_pitch: float,
        decay_factor: float = 0.8,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基座姿态稳定性奖励 - 惩罚过大的Roll/Pitch震荡 / Base orientation stability reward - penalizes excessive Roll/Pitch oscillations.

    用于2.2任务，鼓励机器人保持稳定姿态 / Used for task 2.2, encourages robot to maintain stable orientation.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 从四元数中提取欧拉角 / Extract Euler angles from quaternion
    quat = asset.data.root_quat_w
    # 转换为Roll, Pitch, Yaw / Convert to Roll, Pitch, Yaw
    # Using same formula as base_orientation_stability() - clamp asin input to avoid NaN
    roll = torch.atan2(2.0 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
                       1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2))
    pitch = torch.asin(torch.clamp(2.0 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1.0, 1.0))

    # 计算姿态稳定性分数 / Compute orientation stability score
    roll_stability = torch.exp(-torch.abs(roll) / max_roll)
    pitch_stability = torch.exp(-torch.abs(pitch) / max_pitch)

    # 综合稳定性分数 / Combined stability score
    stability_score = (roll_stability + pitch_stability) / 2.0

    return stability_score


# ============= 新增：观测相关函数（用于观测项） =============
def velocity_tracking_error(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算速度追踪误差 / Compute velocity tracking error.

    用于观测项，提供给Policy网络 / Used for observation term, provided to Policy network.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)

    # 计算线速度和角速度误差 / Compute linear and angular velocity errors
    lin_vel_error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]
    ang_vel_error = commands[:, 2] - asset.data.root_ang_vel_w[:, 2]

    # 返回误差向量 / Return error vector
    return torch.cat([lin_vel_error, ang_vel_error.unsqueeze(1)], dim=1)


def base_orientation_stability(
        env: ManagerBasedRLEnv,
        window_size: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算基座姿态稳定性指标 / Compute base orientation stability metric.

    用于观测项，计算Roll/Pitch的震荡幅度 / Used for observation term, computes Roll/Pitch oscillation amplitude.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 从四元数中提取欧拉角 / Extract Euler angles from quaternion
    quat = asset.data.root_quat_w
    roll = torch.atan2(2.0 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
                       1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2))
    pitch = torch.asin(2.0 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]))

    # 在实际实现中，这里需要历史数据来计算震荡幅度 / In actual implementation, historical data is needed to compute oscillation amplitude
    # 这里简化实现，返回当前姿态角的绝对值 / Simplified implementation, return absolute value of current orientation angles
    orientation_instability = torch.abs(roll) + torch.abs(pitch)

    return orientation_instability


class GaitReward(ManagerTermBase):
    """步态奖励类 - 核心步态控制奖励函数 / Gait reward class - core gait control reward function"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """初始化步态奖励项 / Initialize the gait reward term.

        Args:
            cfg: 奖励配置 / The configuration of the reward.
            env: RL环境实例 / The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # 提取使用的数量（启用类型提示）/ Extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # 存储配置参数 / Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            tracking_contacts_shaped_force,
            tracking_contacts_shaped_vel,
            gait_force_sigma,
            gait_vel_sigma,
            kappa_gait_probs,
            command_name,
            sensor_cfg,
            asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
                1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """计算基于力的奖励组件 / Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """计算基于速度的奖励组件 / Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """动作平滑性惩罚类 - 惩罚网络动作输出的大幅瞬时变化 / Action smoothness penalty class - penalizes large instantaneous changes in network action output.

    此惩罚鼓励动作随时间更平滑。/ This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


# ============= 新增：其他必要的函数 =============
# 注意：以下函数可能在其他地方有定义，但为了完整性在这里提供简化版本
# Note: The following functions may be defined elsewhere, but provided here in simplified version for completeness

def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """线速度XY追踪奖励 - 原始版本 / Linear velocity XY tracking reward - original version."""
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    lin_vel_error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]
    error_norm = torch.norm(lin_vel_error, dim=1)
    return torch.exp(-error_norm ** 2 / std ** 2)


def track_ang_vel_z_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """角速度Z追踪奖励 - 原始版本 / Angular velocity Z tracking reward - original version."""
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    ang_vel_error = commands[:, 2] - asset.data.root_ang_vel_w[:, 2]
    return torch.exp(-ang_vel_error ** 2 / std ** 2)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Z方向线速度L2惩罚 / Z-direction linear velocity L2 penalty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """XY方向角速度L2惩罚 / XY-direction angular velocity L2 penalty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, :2]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节力矩L2惩罚 / Joint torques L2 penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节加速度L2惩罚 / Joint acceleration L2 penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    # 注意：可能需要计算关节加速度 / Note: May need to compute joint acceleration
    return torch.zeros(env.num_envs, device=env.device)  # 简化实现 / Simplified implementation


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """动作变化率L2惩罚 / Action rate L2 penalty."""
    # 这个函数的具体实现依赖于环境如何跟踪动作历史 / Implementation depends on how env tracks action history
    return torch.zeros(env.num_envs, device=env.device)  # 简化实现 / Simplified implementation


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节位置限制惩罚 / Joint position limits penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    # 检查关节位置是否超出限制 / Check if joint positions exceed limits
    return torch.zeros(env.num_envs, device=env.device)  # 简化实现 / Simplified implementation


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节速度L2惩罚 / Joint velocity L2 penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """不期望接触惩罚 / Undesired contacts penalty."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    undesired = contact_forces > threshold
    return torch.sum(undesired.float(), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """平坦朝向L2惩罚 / Flat orientation L2 penalty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    # 惩罚非垂直朝向 / Penalize non-vertical orientation
    return torch.square(quat[:, 0]) + torch.square(quat[:, 1])  # 惩罚Roll和Pitch / Penalize Roll and Pitch


# ============= 新增：平面地形课程学习函数 =============
def flat_terrain_levels(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        step_size: float = 0.1,
        max_level: int = 10,
) -> torch.Tensor:
    """平面地形课程学习函数 - 基于性能渐进增加难度 / Flat terrain curriculum function - gradually increase difficulty based on performance.

    由于使用平面地形，这里可以基于追踪精度或抗干扰能力来增加难度。
    Since flat terrain is used, difficulty can be increased based on tracking accuracy or disturbance rejection ability.
    """
    # 简化实现：返回固定等级或基于性能的等级
    # 这里可以根据实际训练进度动态调整
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 获取当前训练进度（可根据需要实现）
    # 这里简单返回一个固定的等级
    levels = torch.ones(len(env_ids), device=env.device, dtype=torch.int32)

    return levels


# ============= 2.5 新增：双足跳（Pronk）相关奖励函数 =============
def feet_synchronization(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """双足同步奖励 - 鼓励两脚状态一致 / Feet synchronization reward - encourage both feet to be in the same state.
    
    用于Pronking（双足跳）步态 / Used for Pronking gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触状态 (Threshold 1.0N)
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    
    # 逻辑：如果两脚状态不同（异或为真），则惩罚；相同则奖励
    # Logic: If feet states are different (XOR is true), penalize; if same, reward
    # Assuming 2 feet. contacts shape: (num_envs, 2)
    desync = torch.logical_xor(contacts[:, 0], contacts[:, 1])
    
    return 1.0 - desync.float()

def jump_vertical_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """跳跃垂直速度奖励 / Jump vertical velocity reward.
    
    鼓励向上运动，忽略向下运动 / Encourage upward motion, ignore downward motion.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z_vel = asset.data.root_lin_vel_w[:, 2]
    # 仅奖励正向（向上）速度 / Only reward positive (upward) velocity
    return torch.clamp(z_vel, min=0.0)

def pronk_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pronk腾空时间奖励 / Pronk air time reward.
    
    仅当两脚同时腾空时给予奖励 / Reward only when BOTH feet are in the air.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    
    # 计算接触脚的数量 / Count number of feet in contact
    num_contacts = torch.sum(contacts.float(), dim=1)
    
    # 腾空阶段：接触数量为0 / Flight phase: number of contacts is 0
    in_air = (num_contacts == 0)
    
    return in_air.float()