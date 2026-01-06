"""此子模块包含可用于LimX Point Foot运动任务的奖励函数 / This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

这些函数可以传递给:class:`isaaclab.managers.RewardTermCfg`对象来指定奖励函数及其参数。
The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional
import math

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
        sensor: RayCaster = env.scene[sensorsensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


# ========================================================
# [任务2.3:新增] 抗干扰相关奖励函数
# ========================================================

def disturbance_recovery_reward(
        env: ManagerBasedRLEnv,
        command_name: str,
        recovery_time_window: float = 1.0,
        velocity_error_threshold: float = 0.3,
) -> torch.Tensor:
    """抗干扰恢复奖励 - 鼓励机器人在受到干扰后快速恢复目标速度 / Disturbance recovery reward - encourages robot to quickly recover target velocity after disturbance

    评分标准：受到干扰后恢复步态的速度 / Evaluation criteria: speed of gait recovery after disturbance
    """
    # 获取当前速度和目标速度 / Get current and target velocities
    asset = env.scene["robot"]
    commands = env.command_manager.get_command(command_name)

    current_lin_vel = asset.data.root_lin_vel_w[:, :2]
    target_lin_vel = commands[:, :2]

    # 计算速度误差 / Calculate velocity error
    velocity_error = torch.norm(current_lin_vel - target_lin_vel, dim=1)

    # 检测是否刚受到干扰（通过速度突变）/ Detect if just disturbed (by velocity change)
    if not hasattr(env, '_last_velocities'):
        env._last_velocities = current_lin_vel.clone()
        env._disturbance_timers = torch.zeros(env.num_envs, device=env.device)

    velocity_change = torch.norm(current_lin_vel - env._last_velocities, dim=1)
    disturbance_detected = velocity_change > velocity_error_threshold

    # 更新计时器 / Update timers
    env._disturbance_timers[disturbance_detected] = 0.0
    env._disturbance_timers[~disturbance_detected] += env.step_dt

    # 计算恢复奖励（误差随时间指数衰减）/ Calculate recovery reward (error decays exponentially with time)
    recovery_factor = torch.exp(-env._disturbance_timers / recovery_time_window)
    reward = recovery_factor * torch.exp(-velocity_error / velocity_error_threshold)

    # 更新最后速度 / Update last velocities
    env._last_velocities = current_lin_vel.clone()

    return reward


def posture_robustness_reward(
        env: ManagerBasedRLEnv,
        max_tilt_angle: float = math.pi / 6,
        recovery_time: float = 0.5,
) -> torch.Tensor:
    """姿态抗干扰奖励 - 鼓励机器人在干扰下保持稳定姿态 / Posture robustness reward - encourages robot to maintain stable posture under disturbance

    评分标准：机器人能够承受不摔倒的最大推力冲量 / Evaluation criteria: maximum impulse the robot can withstand without falling
    """
    asset = env.scene["robot"]

    # 计算当前姿态的俯仰和滚转角度 / Calculate pitch and roll angles of current posture
    quat = asset.data.root_quat_w
    # 将四元数转换为欧拉角（roll, pitch, yaw）/ Convert quaternion to Euler angles
    # 简化计算：使用四元数提取倾斜角度 / Simplified: extract tilt angle from quaternion
    tilt_angle = 2 * torch.acos(torch.clamp(torch.abs(quat[:, 3]), 0.0, 1.0))  # 使用四元数标量部分 / Use quaternion scalar part

    # 检测是否刚受到干扰 / Detect if just disturbed
    if not hasattr(env, '_last_tilt_angles'):
        env._last_tilt_angles = tilt_angle.clone()
        env._tilt_recovery_timers = torch.zeros(env.num_envs, device=env.device)

    tilt_change = torch.abs(tilt_angle - env._last_tilt_angles)
    disturbance_detected = tilt_change > 0.1  # 倾斜变化阈值 / Tilt change threshold

    # 更新计时器 / Update timers
    env._tilt_recovery_timers[disturbance_detected] = 0.0
    env._tilt_recovery_timers[~disturbance_detected] += env.step_dt

    # 计算姿态稳定性奖励 / Calculate posture stability reward
    # 1. 惩罚过度倾斜 / Penalize excessive tilt
    tilt_penalty = torch.clamp(tilt_angle / max_tilt_angle, 0.0, 1.0)

    # 2. 奖励快速恢复 / Reward fast recovery
    recovery_factor = torch.exp(-env._tilt_recovery_timers / recovery_time)

    # 综合奖励 / Combined reward
    reward = recovery_factor * (1.0 - tilt_penalty)

    # 更新最后倾斜角度 / Update last tilt angles
    env._last_tilt_angles = tilt_angle.clone()

    return reward


def disturbance_detection_reward(
        env: ManagerBasedRLEnv,
        response_delay_threshold: float = 0.2,
        min_response_magnitude: float = 0.1,
) -> torch.Tensor:
    """干扰检测奖励 - 鼓励策略主动感知和响应干扰 / Disturbance detection reward - encourages policy to actively perceive and respond to disturbance"""

    asset = env.scene["robot"]

    # 获取关节动作变化（策略响应）/ Get joint action changes (policy response)
    if not hasattr(env, '_last_actions'):
        env._last_actions = env.action_manager.action.clone()
        env._response_timers = torch.zeros(env.num_envs, device=env.device)
        env._disturbance_signals = torch.zeros(env.num_envs, device=env.device)

    current_actions = env.action_manager.action.clone()
    action_change = torch.norm(current_actions - env._last_actions, dim=1)

    # 检测外部干扰（通过基座速度突变）/ Detect external disturbance (by base velocity change)
    current_vel = asset.data.root_lin_vel_w[:, :2]
    if not hasattr(env, '_last_velocities_detection'):
        env._last_velocities_detection = current_vel.clone()

    vel_change = torch.norm(current_vel - env._last_velocities_detection, dim=1)
    disturbance_signal = vel_change > min_response_magnitude

    # 更新干扰信号 / Update disturbance signals
    env._disturbance_signals = disturbance_signal.float()

    # 计算响应延迟奖励 / Calculate response delay reward
    # 理想情况：干扰发生后立即有动作响应 / Ideal: action response immediately after disturbance
    response_delay = env._response_timers.clone()

    # 重置检测到响应时的计时器 / Reset timer when response is detected
    response_detected = action_change > min_response_magnitude
    env._response_timers[response_detected] = 0.0
    env._response_timers[~response_detected] += env.step_dt

    # 计算奖励：快速响应干扰 / Calculate reward: quick response to disturbance
    delay_penalty = torch.clamp(response_delay / response_delay_threshold, 0.0, 1.0)
    reward = env._disturbance_signals * (1.0 - delay_penalty)

    # 更新最后值 / Update last values
    env._last_actions = current_actions.clone()
    env._last_velocities_detection = current_vel.clone()

    return reward


def disturbance_energy_penalty(
        env: ManagerBasedRLEnv,
        energy_window: float = 2.0,
        baseline_power: float = 50.0,
) -> torch.Tensor:
    """抗干扰能效惩罚 - 惩罚在干扰下过度消耗能量 / Disturbance energy penalty - penalizes excessive energy consumption under disturbance

    鼓励在保持抗干扰能力的同时提高能效 / Encourages energy efficiency while maintaining disturbance rejection capability
    """
    asset = env.scene["robot"]

    # 计算当前功率 / Calculate current power
    joint_power = torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)

    # 维护功率历史窗口 / Maintain power history window
    if not hasattr(env, '_power_history'):
        env._power_history = torch.zeros((env.num_envs, int(energy_window / env.step_dt)), device=env.device)
        env._power_index = 0

    # 更新历史 / Update history
    env._power_history[:, env._power_index] = joint_power
    env._power_index = (env._power_index + 1) % env._power_history.shape[1]

    # 计算平均功率 / Calculate average power
    avg_power = torch.mean(env._power_history, dim=1)

    # 检测干扰期间（通过速度变化）/ Detect disturbance period (by velocity change)
    current_vel = asset.data.root_lin_vel_w[:, :2]
    if not hasattr(env, '_last_vel_energy'):
        env._last_vel_energy = current_vel.clone()
        env._disturbance_energy_flags = torch.zeros(env.num_envs, device=env.device)

    vel_change = torch.norm(current_vel - env._last_vel_energy, dim=1)
    in_disturbance = vel_change > 0.5  # 速度变化阈值 / Velocity change threshold

    # 更新干扰标志 / Update disturbance flags
    env._disturbance_energy_flags = in_disturbance.float()

    # 计算惩罚：干扰期间的高能耗 / Calculate penalty: high energy consumption during disturbance
    energy_ratio = avg_power / baseline_power
    penalty = env._disturbance_energy_flags * torch.clamp(energy_ratio - 1.0, 0.0, 10.0)

    # 更新最后速度 / Update last velocities
    env._last_vel_energy = current_vel.clone()

    return penalty


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