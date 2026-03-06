from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pen_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
) -> torch.Tensor:
    """The position of the pen in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]
    pen_pos_w = pen.data.root_pos_w[:, :3]
    pen_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], pen_pos_w
    )
    return pen_pos_b


def pen_holder_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder"),
) -> torch.Tensor:
    """The position of the pen holder in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    holder: RigidObject = env.scene[object_cfg.name]
    holder_pos_w = holder.data.root_pos_w[:, :3]
    holder_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], holder_pos_w
    )
    return holder_pos_b


def reset_root_state_annular(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    r_min: float,
    r_max: float,
    z_offset: float,
    angle_min: float = -math.pi / 3,
    angle_max: float = math.pi / 3,
    randomize_yaw: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
):
    """Reset root state to a random position within an annular region.

    Samples uniformly within an annulus (ring) in front of the robot.
    Uses r = sqrt(uniform(r_min^2, r_max^2)) for uniform area sampling.

    Args:
        env: The environment.
        env_ids: The environment indices to reset.
        r_min: Minimum radius from the robot center (meters).
        r_max: Maximum radius from the robot center (meters).
        z_offset: Height offset above table surface (meters).
        angle_min: Minimum angle in radians (default: -60 degrees).
        angle_max: Maximum angle in radians (default: +60 degrees).
        randomize_yaw: If True, apply a random Z-axis rotation to the
            default orientation (e.g. pen pointing in random direction).
        asset_cfg: The asset to reset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get default state
    root_states = asset.data.default_root_state[env_ids].clone()

    # Uniform sampling in annulus: r = sqrt(U * (r_max^2 - r_min^2) + r_min^2)
    n = len(env_ids)
    r = torch.sqrt(
        torch.rand(n, device=env.device) * (r_max**2 - r_min**2) + r_min**2
    )
    theta = torch.rand(n, device=env.device) * (angle_max - angle_min) + angle_min

    # Convert polar to cartesian (X is forward from robot, Y is lateral)
    root_states[:, 0] = r * torch.cos(theta) + env.scene.env_origins[env_ids, 0]
    root_states[:, 1] = r * torch.sin(theta) + env.scene.env_origins[env_ids, 1]
    root_states[:, 2] = z_offset + env.scene.env_origins[env_ids, 2]

    # Randomize yaw (Z-axis rotation) composited with default orientation
    if randomize_yaw:
        yaw = torch.rand(n, device=env.device) * 2 * math.pi  # [0, 2pi)
        # Quaternion for Z-axis rotation: (cos(yaw/2), 0, 0, sin(yaw/2))
        q_yaw = torch.zeros(n, 4, device=env.device)
        q_yaw[:, 0] = torch.cos(yaw / 2)
        q_yaw[:, 3] = torch.sin(yaw / 2)
        # Compose: q_yaw * q_default → rotates the default orientation by yaw
        root_states[:, 3:7] = quat_mul(q_yaw, root_states[:, 3:7])

    # Zero velocities
    root_states[:, 7:13] = 0.0

    asset.write_root_state_to_sim(root_states, env_ids)
