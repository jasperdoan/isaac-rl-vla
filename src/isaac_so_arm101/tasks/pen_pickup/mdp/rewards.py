from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pen_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for the end-effector reaching the pen using tanh kernel."""
    pen: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pen_pos_w = pen.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(pen_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def pen_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
) -> torch.Tensor:
    """Reward if pen is lifted above a minimal height from the table surface."""
    pen: RigidObject = env.scene[object_cfg.name]
    return torch.where(pen.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def pen_holder_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    holder_half_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder"),
) -> torch.Tensor:
    """Reward for moving the lifted pen toward the pen holder top.

    Only gives reward when pen is above minimal_height (i.e., being carried).
    Uses distance to the top-center of the holder.
    """
    pen: RigidObject = env.scene[object_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]
    pen_pos = pen.data.root_pos_w[:, :3]
    holder_pos = holder.data.root_pos_w[:, :3]

    # Target is the top-center of the pen holder
    target = holder_pos.clone()
    target[:, 2] = target[:, 2] + holder_half_height

    distance = torch.norm(pen_pos - target, dim=1)
    is_lifted = pen_pos[:, 2] > minimal_height
    return is_lifted.float() * (1 - torch.tanh(distance / std))


def pen_at_holder(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_min: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder"),
) -> torch.Tensor:
    """Reward for pen being directly above the pen holder opening.

    Args:
        xy_threshold: Max XY distance from holder center to count as "above".
        z_min: Minimum Z height above holder center to count as valid drop position.
    """
    pen: RigidObject = env.scene[object_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]
    pen_pos = pen.data.root_pos_w[:, :3]
    holder_pos = holder.data.root_pos_w[:, :3]

    xy_dist = torch.norm(pen_pos[:, :2] - holder_pos[:, :2], dim=1)
    above_holder = pen_pos[:, 2] > (holder_pos[:, 2] + z_min)
    within_radius = xy_dist < xy_threshold

    return (above_holder & within_radius).float()
