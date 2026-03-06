from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pen_dropped_off_table(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
) -> torch.Tensor:
    """Terminate if pen falls below the table (dropped off)."""
    pen: RigidObject = env.scene[object_cfg.name]
    return pen.data.root_pos_w[:, 2] < minimum_height


def pen_reached_holder(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.03,
    z_threshold: float = 0.02,
    holder_half_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder"),
) -> torch.Tensor:
    """Success termination: pen is inside the pen holder volume.

    Checks if the pen XY position is within the holder radius and
    the pen Z position is near the holder top.
    """
    pen: RigidObject = env.scene[object_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]
    pen_pos = pen.data.root_pos_w[:, :3]
    holder_pos = holder.data.root_pos_w[:, :3]

    xy_dist = torch.norm(pen_pos[:, :2] - holder_pos[:, :2], dim=1)
    holder_top_z = holder_pos[:, 2] + holder_half_height
    z_near_top = torch.abs(pen_pos[:, 2] - holder_top_z) < z_threshold
    within_radius = xy_dist < xy_threshold

    return within_radius & z_near_top
