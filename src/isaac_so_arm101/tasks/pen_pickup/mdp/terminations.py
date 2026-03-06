from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

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


def pen_holder_knocked_over(
    env: ManagerBasedRLEnv,
    tilt_threshold: float = 0.5,
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder"),
) -> torch.Tensor:
    """Terminate if the pen holder is knocked over (tilted beyond threshold).

    Checks the dot product of the holder's local up-vector with world up.
    A perfectly upright holder gives dot=1.0. Threshold of 0.5 corresponds
    to ~60 degrees of tilt.

    Args:
        tilt_threshold: Minimum dot product with world up to be "upright".
            0.5 = ~60 deg tilt allowed. 0.7 = ~45 deg. 0.87 = ~30 deg.
    """
    holder: RigidObject = env.scene[holder_cfg.name]
    quat = holder.data.root_quat_w  # (N, 4) as (w, x, y, z)
    # The holder model's local Y-axis is its tall axis (points up when standing).
    # init rot=(0.7071,0.7071,0,0) is 90° around X, which maps local Y → world Z.
    # So we rotate local Y by current orientation and check its world-Z component.
    local_up = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(quat.shape[0], -1)
    holder_up = quat_apply(quat, local_up)
    dot = holder_up[:, 2]
    return dot < tilt_threshold


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
