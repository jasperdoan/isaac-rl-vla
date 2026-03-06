"""SO-ARM101 specific configuration for the pen pickup task."""

import isaac_so_arm101.tasks.pen_pickup.mdp as mdp
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.pen_pickup.pen_pickup_env_cfg import PenPickupEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm101PenPickupEnvCfg(PenPickupEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SO-ARM101 as the robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override actions: 5 DOF arm + binary gripper
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )

        # End-effector frame tracker (same as lift task)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
            ],
        )


@configclass
class SoArm101PenPickupEnvCfg_PLAY(SoArm101PenPickupEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Keep 1 env for visual inspection
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
