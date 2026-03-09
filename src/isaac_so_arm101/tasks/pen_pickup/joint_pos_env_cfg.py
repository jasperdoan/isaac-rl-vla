"""SO-ARM101 specific configuration for the pen pickup task."""

import isaac_so_arm101.tasks.pen_pickup.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.pen_pickup.pen_pickup_env_cfg import (
    FRONT_CAM_FORWARD,
    FRONT_CAM_HEIGHT,
    PenPickupEnvCfg,
)

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
        # -- Wrist Camera --
        # 640x480, 30fps USB camera module.
        # Mounted on wrist_link, 1 inch above the wrist joint,
        # on the right side of the arm, pointing toward the gripper/claw.
        #
        # TUNING GUIDE (all values in this offset block):
        #   pos=(X, Y, Z) in wrist_link local frame:
        #     X=0.0   : no forward/backward offset
        #     Y=-0.02 : right side of arm (negative Y = right in ROS)
        #     Z=WRIST_CAM_ABOVE : height above wrist joint (change constant on line 58)
        #   rot=(w, x, y, z) quaternion in ROS convention (camera +Z = optical axis):
        #     Current: points toward gripper (downward along -Z of wrist_link)
        #     To tilt more downward: decrease w, increase x magnitude
        #     To tilt less: increase w toward 1.0, decrease x toward 0.0
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/WristCamera",
            update_period=1.0 / 30.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=3.0,
                focus_distance=0.4,
                horizontal_aperture=3.6,
                clipping_range=(0.01, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                # Values matched from Isaac Sim GUI (local wrist_link frame, meters):
                # To tweak position: change the three numbers in pos=(X, Y, Z)
                #   X: forward/back along wrist  Y: left(+)/right(-) side  Z: height above joint
                pos=(0.005, 0.075, 0.0),
                # Euler (x, y, z) deg → quaternion (w, x, y, z):
                # This is (-25, 0, 0) in XYZ order, ROS convention (camera +Z forward):
                rot=(-0.216, -0.976, 0.0, 0.0),
                convention="ros",
            ),
        )
        # -- Front / Top-Down Camera --
        # 640x480, 30fps USB camera module.
        # Mounted 24 inches forward from robot center (facing the robot),
        # 25 inches above the table surface, tilted ~60 degrees downward.
        #
        # Rotation quaternion (ROS convention, camera +Z = forward):
        #   Pitch -150 deg around Y, then Roll +90 deg around Z.
        #   Result: camera looks toward -X and 60 deg below horizontal.
        #   q = (0.183, -0.683, -0.683, 0.183)
        #
        # NOTE: Fine-tune in Isaac Sim if the image orientation is off.
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=1.0 / 30.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=3.0,
                focus_distance=0.8,
                horizontal_aperture=3.6,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(FRONT_CAM_FORWARD, 0.0, FRONT_CAM_HEIGHT),
                rot=(0.123, -0.696, -0.696, 0.123),
                convention="ros",
            ),
        )
