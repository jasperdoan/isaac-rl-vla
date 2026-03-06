"""Configuration for the pen pickup and drop environment.

The robot must pick up a pen from a random position on the table and
drop it into a pen holder placed at another random position, both
within the robot's reachable annular region (4-14 inches from base).

Scene includes:
- SO-ARM101 robot (dark gray, 3D-printed look)
- Pale wood table (24 x 36 x 29 inches)
- Pen (bright red cylinder, ~14cm long, ~1cm diameter)
- Pen holder (black cylinder, ~8cm diameter, ~10cm tall)
- Wrist camera (640x480, 30fps, mounted on wrist link)
- Front camera (640x480, 30fps, mounted 24in forward, 25in above table)
"""

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import isaac_so_arm101.tasks.pen_pickup.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.utils import configclass

# ==============================================================================
# Unit conversions
# ==============================================================================
INCHES_TO_METERS = 0.0254

# Path to the assets/ folder next to this file
# After converting FBX → USD (see assets/convert_assets.py), the files are:
#   assets/pen.usd  and  assets/pen_holder.usd
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# 3D model scales — FBX files from Sketchfab are usually authored in centimetres.
# 0.01 converts cm → metres.  If objects appear too big/small, change these.
PEN_SCALE    = 0.0002
HOLDER_SCALE = 0.0075

# Table dimensions
TABLE_LENGTH = 24 * INCHES_TO_METERS   # 0.6096m 
TABLE_WIDTH = 48 * INCHES_TO_METERS    # 0.9144m 
TABLE_HEIGHT = 29 * INCHES_TO_METERS   # 0.7366m (Z axis)
TABLE_TOP_THICKNESS = 0.03             # 3cm thick tabletop slab

# Reachable annular region (from robot center)
REACH_MIN = 4 * INCHES_TO_METERS      # 0.1016m - too close below this
REACH_MAX = 14 * INCHES_TO_METERS     # 0.3556m - too far beyond this

# Camera positions
FRONT_CAM_FORWARD = 24 * INCHES_TO_METERS   # 0.6096m from robot center
FRONT_CAM_HEIGHT = 30 * INCHES_TO_METERS     # 0.635m above table surface
WRIST_CAM_ABOVE = 2 * INCHES_TO_METERS       # 0.0508m above wrist joint

# Pen dimensions (ProGel 0.7 approximation)
PEN_RADIUS = 0.005       # 1cm diameter
PEN_LENGTH = 0.14        # 14cm long
PEN_MASS = 0.012         # ~12 grams

# Pen holder dimensions (wire mesh cup approximation)
HOLDER_RADIUS = 0.04     # 8cm diameter
HOLDER_HEIGHT = 0.10     # 10cm tall


##
# Scene definition
##


@configclass
class PenPickupSceneCfg(InteractiveSceneCfg):
    """Scene configuration for pen pickup and drop task.

    Robot is placed at the origin (middle edge of the table's short side).
    Table extends in the +X direction. Objects spawn within the reachable
    annular region (4-14 inches from robot center).
    """

    # Robot: will be populated by the agent-specific env cfg
    robot: ArticulationCfg = MISSING
    # End-effector frame: will be populated by the agent-specific env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # -- Table --
    # Pale wood colored tabletop slab (24 x 36 x 29 inches)
    # Robot sits at (0,0,0) on the middle of the short edge.
    # Table center is at (TABLE_LENGTH/2, 0, -TABLE_TOP_THICKNESS/2).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_LENGTH, TABLE_WIDTH, TABLE_TOP_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.75, 0.60),  # pale / light wood
                roughness=0.7,
                metallic=0.0,
            ),
            collision_props=CollisionPropertiesCfg(),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_LENGTH / 2, 0.0, -TABLE_TOP_THICKNESS / 2),
        ),
    )

    # -- Pen (rigid body, graspable) --
    # Loaded from assets/pen.usd (converted from pen.fbx).
    # Scale is controlled by PEN_SCALE at the top of this file.
    pen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "pen.usd"),
            scale=(PEN_SCALE, PEN_SCALE, PEN_SCALE),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=PEN_MASS),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.20, 0.0, PEN_RADIUS + 0.001),
            # Lying flat along X axis
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )

    # -- Pen Holder (kinematic, does not move) --
    # Loaded from assets/pen_holder.usd (converted from pen_holder.fbx).
    # Scale is controlled by HOLDER_SCALE at the top of this file.
    # The mesh is hollow visually but collision is convexHull by default (solid).
    # For a physically hollow holder, open the USD in Isaac Sim and set
    # Collision Approximation to "convexDecomposition" on the mesh prim.
    pen_holder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PenHolder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "pen_holder.usd"),
            scale=(HOLDER_SCALE, HOLDER_SCALE, HOLDER_SCALE),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.10, HOLDER_HEIGHT / 2 + 0.001),
            # 90° rotation around X makes the holder stand upright
            rot=(0.7071, 0.7071, 0.0, 0.0),
        ),
    )

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
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/WristCamera",
        update_period=1.0 / 30.0,  # 30 fps
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.0,           # ~3mm typical for wide-angle USB cam
            focus_distance=0.4,
            horizontal_aperture=3.6,    # approximation for 640x480 USB module
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Values matched from Isaac Sim GUI (local wrist_link frame, meters):
            # To tweak position: change the three numbers in pos=(X, Y, Z)
            #   X: forward/back along wrist  Y: left(+)/right(-) side  Z: height above joint
            pos=(0.0, -0.05, 0.15),
            # Euler (x, y, z) deg → quaternion (w, x, y, z):
            rot=(0.0, 0.0, -0.94, 0.342),
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
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=1.0 / 30.0,  # 30 fps
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
            rot=(0.138, -0.701, -0.701, 0.138),
            convention="ros",
        ),
    )

    # -- Ground Plane --
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -TABLE_HEIGHT)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # -- Lighting --
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Policy observations (22D total):
    - Joint positions relative to default (5D)
    - Joint velocities relative (5D)
    - Pen position in robot frame (3D)
    - Pen holder position in robot frame (3D)
    - Last action (6D: 5 arm + 1 gripper)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        pen_position = ObsTerm(func=mdp.pen_position_in_robot_root_frame)
        pen_holder_position = ObsTerm(func=mdp.pen_holder_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events.

    On each reset:
    - Robot joints return to default
    - Pen is placed randomly within the inner half of the reachable annulus
    - Pen holder is placed randomly within the outer half
    This ensures they are separated and both reachable.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_pen_position = EventTerm(
        func=mdp.reset_root_state_annular,
        mode="reset",
        params={
            "r_min": REACH_MIN,
            "r_max": (REACH_MIN + REACH_MAX) / 2,  # inner half of annulus
            "z_offset": PEN_RADIUS + 0.001,         # pen rests on surface
            "angle_min": -math.pi / 3,              # -60 degrees
            "angle_max": math.pi / 3,               # +60 degrees
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )

    reset_pen_holder_position = EventTerm(
        func=mdp.reset_root_state_annular,
        mode="reset",
        params={
            "r_min": (REACH_MIN + REACH_MAX) / 2,   # outer half of annulus
            "r_max": REACH_MAX,
            "z_offset": HOLDER_HEIGHT / 2 + 0.001,  # holder sits on surface
            "angle_min": -math.pi / 3,
            "angle_max": math.pi / 3,
            "asset_cfg": SceneEntityCfg("pen_holder"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Multi-phase reward shaping:
    1. Reach the pen (EE approaches pen)
    2. Lift the pen (pen above table)
    3. Move pen to holder (pen approaches holder top while lifted)
    4. Drop pen in holder (pen above holder opening)
    """

    # Phase 1: Reach the pen
    reaching_pen = RewTerm(
        func=mdp.pen_ee_distance,
        params={"std": 0.05},
        weight=1.0,
    )

    # Phase 2: Lift the pen
    lifting_pen = RewTerm(
        func=mdp.pen_is_lifted,
        params={"minimal_height": 0.025},
        weight=15.0,
    )

    # Phase 3: Move pen toward holder (coarse)
    pen_to_holder_coarse = RewTerm(
        func=mdp.pen_holder_distance,
        params={"std": 0.3, "minimal_height": 0.025, "holder_half_height": HOLDER_HEIGHT / 2},
        weight=16.0,
    )

    # Phase 3: Move pen toward holder (fine)
    pen_to_holder_fine = RewTerm(
        func=mdp.pen_holder_distance,
        params={"std": 0.05, "minimal_height": 0.025, "holder_half_height": HOLDER_HEIGHT / 2},
        weight=5.0,
    )

    # Phase 4: Pen is above the holder opening
    pen_above_holder = RewTerm(
        func=mdp.pen_at_holder,
        params={"xy_threshold": HOLDER_RADIUS, "z_min": 0.03},
        weight=20.0,
    )

    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Pen fell off the table
    pen_dropped = DoneTerm(
        func=mdp.pen_dropped_off_table,
        params={"minimum_height": -0.05},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class PenPickupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pen pickup and drop environment."""

    # Scene settings
    scene: PenPickupSceneCfg = PenPickupSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0  # longer episodes for pick-and-place
        self.viewer.eye = (1.5, 1.5, 1.0)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
