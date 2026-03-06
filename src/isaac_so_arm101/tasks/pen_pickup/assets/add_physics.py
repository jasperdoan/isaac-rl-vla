"""Add USD Physics APIs to converted FBX assets.

pxr is bundled with Isaac Sim and only available after AppLauncher runs.
Run with:
    uv run --active python src/isaac_so_arm101/tasks/pen_pickup/assets/add_physics.py
"""

import argparse
import sys
from pathlib import Path

# -- Launch Isaac Sim first so it adds pxr to sys.path --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Add physics APIs to USD assets.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is now importable (Isaac Sim added its USD libs to sys.path)
from pxr import Usd, UsdPhysics  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent

ASSETS = [
    # (filename,         kinematic,  mass_kg)
    ("pen.usd",        False,      0.012),   # dynamic, 12g
    ("pen_holder.usd", True,       0.100),   # static/kinematic
]


def add_physics(usd_path: Path, kinematic: bool, mass: float) -> None:
    if not usd_path.exists():
        print(f"[SKIP]  {usd_path.name} not found — run convert_assets.py first")
        return

    stage = Usd.Stage.Open(str(usd_path))

    # Find (or set) the default prim
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        root_children = list(stage.GetPseudoRoot().GetChildren())
        if not root_children:
            print(f"[ERROR] {usd_path.name}: no prims found")
            return
        default_prim = root_children[0]
        stage.SetDefaultPrim(default_prim)
        print(f"  Set default prim → {default_prim.GetPath()}")

    # RigidBodyAPI on root prim
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(default_prim)
    rigid_api.CreateKinematicEnabledAttr(kinematic)

    # MassAPI on root prim
    mass_api = UsdPhysics.MassAPI.Apply(default_prim)
    mass_api.CreateMassAttr(mass)

    # CollisionAPI on every Mesh prim (fallback: root prim)
    # Explicitly set convexHull so PhysX doesn't log a fallback error.
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.CreateApproximationAttr("convexHull")
            mesh_count += 1
    if mesh_count == 0:
        UsdPhysics.CollisionAPI.Apply(default_prim)

    stage.Save()
    print(
        f"[OK]    {usd_path.name}  "
        f"kinematic={kinematic}  mass={mass}kg  meshes={mesh_count}"
    )


print("Applying physics APIs...")
for filename, kinematic, mass in ASSETS:
    add_physics(ASSETS_DIR / filename, kinematic, mass)

simulation_app.close()
print("Done. Re-run random_agent to test.")
