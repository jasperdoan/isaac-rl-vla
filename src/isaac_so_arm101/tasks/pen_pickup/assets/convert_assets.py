"""Convert FBX assets to USD for use in Isaac Sim.

Run from the repo root with:
    uv run --active python src/isaac_so_arm101/tasks/pen_pickup/assets/convert_assets.py

Output USD files are written next to the input FBX files.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# ── Launch Isaac Sim first ────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert FBX assets to USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Enable the asset converter extension (not loaded by the headless kit) ─────
import omni.kit.app  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)

import omni.kit.asset_converter as asset_converter  # noqa: E402

# ── Asset paths ───────────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parent

CONVERSIONS = [
    (ASSETS_DIR / "pen.fbx",        ASSETS_DIR / "pen.usd"),
    (ASSETS_DIR / "pen_holder.fbx", ASSETS_DIR / "pen_holder.usd"),
]


async def convert(src: Path, dst: Path) -> bool:
    ctx = asset_converter.AssetConverterContext()
    ctx.ignore_materials = False   # keep materials/textures from FBX
    ctx.ignore_animations = True   # no animations needed
    ctx.single_mesh = False        # preserve sub-meshes

    task = asset_converter.get_instance().create_converter_task(
        str(src), str(dst), None, ctx
    )
    ok = await task.wait_until_finished()
    if ok:
        print(f"[OK]    {dst.name}  ←  {src.name}")
    else:
        print(f"[ERROR] {src.name}: {task.get_status_string()}", file=sys.stderr)
    return ok


async def main():
    all_ok = True
    for src, dst in CONVERSIONS:
        if not src.exists():
            print(f"[SKIP]  {src.name} not found", file=sys.stderr)
            continue
        if dst.exists():
            print(f"[EXISTS] {dst.name} already exists — delete it to re-convert")
            continue
        all_ok = await convert(src, dst) and all_ok
    return all_ok


ok = asyncio.get_event_loop().run_until_complete(main())
simulation_app.close()
sys.exit(0 if ok else 1)
