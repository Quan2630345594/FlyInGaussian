#!/usr/bin/env python3
"""
Launch AirGym simulator with 3DGS rendering enabled.

Shows the Isaac Gym viewer + OpenCV window with 3DGS RGB/Depth output.

Usage:
    python airgym/gs_renderer/run_sim.py --ply /path/to/model.ply [--num-envs 4] [--sh-degree 3]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument("--ply", type=str, required=True, help="Path to 3DGS .ply model")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--render-w", type=int, default=640)
parser.add_argument("--render-h", type=int, default=512)
parser.add_argument("--fov", type=float, default=87.0)
parser.add_argument("--ctl-mode", type=str, default="pos", help="pos/vel/atti/rate/prop")
parser.add_argument("--sim-device", type=str, default="cuda:0")
parser.add_argument("--headless", action="store_true", help="Run without viewer windows")
args_cli = parser.parse_args()

# isaacgym MUST be imported before torch
import isaacgym
from isaacgym import gymapi, gymutil

import numpy as np
import torch

from airgym.envs import *
from airgym.utils import task_registry
from airgym.utils.helpers import parse_sim_params, class_to_dict


class SimpleArgs:
    """Manual args to avoid argparse conflicts with Isaac Gym."""
    sim_device = args_cli.sim_device
    sim_device_type = "cuda" if "cuda" in args_cli.sim_device else "cpu"
    sim_device_id = 0
    graphics_device_id = 0
    headless = args_cli.headless
    physics_engine = gymapi.SIM_PHYSX
    pipeline = "gpu"
    compute_device_id = 0
    num_threads = 4
    subscenes = 0
    slices = None
    num_envs = args_cli.num_envs
    use_gpu = True
    use_gpu_pipeline = True
    device = args_cli.sim_device


def main():
    args = SimpleArgs()
    env_cfg = task_registry.get_cfgs("customized")

    env_cfg.env.num_envs = args_cli.num_envs
    env_cfg.env.ctl_mode = args_cli.ctl_mode
    env_cfg.env.episode_length_s = 600

    # Minimal scene
    env_cfg.asset_config.include_group_asset = {}
    env_cfg.asset_config.include_single_asset = {}
    env_cfg.asset_config.include_boundary = {"18x18ground": {"num_assets": 1}}

    # Enable 3DGS
    env_cfg.asset_config.gs_model = {
        "enabled": True,
        "ply_path": args_cli.ply,
        "sh_degree": args_cli.sh_degree,
        "render_width": args_cli.render_w,
        "render_height": args_cli.render_h,
        "horizontal_fov": args_cli.fov,
        "near": 0.01,
        "far": 1000.0,
        "background": 0,
        "replace_isaac_camera": True,
        "visualize": not args_cli.headless,
        "cam_offset": [0.15, 0.0, 0.1],
    }

    seed = env_cfg.seed
    if seed == -1:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    print("="*60)
    print("FlyInGaussian Simulator")
    print("  Model  :", args_cli.ply)
    print("  Res    :", args_cli.render_w, "x", args_cli.render_h)
    print("  Envs   :", args_cli.num_envs)
    print("  Device :", args_cli.sim_device)
    print("  Ctrl   :", args_cli.ctl_mode)
    print("="*60)

    env = task_registry.task_classes["customized"](
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )

    print("\nResetting...")
    env.reset()

    num_actions = env_cfg.env.num_actions
    actions = torch.zeros((env_cfg.env.num_envs, num_actions), device=args.sim_device)
    if num_actions >= 4:
        actions[:, -1] = -0.6

    print("Running. ESC in viewer to quit.\n")

    step = 0
    try:
        while True:
            obs, privileged, rewards, resets, extras = env.step(actions)

            if step % 200 == 0:
                p = env.root_positions[0].cpu().numpy()
                print(f"  step {step:5d}  pos=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}]")

            step += 1
    except KeyboardInterrupt:
        pass

    print(f"\nDone ({step} steps).")
    import cv2
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
