#!/usr/bin/env python3
"""
Real-time 3DGS rendering visualizer.

Displays RGB + depth from the 3DGS model in an OpenCV window.
Press ESC to exit, SPACE to cycle environments, S to save current frame.

Usage:
    python airgym/gs_renderer/test_render.py [--ply PATH] [--res W H] [--viz]
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from airgym.gs_renderer import GSRenderer, compute_camera_pose, fov_to_focal


def live_viz(renderer, fx, fy, cx, cy, width, height, output_dir=None):
    """Interactive real-time visualization with flyable camera."""
    print("\n" + "=" * 60)
    print("3DGS Live Visualizer")
    print("  Keys: ESC/quit, W/S/A/D = move, Q/E = altitude, R = reset")
    print("        Space = save frame, Arrow keys = look")
    print("=" * 60)

    pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # x, y, z
    yaw, pitch = 0.0, 0.0
    move_speed = 0.2
    look_speed = 3.0
    frame = 0
    save_dir = output_dir or "/tmp/gs_viz"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        # Build rotation from yaw/pitch (camera looks along +Z in OpenCV convention)
        cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))

        R = np.array([
            [cy, 0, sy],
            [-sp * sy, cp, sp * cy],
            [-cp * sy, -sp, cp * cy],
        ], dtype=np.float32)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = pos

        pose_t = torch.from_numpy(pose[:3]).unsqueeze(0).to("cuda")

        t0 = time.perf_counter()
        result = renderer.render(pose_t, fx, fy, cx, cy, width, height,
                                 near_plane=0.01, far_plane=1000.0)
        elapsed = (time.perf_counter() - t0) * 1000

        color = result["color"][0].cpu().numpy()
        depth = result["depth"][0].squeeze(-1).cpu().numpy()

        # RGB display
        rgb_bgr = cv2.cvtColor((color * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Depth display
        valid = depth > 0
        if valid.any():
            d_min, d_max = depth[valid].min(), depth.max()
            depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else depth
        else:
            depth_norm = np.zeros_like(depth)
        depth_colored = cv2.applyColorMap((depth_norm * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        # Combine
        combined = np.hstack([rgb_bgr, depth_colored])
        h, w = combined.shape[:2]

        # HUD overlay
        lines = [
            f"FPS: {1000/elapsed:.0f}" if elapsed > 0 else "FPS: --",
            f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"Yaw: {yaw:.0f} Pitch: {pitch:.0f}",
            f"Frame: {frame}  |  ESC: quit  Space: save",
        ]
        for i, line in enumerate(lines):
            cv2.putText(combined, line, (10, h - 15 * (len(lines) - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("3DGS Live", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):
            path = os.path.join(save_dir, f"frame_{frame:04d}.png")
            cv2.imwrite(path, combined)
            print(f"  Saved: {path}")
        elif key == ord('r'):
            pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
            yaw, pitch = 0.0, 0.0
        elif key == ord('w'):
            pos[2] += move_speed
        elif key == ord('s'):
            pos[2] -= move_speed
        elif key == ord('a'):
            pos[0] -= move_speed
        elif key == ord('d'):
            pos[0] += move_speed
        elif key == ord('q'):
            pos[1] += move_speed
        elif key == ord('e'):
            pos[1] -= move_speed
        elif key == 81:  # left arrow
            yaw -= look_speed
        elif key == 83:  # right arrow
            yaw += look_speed
        elif key == 82:  # up arrow
            pitch = min(89.0, pitch + look_speed)
        elif key == 84:  # down arrow
            pitch = max(-89.0, pitch - look_speed)

        frame += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="3DGS Real-time Visualizer")
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--output", type=str, default="/tmp/gs_viz")
    parser.add_argument("--res", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    width, height = args.res
    fx = fov_to_focal(87.0, width)
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    print(f"Loading model: {args.ply}")
    renderer = GSRenderer(args.ply, sh_degree=args.sh_degree, device="cuda:0")

    if args.no_viz:
        # Headless benchmark
        print(f"\nHeadless benchmark: {width}x{height}, 100 frames")
        pose = torch.eye(4, device="cuda")[:3, :].unsqueeze(0)
        pose[0, 2, 3] = 3.0

        times = []
        for i in range(100):
            t0 = time.perf_counter()
            renderer.render(pose, fx, fy, cx, cy, width, height)
            times.append(time.perf_counter() - t0)

        times = np.array(times[10:])  # skip warmup
        print(f"  Avg: {times.mean()*1000:.2f}ms")
        print(f"  FPS: {1/times.mean():.0f}")
    else:
        live_viz(renderer, fx, fy, cx, cy, width, height, args.output)


if __name__ == "__main__":
    main()
