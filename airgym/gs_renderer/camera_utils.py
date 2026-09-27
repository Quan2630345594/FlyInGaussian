"""
Camera and coordinate system conversion utilities for integrating
Isaac Gym drone cameras with the 3DGS renderer.
"""

import torch
import numpy as np


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert WXYZ quaternions to rotation matrices without extra dependencies."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def compute_camera_pose(
    drone_pos: torch.Tensor,
    drone_quat: torch.Tensor,
    cam_offset: torch.Tensor = None,
) -> torch.Tensor:
    """
    Convert Isaac Gym drone pose to 3DGS camera extrinsics (camera-to-world).

    Isaac Gym convention:
        - World: Z-up
        - Drone: body X-forward, body Y-left, body Z-up
        - Quaternion: XYZW order
        - Camera offset relative to base_link: (x_forward, y_left, z_up) in meters

    3DGS / OpenCV convention:
        - Camera-to-world: [R | t] where camera looks along +Z, Y-down

    The conversion applies a roll=180 degrees rotation (pi around camera Z-axis)
    to go from Isaac Gym's Y-up-in-image to OpenCV's Y-down-in-image.

    Args:
        drone_pos: [N, 3] drone position in world frame (Isaac Gym).
        drone_quat: [N, 4] drone orientation as XYZW quaternion (Isaac Gym).
        cam_offset: [3] camera offset in body frame (x_forward, y_left, z_up).
                    Default: (0.15, 0.0, 0.1)

    Returns:
        pose_cam2world: [N, 3, 4] camera-to-world extrinsics (OpenCV convention).
    """
    if cam_offset is None:
        cam_offset = torch.tensor([0.15, 0.0, 0.1], device=drone_pos.device, dtype=drone_pos.dtype)

    N = drone_pos.shape[0]
    device = drone_pos.device

    # Isaac Gym quaternion is XYZW, convert to WXYZ for pytorch3d
    drone_quat_wxyz = drone_quat[:, [3, 0, 1, 2]]
    drone_R_wxyz = _quaternion_to_matrix(drone_quat_wxyz)

    # Step 1: Camera position in world frame
    cam_pos_world = drone_pos + torch.bmm(
        drone_R_wxyz, cam_offset.view(1, 3, 1).expand(N, 3, 1)
    ).squeeze(-1)

    # Step 2: Isaac Gym body orientation:
    #   body X -> forward
    #   body Y -> left
    #   body Z -> up
    # OpenCV camera orientation:
    #   camera Z -> forward (look direction)
    #   camera X -> right
    #   camera Y -> down

    # Rotation from body frame to OpenCV camera frame:
    #   body X (forward) -> camera Z (forward)
    #   body Y (left)    -> camera -X (right in OpenCV is -Y axis... hmm)
    #
    # Actually, let's think of this as:
    #   We want a rotation matrix R_body_to_cam such that:
    #   - Body X axis maps to camera Z axis
    #   - Body Y axis maps to camera -X axis (since body left = camera right, and OpenCV camera X = right)
    #   - Body Z axis maps to camera -Y axis (body up -> OpenCV camera Y is down, so body Z -> camera -Y)

    # R_body_to_cam:
    #   col0 = [0, -1, 0]   (body X -> camera Z)
    #   col1 = [-1, 0, 0]   (body Y -> camera -X)
    #   col2 = [0, 0, -1]    (body Z -> camera -Y)
    # Wait, let me think again...

    # Isaac Gym drone body frame:
    #   X-forward, Y-left, Z-up
    # OpenCV camera frame:
    #   X-right, Y-down, Z-forward (looking direction)

    # Mapping:
    #   body X (forward) -> camera Z (forward)
    #   body Y (left)    -> camera -X (right is -Y? no, X is right in OpenCV, left is -X)
    #   body Z (up)      -> camera -Y (down is Y, up is -Y)

    # R_body_to_cam = [[0, -1, 0],    # camera X from body axes
    #                   [0, 0, -1],    # camera Y from body axes
    #                   [1, 0, 0]]     # camera Z from body axes

    R_body_to_cam = torch.tensor(
        [[0., -1., 0.],
         [0., 0., -1.],
         [1., 0., 0.]],
        device=device, dtype=drone_R_wxyz.dtype
    ).unsqueeze(0).expand(N, 3, 3)

    # Camera rotation in world frame: R_world_cam = R_world_body * R_body_to_cam
    R_cam2world = torch.bmm(drone_R_wxyz, R_body_to_cam.transpose(1, 2))
    # R_cam2world: [N, 3, 3] - camera-to-world rotation

    # Build pose_cam2world = [R | t]
    pose_cam2world = torch.cat([R_cam2world, cam_pos_world.unsqueeze(-1)], dim=-1)  # [N, 3, 4]

    return pose_cam2world


def fov_to_focal(fov_deg: float, image_size: int) -> float:
    """Convert horizontal FOV (degrees) to focal length in pixels."""
    return 0.5 * image_size / np.tan(np.deg2rad(0.5 * fov_deg))


def focal_to_fov(focal: float, image_size: int) -> float:
    """Convert focal length in pixels to horizontal FOV (degrees)."""
    return 2.0 * np.rad2deg(np.arctan(0.5 * image_size / focal))


def prepare_policy_observation(
    color: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    observation: str,
    far_plane: float,
    scene_scale: float,
    alpha_threshold: float,
):
    """Convert renderer HWC output into the policy's normalized CWH layout."""
    color = color.clamp(0.0, 1.0)
    depth = depth * scene_scale
    depth = torch.where(
        alpha >= alpha_threshold,
        depth,
        torch.full_like(depth, far_plane),
    ).clamp(0.0, far_plane)
    if observation == "rgb":
        policy_observation = color.permute(0, 3, 2, 1)
    else:
        policy_observation = (depth / far_plane).permute(0, 3, 2, 1)
    return policy_observation, color, depth
