"""
GS-SDF 3D Gaussian Splatting Renderer Module

Provides:
    GSRenderer:  Loads 3DGS .ply models and renders RGB + depth images
    compute_camera_pose: Convert Isaac Gym drone pose to 3DGS camera extrinsics
"""
import os as _os

_SETUP_DONE = False

def _setup():
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    import torch as _torch
    _torch_path = _os.path.dirname(_os.path.abspath(_torch.__file__))
    _nvidia_cuda = _os.path.join(_torch_path, "..", "nvidia", "cuda_runtime", "lib")
    _nvidia_cuda = _os.path.normpath(_nvidia_cuda)
    if _os.path.isdir(_nvidia_cuda):
        _ld_path = _os.environ.get("LD_LIBRARY_PATH", "")
        if _nvidia_cuda not in _ld_path:
            _os.environ["LD_LIBRARY_PATH"] = _nvidia_cuda + ":" + _ld_path

from .renderer import GSRenderer
from .camera_utils import (
    compute_camera_pose,
    focal_to_fov,
    fov_to_focal,
    prepare_policy_observation,
)
