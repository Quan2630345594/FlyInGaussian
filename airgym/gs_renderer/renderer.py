"""
3D Gaussian Splatting Renderer

Wraps the _gs_bridge C++ pybind11 module to provide a clean Python interface
for loading 3DGS .ply models and rendering RGB + depth images.
"""

import torch


class GSRenderer:
    """Wrapper around the C++ 3DGS rasterizer for inference rendering."""

    def __init__(self, ply_path: str, sh_degree: int = 0, device: str = "cuda:0"):
        self.device = device
        self.sh_degree = sh_degree
        self.ply_path = ply_path
        self._loaded = False

        from . import _gs_bridge
        self._model = _gs_bridge.GaussianModel()
        ok = self._model.load_ply(ply_path, sh_degree=sh_degree, device=device)
        if not ok:
            raise RuntimeError(f"Failed to load PLY file: {ply_path}")
        self._loaded = True

    def render(
        self,
        pose_cam2world: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        background: int = 0,
        packed: bool = True,
    ) -> dict:
        """
        Render the 3DGS model from one or multiple camera poses.

        Args:
            pose_cam2world: [B, 3, 4] camera-to-world extrinsics matrix.
                Each slice is [R | t] where R is 3x3 rotation and t is 3x1 translation.
                Uses OpenCV convention: camera looks along +Z, Y points down.
            fx, fy: Focal lengths in pixels.
            cx, cy: Principal point in pixels.
            width, height: Image dimensions.
            near_plane: Near clipping plane.
            far_plane: Far clipping plane.
            background: 0=black, 1=white, 2=random.
            packed: Whether to use packed gaussian representation.

        Returns:
            dict with keys:
                - "color": [B, H, W, 3] float32 tensor, RGB image.
                - "depth": [B, H, W, 1] float32 tensor, expected depth.
                - "alpha": [B, H, W, 1] float32 tensor, alpha channel.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_ply first.")

        if pose_cam2world.dim() == 2:
            pose_cam2world = pose_cam2world.unsqueeze(0)

        if pose_cam2world.device != torch.device(self.device):
            pose_cam2world = pose_cam2world.to(self.device)

        with torch.no_grad():
            result = self._model.render(
                pose_cam2world,
                float(fx), float(fy), float(cx), float(cy),
                int(width), int(height),
                float(near_plane), float(far_plane),
                int(background), bool(packed),
            )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded
