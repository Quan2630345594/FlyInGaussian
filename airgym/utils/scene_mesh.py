import math

import numpy as np


def rotation_matrix_from_rpy_deg(rotation_rpy_deg):
    roll, pitch, yaw = np.deg2rad(rotation_rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def load_collision_mesh(config):
    """Load and transform a triangle mesh from scene coordinates to simulation coordinates."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Collision meshes require trimesh. Install the project dependencies first."
        ) from exc

    mesh = trimesh.load(config["mesh_path"], force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {config['mesh_path']}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Collision mesh contains no valid vertices.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Collision mesh contains no triangular faces.")

    scale = float(config.get("scale", 1.0))
    rotation = rotation_matrix_from_rpy_deg(
        config.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
    )
    translation = np.asarray(
        config.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32
    )
    vertices = (vertices * scale) @ rotation.T + translation
    return np.ascontiguousarray(vertices), np.ascontiguousarray(triangles), rotation
