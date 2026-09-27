import importlib.util
import os
import tempfile
import unittest

import numpy as np


MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "airgym", "utils", "scene_mesh.py"
)
SPEC = importlib.util.spec_from_file_location("scene_mesh", MODULE_PATH)
SCENE_MESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENE_MESH)


class SceneMeshTest(unittest.TestCase):
    def test_rotation_matrix_uses_xyz_rpy(self):
        rotation = SCENE_MESH.rotation_matrix_from_rpy_deg([0.0, 0.0, 90.0])
        transformed = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(transformed, [0.0, 1.0, 0.0], atol=1e-6)

    def test_loads_and_transforms_ply_mesh(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "triangle.ply")
            trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
                process=False,
            ).export(path)
            vertices, triangles, _ = SCENE_MESH.load_collision_mesh(
                {
                    "mesh_path": path,
                    "scale": 2.0,
                    "rotation_rpy_deg": [0.0, 0.0, 90.0],
                    "translation": [1.0, 2.0, 3.0],
                }
            )

            np.testing.assert_allclose(vertices[0], [1.0, 2.0, 3.0], atol=1e-6)
            self.assertEqual(triangles.shape, (1, 3))
            self.assertEqual(vertices.dtype, np.float32)
            self.assertEqual(triangles.dtype, np.uint32)


if __name__ == "__main__":
    unittest.main()
