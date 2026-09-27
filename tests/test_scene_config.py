import importlib.util
import os
import tempfile
import unittest


MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "airgym", "utils", "scene_config.py"
)
SPEC = importlib.util.spec_from_file_location("scene_config", MODULE_PATH)
SCENE_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENE_CONFIG)


class Object:
    pass


class SceneConfigTest(unittest.TestCase):
    def test_loads_relative_paths_and_disables_random_obstacles(self):
        with tempfile.TemporaryDirectory() as directory:
            ply_path = os.path.join(directory, "scene.ply")
            mesh_path = os.path.join(directory, "mesh.ply")
            config_path = os.path.join(directory, "scene.yaml")
            for path in (ply_path, mesh_path):
                with open(path, "wb"):
                    pass
            with open(config_path, "w", encoding="utf-8") as stream:
                stream.write(
                    "gs_model:\n"
                    "  ply_path: scene.ply\n"
                    "collision_mesh:\n"
                    "  mesh_path: mesh.ply\n"
                    "environment:\n"
                    "  create_ground_plane: false\n"
                )

            cfg = Object()
            cfg.asset_config = Object()
            cfg.asset_config.gs_model = {"enabled": False}
            cfg.asset_config.collision_mesh = {"enabled": False}
            cfg.asset_config.include_group_asset = {"thin": {"num_assets": 1}}
            cfg.env = Object()
            cfg.env.create_ground_plane = True

            SCENE_CONFIG.load_scene_config(cfg, config_path)

            self.assertEqual(cfg.asset_config.gs_model["ply_path"], ply_path)
            self.assertEqual(cfg.asset_config.collision_mesh["mesh_path"], mesh_path)
            self.assertTrue(cfg.asset_config.gs_model["enabled"])
            self.assertTrue(cfg.asset_config.collision_mesh["enabled"])
            self.assertEqual(cfg.asset_config.include_group_asset, {})
            self.assertFalse(cfg.env.create_ground_plane)

    def test_requires_enabled_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "scene.yaml")
            with open(config_path, "w", encoding="utf-8") as stream:
                stream.write("gs_model: {}\ncollision_mesh: {}\n")
            cfg = Object()
            cfg.asset_config = Object()
            cfg.asset_config.include_group_asset = {}
            cfg.env = Object()

            with self.assertRaisesRegex(ValueError, "ply_path"):
                SCENE_CONFIG.load_scene_config(cfg, config_path)


if __name__ == "__main__":
    unittest.main()
