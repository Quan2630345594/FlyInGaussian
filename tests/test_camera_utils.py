import importlib.util
import os
import unittest

import torch


MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "airgym", "gs_renderer", "camera_utils.py"
)
SPEC = importlib.util.spec_from_file_location("camera_utils", MODULE_PATH)
CAMERA_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMERA_UTILS)


class CameraUtilsTest(unittest.TestCase):
    def test_identity_drone_camera_axes_and_offset(self):
        position = torch.tensor([[1.0, 2.0, 3.0]])
        quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        offset = torch.tensor([0.15, 0.0, 0.1])

        pose = CAMERA_UTILS.compute_camera_pose(position, quaternion, offset)

        torch.testing.assert_close(pose[0, :, 3], torch.tensor([1.15, 2.0, 3.1]))
        torch.testing.assert_close(
            pose[0, :, :3],
            torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        )

    def test_depth_observation_fills_transparent_pixels_and_uses_cwh(self):
        color = torch.zeros((1, 2, 3, 3))
        depth = torch.ones((1, 2, 3, 1))
        alpha = torch.ones((1, 2, 3, 1))
        alpha[:, 0, 0] = 0.0

        observation, _, metric_depth = CAMERA_UTILS.prepare_policy_observation(
            color, depth, alpha, "depth", 4.0, 2.0, 0.01
        )

        self.assertEqual(observation.shape, (1, 1, 3, 2))
        self.assertEqual(metric_depth[0, 0, 0, 0].item(), 4.0)
        self.assertEqual(observation[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(observation[0, 0, 1, 0].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
