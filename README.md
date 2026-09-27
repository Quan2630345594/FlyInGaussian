# FlyInGaussian

FlyInGaussian is a quadrotor reinforcement-learning simulator that combines NVIDIA Isaac Gym physics with 3D Gaussian Splatting (3DGS) observations. A GS-SDF Gaussian model supplies RGB or depth images, while the triangle mesh reconstructed from the same GS-SDF run supplies the static collision geometry.

The project keeps the PPO training pipeline and the Hovering, Balloon, Tracking, Avoid, Planning, Customized, DepthGen and multi-agent Planning environments inherited from AirGym. The Gaussian scene integration is optional, so the original non-Gaussian tasks remain available.

## Scene Pipeline

```text
GS-SDF training output
  model/gs.ply                 -> batched RGB/depth rendering
  gs_log/mesh_<resolution>.ply -> Isaac Gym triangle-mesh collision
                                      |
drone state -> camera pose -> policy image + state -> PPO action
                                      |
                             Isaac Gym dynamics/contact
```

The same scale, roll/pitch/yaw and translation are applied to rendering and collision geometry. Scene paths are supplied through a local YAML file and are not committed to Git.

## Requirements

- Ubuntu 20.04
- NVIDIA GPU and CUDA 11.8 or a compatible CUDA toolchain
- Python 3.8
- PyTorch 2.4.1
- NVIDIA Isaac Gym Preview 4
- CMake 3.18 or newer
- `rlPx4Controller`, PyTorch3D, OpenCV and trimesh

Isaac Gym is distributed separately by NVIDIA and is not included in this repository.

## Installation

Clone all submodules:

```bash
git clone --recursive https://github.com/Quan2630345594/FlyInGaussian.git
cd FlyInGaussian
```

Install system prerequisites such as `libeigen3-dev`, a CUDA compiler and CMake first. The setup helper creates or reuses the Conda environment, installs Python dependencies and installs `rlPx4Controller`. If Isaac Gym has already been extracted to `~/isaacgym`, it is installed automatically; otherwise the script prints the remaining manual step.

```bash
chmod +x configuration.sh
./configuration.sh
conda activate flyingaussian
```

Set `ISAAC_GYM_ROOT` or `RLPX4_ROOT` when those projects are stored elsewhere. The script never removes an existing Conda environment.

To initialize submodules in an existing checkout:

```bash
git submodule update --init --recursive
```

## Build The Renderer

The renderer extension is compiled against the active Python and PyTorch environment. No prebuilt `.so` is stored in Git.

```bash
conda activate flyingaussian
cmake -S airgym/gs_renderer -B build/gs_renderer \
  -DPython3_EXECUTABLE="$(which python3)"
cmake --build build/gs_renderer -j"$(nproc)"
```

The build writes `airgym/gs_renderer/_gs_bridge.so`, which is ignored by Git.

Test a Gaussian model without starting Isaac Gym:

```bash
python3 airgym/gs_renderer/test_render.py \
  --ply /path/to/gs-sdf/output/run/model/gs.ply \
  --sh-degree 0 --no-viz
```

## Export A Collision Mesh

A standalone `gs.ply` does not contain a triangle mesh. Mesh export requires the complete GS-SDF run, including `local_map_checkpoint.pt`, `as_occ_prior.ply`, the run configuration and the original data path.

Build GS-SDF according to its documentation, then open the complete run directory:

```bash
./submodules/GS-SDF/build/neural_mapping_node view \
  /path/to/gs-sdf/output/run
```

In the GS-SDF terminal, enter:

```text
m 0.02
```

The mesh is normally written to:

```text
/path/to/gs-sdf/output/run/gs_log/mesh_0.020000.ply
```

If mesh culling is enabled, `mesh_culled_0.020000.ply` is also produced. A coarser mesh or a decimated copy is recommended for simulation.

In the non-ROS GS-SDF build, the interactive resolution argument may only affect the filename. Set `export_resolution` in the GS-SDF scene configuration when an exact resolution is required.

Replica, FAST-LIVO and COLMAP GS-SDF outputs normally keep the mesh and Gaussian model in the same coordinate system. NeuralRGBD data may apply an `(x, y, z) -> (x, z, -y)` mesh conversion and must be calibrated before training.

## Scene Configuration

Copy `configs/scene.example.yaml` outside the repository or to a file ending in `.local.yaml`, then edit the model paths and transform:

```bash
cp configs/scene.example.yaml configs/my_scene.local.yaml
```

Important fields:

| Field | Meaning |
| --- | --- |
| `gs_model.ply_path` | GS-SDF `model/gs.ply` |
| `gs_model.observation` | `depth` for Planning; `rgb` is available only to custom three-channel policies |
| `gs_model.render_batch_size` | Number of environments rendered in one rasterizer call |
| `collision_mesh.mesh_path` | GS-SDF triangle mesh PLY |
| `collision_mesh.scale` | Uniform scene scale |
| `collision_mesh.rotation_rpy_deg` | Scene rotation in XYZ roll/pitch/yaw degrees |
| `collision_mesh.translation` | Scene translation in simulation metres |
The built-in Planning task resets all parallel actors into the same world-space scene and isolates them with collision groups. FlyInGaussian therefore creates one shared static triangle mesh instead of duplicating a large mesh for every environment.

## Training

First verify the original non-image training path:

```bash
python3 scripts/runner.py \
  --task hovering --ctl_mode rate --headless --num_envs 4
```

Train Planning with the Gaussian scene and mesh collision:

```bash
python3 scripts/runner.py \
  --task planning --ctl_mode rate --headless --num_envs 4 \
  --scene-config configs/my_scene.local.yaml
```

Start with 4 to 32 environments. Gaussian rasterization and detailed triangle meshes have substantially higher GPU memory requirements than the original primitive-based Planning task.

Checkpoints and TensorBoard logs are written below `runs/`.

## Playing A Checkpoint

```bash
python3 scripts/runner.py \
  --play --task planning --ctl_mode rate --num_envs 4 \
  --checkpoint /path/to/checkpoint.pth \
  --scene-config configs/my_scene.local.yaml
```

The legacy pretrained Planning checkpoint expects normalized single-channel depth with shape `[1, 212, 120]`. FlyInGaussian keeps this observation contract when `gs_model.observation: depth` is selected.

## Other Tasks

The inherited tasks remain registered:

```bash
python3 airgym/scripts/example.py --task hovering --ctl_mode pos --num_envs 4
python3 airgym/scripts/ma_example.py --task maplanning --ctl_mode pos --num_envs 4
```

Supported control modes are `pos`, `vel`, `atti`, `rate` and `prop`.

## Model Data

Gaussian models, reconstructed meshes, datasets, local scene YAML files, extension binaries and training runs are intentionally excluded from Git. Verify that a Gaussian model and mesh are legally redistributable before publishing them separately.

FlyInGaussian is intended to run from a source checkout installed with `pip install -e .`; wheel packaging of Isaac Gym assets is not supported.

## Limitations

- Isaac Gym triangle meshes are static and simulation-global.
- Very detailed meshes should be decimated before use.
- Contact-force reporting against triangle meshes must be validated for each PhysX configuration.
- The GS-SDF renderer requires CUDA and a locally compiled extension.
- Full simulation tests require Isaac Gym, which cannot be installed from PyPI.

## Origin And Attribution

FlyInGaussian is a derivative project, not an official AirGym or GS-SDF release.

The simulator and training code are based on [emNavi/AirGym](https://github.com/emNavi/AirGym), which itself acknowledges and derives from [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator). The PPO implementation under `lib/` is derived from `rl_games` 1.6.1. Gaussian rendering and mesh reconstruction use [hku-mars/GS-SDF](https://github.com/hku-mars/GS-SDF) as a pinned submodule.

The original BSD-3-Clause copyright and license are retained in `LICENSE`. Additional notices are listed in `THIRD_PARTY_NOTICES.md`; each submodule also retains its own license.
