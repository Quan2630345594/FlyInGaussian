# FlyInGaussian

FlyInGaussian 是一个面向四旋翼强化学习的 GPU 并行仿真项目，将 NVIDIA Isaac Gym 的动力学与 3D Gaussian Splatting（3DGS）场景结合：

- 外部训练好的 3DGS PLY 模型提供 RGB 或深度观测。
- 与模型配准的三角网格提供静态碰撞几何。
- 保留 AirGym 的 PPO 训练流程以及 Hovering、Balloon、Tracking、Avoid、Planning 等任务。
- 只有传入 `--scene-config` 时才启用高斯渲染和网格碰撞，原有任务仍可独立运行。

## 环境要求

- Ubuntu 20.04
- NVIDIA GPU、CUDA 11.8
- Conda、Python 3.8
- NVIDIA Isaac Gym Preview 4
- CMake 3.18 或更高版本

Isaac Gym 需要从 NVIDIA 单独下载，本仓库不包含其安装包。

## 安装

克隆仓库及全部子模块：

```bash
git clone --recursive https://github.com/Quan2630345594/FlyInGaussian.git
cd FlyInGaussian
```

如果已经克隆过仓库：

```bash
git submodule update --init --recursive
```

`submodules/GS-SDF` 仅提供 3DGS 渲染器所需的接口、头文件和 CUDA 实现，本项目不会使用该子模块训练模型。

安装脚本会创建或复用 `gaussiangym` Conda 环境，并安装 PyTorch、PyTorch3D、`rlPx4Controller` 和本项目。默认从 `~/isaacgym` 安装 Isaac Gym，也可以指定其他位置：

```bash
chmod +x configuration.sh
ISAAC_GYM_ROOT=/path/to/isaacgym ./configuration.sh
conda activate gaussiangym
```

脚本不会删除已有 Conda 环境。也可以通过 `RLPX4_ROOT` 指定 `rlPx4Controller` 的目录。

## 编译 3DGS 渲染器

渲染器需要在目标机器上使用当前 Python、PyTorch 和 CUDA 环境编译：

```bash
conda activate gaussiangym
cmake -S airgym/gs_renderer -B build/gs_renderer \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build/gs_renderer -j"$(nproc)"
```

编译结果为 `airgym/gs_renderer/_gs_bridge.so`。可在不启动 Isaac Gym 的情况下测试渲染：

```bash
python airgym/gs_renderer/test_render.py \
  --ply models/my_scene/gs.ply \
  --sh-degree 0 --no-viz
```

## 准备场景模型

请在其他项目或机器上完成 3DGS 训练和网格重建，再将配准好的模型文件放入本项目的 `models/` 目录。例如：

```text
models/my_scene/
├── gs.ply      # 3DGS 视觉模型
└── mesh.ply    # Isaac Gym 碰撞网格
```

`models/` 已被 Git 忽略，不会提交模型数据。用于仿真前建议删除漂浮的小型三角面组件，并对过密网格进行简化；孤立顶点不会碰撞，但所有有效三角面都会进入 PhysX。

## 场景配置

复制配置模板并修改本地模型路径：

```bash
cp configs/scene.example.yaml configs/my_scene.local.yaml
```

模板中的相对路径以 YAML 文件所在目录为基准，因此 `configs/my_scene.local.yaml` 可以使用 `../models/my_scene/gs.ply` 和 `../models/my_scene/mesh.ply`。`.local.yaml` 文件同样不会提交到 Git。主要配置项如下：

| 配置项 | 说明 |
| --- | --- |
| `gs_model.ply_path` | 外部训练好的 3DGS PLY 模型 |
| `gs_model.observation` | Planning 使用 `depth`；`rgb` 仅适用于自定义三通道策略 |
| `gs_model.render_batch_size` | 单次批量渲染的环境数量 |
| `collision_mesh.mesh_path` | 用于 PhysX 的三角网格 |
| `collision_mesh.scale` | 场景统一缩放比例 |
| `collision_mesh.rotation_rpy_deg` | XYZ 顺序的滚转、俯仰、偏航角，单位为度 |
| `collision_mesh.translation` | 场景在仿真坐标系中的平移，单位为米 |

缩放、旋转和平移会同时用于视觉与碰撞坐标对齐。当前碰撞网格是全局静态场景，不会为每个并行环境复制一份。

## 训练与推理

先验证不使用 3DGS 的基础任务：

```bash
python scripts/runner.py \
  --train --task hovering --ctl_mode rate --headless --num_envs 4
```

使用 3DGS 深度观测和网格碰撞训练 Planning：

```bash
python scripts/runner.py \
  --train --task planning --ctl_mode rate --headless --num_envs 4 \
  --scene-config configs/my_scene.local.yaml
```

加载 checkpoint 进行推理：

```bash
python scripts/runner.py \
  --play --task planning --ctl_mode rate --num_envs 4 \
  --checkpoint /path/to/checkpoint.pth \
  --scene-config configs/my_scene.local.yaml
```

支持的控制模式为 `pos`、`vel`、`atti`、`rate` 和 `prop`。可用任务包括 `hovering`、`customized`、`balloon`、`avoid`、`tracking`、`planning`、`maplanning` 和 `depthgen`。训练日志与 checkpoint 默认写入 `runs/`。

建议先使用 4 至 32 个环境验证场景。3DGS 渲染和高精度三角网格会显著增加显存及 PhysX 开销。

## 注意事项

- Planning 的内置策略要求单通道深度观测，默认尺寸为 `[1, 212, 120]`。
