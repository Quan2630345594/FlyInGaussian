# Third-Party Notices

FlyInGaussian contains or depends on software from other projects. This file is informational and does not replace the license text shipped by each project.

## AirGym

- Project: https://github.com/emNavi/AirGym
- License: BSD-3-Clause
- Use: simulator environments, assets, controllers integration and training framework

The original AirGym and Aerial Gym copyright notices are retained in the root `LICENSE` and relevant source headers.

## Aerial Gym Simulator

- Project: https://github.com/ntnu-arl/aerial_gym_simulator
- License: BSD-3-Clause
- Use: upstream foundation of AirGym and portions of the task infrastructure

## rl_games

- Project: https://github.com/Denys88/rl_games
- Version lineage: 1.6.1
- License: MIT
- Use: PPO and actor-critic implementation under `lib/`

The MIT license text is stored at `LICENSES/rl_games-MIT.txt`.

## GS-SDF

- Project: https://github.com/hku-mars/GS-SDF
- License: GPL-2.0
- Use: Gaussian rasterization source dependency and SDF mesh reconstruction workflow
- Location: `submodules/GS-SDF`

GS-SDF is included as a Git submodule and retains its own license and nested dependency notices. Initialize it with `git submodule update --init --recursive`.

## NVIDIA Isaac Gym

- Project: NVIDIA Isaac Gym Preview 4
- Use: physics simulation, GPU tensor API and camera API

Isaac Gym is an external NVIDIA distribution and is not redistributed by this repository. Retained BSD-licensed NVIDIA source headers and notices must not be removed.

## rlPx4Controller

- Project: https://github.com/emNavi/rlPx4Controller
- Use: parallel quadrotor control loops

`rlPx4Controller` is installed as an external Python package and is not vendored here.
