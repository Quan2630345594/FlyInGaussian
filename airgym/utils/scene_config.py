import os

import yaml


def _resolve_path(config_dir, value):
    if not value:
        return value
    value = os.path.expandvars(os.path.expanduser(value))
    if not os.path.isabs(value):
        value = os.path.join(config_dir, value)
    return os.path.abspath(value)


def load_scene_config(env_cfg, config_path):
    """Load a local 3DGS and collision-mesh scene into an environment config."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Scene config does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as stream:
        scene_cfg = yaml.safe_load(stream) or {}

    if not isinstance(scene_cfg, dict):
        raise ValueError("Scene config must contain a YAML mapping at its root.")
    if not hasattr(env_cfg, "asset_config"):
        raise ValueError("The selected task does not support scene assets.")

    config_dir = os.path.dirname(config_path)
    gs_section = scene_cfg.get("gs_model", {}) or {}
    collision_section = scene_cfg.get("collision_mesh", {}) or {}
    gs_model = dict(getattr(env_cfg.asset_config, "gs_model", {}) or {})
    gs_model.update(gs_section)
    collision_mesh = dict(
        getattr(env_cfg.asset_config, "collision_mesh", {}) or {}
    )
    collision_mesh.update(collision_section)
    environment = scene_cfg.get("environment", {}) or {}

    if not isinstance(environment, dict):
        raise ValueError("'environment' must be a YAML mapping.")

    if "gs_model" in scene_cfg and "enabled" not in gs_section:
        gs_model["enabled"] = True
    gs_model.setdefault("observation", "depth")
    gs_model.setdefault("render_width", 212)
    gs_model.setdefault("render_height", 120)
    gs_model.setdefault("render_batch_size", 16)
    gs_model.setdefault("far", 4.5)
    if "collision_mesh" in scene_cfg and "enabled" not in collision_section:
        collision_mesh["enabled"] = True
    gs_model["ply_path"] = _resolve_path(config_dir, gs_model.get("ply_path"))
    collision_mesh["mesh_path"] = _resolve_path(
        config_dir, collision_mesh.get("mesh_path")
    )

    if gs_model["enabled"] and not gs_model.get("ply_path"):
        raise ValueError("Enabled gs_model requires 'ply_path'.")
    if collision_mesh["enabled"] and not collision_mesh.get("mesh_path"):
        raise ValueError("Enabled collision_mesh requires 'mesh_path'.")

    for label, path in (
        ("3DGS PLY", gs_model.get("ply_path")),
        ("collision mesh", collision_mesh.get("mesh_path")),
    ):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    scale = collision_mesh.get("scale", 1.0)
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("collision_mesh.scale must be a positive number.")

    for key in ("rotation_rpy_deg", "translation"):
        value = collision_mesh.get(key, [0.0, 0.0, 0.0])
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"collision_mesh.{key} must contain three numbers.")

    observation = gs_model.get("observation", "depth")
    if observation not in ("depth", "rgb"):
        raise ValueError("gs_model.observation must be 'depth' or 'rgb'.")
    if gs_model.get("far", 0) <= 0:
        raise ValueError("gs_model.far must be positive.")
    if gs_model.get("render_width", 0) <= 0 or gs_model.get("render_height", 0) <= 0:
        raise ValueError("3DGS render dimensions must be positive.")
    if gs_model.get("render_batch_size", 0) <= 0:
        raise ValueError("gs_model.render_batch_size must be positive.")
    if collision_mesh.get("replicate_per_env", False):
        raise ValueError(
            "replicate_per_env is not supported by the current reset logic. "
            "Use the default shared world-space scene."
        )

    env_cfg.asset_config.gs_model = gs_model
    env_cfg.asset_config.collision_mesh = collision_mesh

    if environment.get("disable_random_obstacles", True):
        env_cfg.asset_config.include_group_asset = {}
    if "create_ground_plane" in environment:
        env_cfg.env.create_ground_plane = bool(environment["create_ground_plane"])

    env_cfg.scene_config_path = config_path
    return env_cfg
