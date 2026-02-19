"""Utility functions for configuration, I/O, and paths.

Paths are imported eagerly (pure stdlib, no heavy deps).
Config, I/O, and logging are lazy-loaded via __getattr__ so that
scripts which only need paths (e.g. TRILLsson extraction in a
TensorFlow-only venv) don't pull in torch.
"""

# --- Eager: paths (stdlib only, no torch/yaml) ---
from .paths import (
    build_audio_path,
    get_asvspoof5_root,
    get_aug_cache_dir,
    get_features_dir,
    get_manifest_path,
    get_manifests_dir,
    get_project_root,
    get_run_dir,
    get_runs_dir,
)

# --- Lazy: config, io, logging (require torch) ---
_CONFIG_ATTRS = {"get_device", "load_config", "merge_configs", "set_seed"}
_IO_ATTRS = {
    "load_checkpoint",
    "load_metrics",
    "save_checkpoint",
    "save_config",
    "save_metrics",
}
_LOGGING_ATTRS = {
    "ExperimentLogger",
    "check_for_nan_grads",
    "compute_grad_norm",
    "get_experiment_context",
    "get_git_info",
    "get_gpu_memory_usage",
    "get_gpu_utilization",
    "setup_logging",
}


def __getattr__(name: str):
    if name in _CONFIG_ATTRS:
        from . import config

        return getattr(config, name)
    if name in _IO_ATTRS:
        from . import io

        return getattr(io, name)
    if name in _LOGGING_ATTRS:
        from . import logging

        return getattr(logging, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (lazy)
    "load_config",
    "merge_configs",
    "set_seed",
    "get_device",
    # I/O (lazy)
    "save_checkpoint",
    "load_checkpoint",
    "save_metrics",
    "load_metrics",
    "save_config",
    # Paths (eager)
    "get_project_root",
    "get_asvspoof5_root",
    "get_aug_cache_dir",
    "get_manifests_dir",
    "get_runs_dir",
    "get_run_dir",
    "get_manifest_path",
    "get_features_dir",
    "build_audio_path",
    # Logging (lazy)
    "setup_logging",
    "get_experiment_context",
    "get_git_info",
    "ExperimentLogger",
    "compute_grad_norm",
    "check_for_nan_grads",
    "get_gpu_memory_usage",
    "get_gpu_utilization",
]
