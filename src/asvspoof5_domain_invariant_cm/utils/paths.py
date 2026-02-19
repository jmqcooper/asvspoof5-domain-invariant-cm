"""Path utilities for project directories."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_asvspoof5_root() -> Path:
    """Get ASVspoof5 data root directory.

    Uses ASVSPOOF5_ROOT environment variable as the single source of truth.

    Returns:
        Path to ASVspoof5 data root.

    Raises:
        EnvironmentError: If ASVSPOOF5_ROOT is not set.
    """
    env_path = os.environ.get("ASVSPOOF5_ROOT")
    if env_path:
        return Path(env_path)

    raise EnvironmentError(
        "ASVSPOOF5_ROOT environment variable is not set. "
        "Please set it to your ASVspoof5 data directory, e.g.:\n"
        "  export ASVSPOOF5_ROOT=/path/to/asvspoof5\n\n"
        "Expected structure:\n"
        "  $ASVSPOOF5_ROOT/\n"
        "    ASVspoof5_protocols/\n"
        "    flac_T/\n"
        "    flac_D/\n"
        "    flac_E_eval/"
    )


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd()


def get_manifests_dir() -> Path:
    """Get manifests directory.

    Returns:
        Path to manifests directory (created if needed).
    """
    manifests_dir = get_project_root() / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return manifests_dir


def get_runs_dir() -> Path:
    """Get runs directory for experiment outputs.

    Uses RUNS_DIR environment variable if set, otherwise defaults to project_root/runs.
    
    Set RUNS_DIR for persistent storage on HPC systems, e.g.:
        export RUNS_DIR=/projects/prjs1904/runs

    Returns:
        Path to runs directory (created if needed).
    """
    env_path = os.environ.get("RUNS_DIR")
    if env_path:
        runs_dir = Path(env_path)
        if not runs_dir.parent.exists():
            raise EnvironmentError(
                f"RUNS_DIR parent directory does not exist: {runs_dir.parent}"
            )
    else:
        runs_dir = get_project_root() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_aug_cache_dir() -> Optional[Path]:
    """Get augmentation cache directory.

    Uses AUGMENTATION_CACHE_DIR environment variable if set.
    
    Set for pre-computed codec augmentations on HPC systems, e.g.:
        export AUGMENTATION_CACHE_DIR=/projects/prjs1904/data/asvspoof5_augmented_cache

    Returns:
        Path to cache directory, or None if not configured.
    """
    env_path = os.environ.get("AUGMENTATION_CACHE_DIR")
    if env_path:
        cache_dir = Path(env_path)
        if cache_dir.exists():
            return cache_dir
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"AUGMENTATION_CACHE_DIR set but path does not exist: {cache_dir}"
            )
    return None


def get_run_dir(name: str = None) -> Path:
    """Get or create a run directory.

    Args:
        name: Experiment name (auto-generated if None).

    Returns:
        Path to run directory.
    """
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}"

    run_dir = get_runs_dir() / name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_manifest_path(split: str) -> Path:
    """Get path to manifest file for a split.

    Args:
        split: Data split ('train', 'dev', 'eval').

    Returns:
        Path to manifest file.
    """
    return get_manifests_dir() / f"{split}.parquet"


def get_audio_dir_for_prefix(prefix: str) -> str:
    """Get audio subdirectory for a filename prefix.

    Args:
        prefix: Filename prefix ('T_', 'D_', 'E_').

    Returns:
        Audio subdirectory name.

    Raises:
        ValueError: If prefix is unknown.
    """
    prefix_map = {
        "T_": "flac_T",
        "D_": "flac_D",
        "E_": "flac_E_eval",
    }

    if prefix not in prefix_map:
        raise ValueError(
            f"Unknown filename prefix: {prefix}. "
            f"Expected one of: {list(prefix_map.keys())}"
        )

    return prefix_map[prefix]


def build_audio_path(flac_filename: str, asvspoof5_root: Path = None) -> Path:
    """Build absolute audio path from FLAC filename.

    Args:
        flac_filename: FLAC filename (e.g., 'T_0001' or 'T_0001.flac').
        asvspoof5_root: ASVspoof5 root (uses env var if None).

    Returns:
        Absolute path to audio file.
    """
    if asvspoof5_root is None:
        asvspoof5_root = get_asvspoof5_root()

    # Add .flac extension if missing
    if not flac_filename.endswith(".flac"):
        flac_filename = f"{flac_filename}.flac"

    prefix = flac_filename[:2]
    audio_dir = get_audio_dir_for_prefix(prefix)

    return asvspoof5_root / audio_dir / flac_filename


def get_features_dir(feature_type: str = "trillsson") -> Path:
    """Get features directory for cached embeddings.

    Args:
        feature_type: Feature type ('trillsson', 'lfcc', etc.).

    Returns:
        Path to features directory (created if needed).
    """
    features_dir = get_project_root() / "data" / "features" / feature_type
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir
