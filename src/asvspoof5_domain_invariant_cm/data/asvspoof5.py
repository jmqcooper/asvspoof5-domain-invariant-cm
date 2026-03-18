"""ASVspoof 5 dataset loading and processing.

Important: Protocol files are whitespace-separated despite .tsv extension.
Label convention: bonafide=0, spoof=1.
"""

import io
import json
import logging
import os
import random
import tarfile
import threading
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import crop_or_pad, load_waveform

# Column names for Track 1 protocol files
PROTOCOL_COLUMNS = [
    "speaker_id",
    "flac_file",
    "gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "tmp",
]

# Label mapping: bonafide=0, spoof=1
KEY_TO_LABEL = {"bonafide": 0, "spoof": 1}

# Explicit label constants for clarity
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1


def normalize_domain_value(value: str, is_codec_q: bool = False) -> str:
    """Normalize domain value, treating '-' and '0' (for CODEC_Q) as 'NONE'.

    In ASVspoof5 protocol files:
    - Train/dev use "-" for uncoded samples (both CODEC and CODEC_Q)
    - Eval uses "-" for CODEC but "0" for CODEC_Q when uncoded

    Both "-" and "0" (for CODEC_Q only) should map to "NONE".

    Args:
        value: Raw domain value.
        is_codec_q: If True, also treat "0" as uncoded/NONE.

    Returns:
        Normalized value.
    """
    if value == "-" or value is None or (isinstance(value, float) and pd.isna(value)):
        return "NONE"
    # For CODEC_Q, "0" means uncoded (used in eval split)
    if is_codec_q and str(value) == "0":
        return "NONE"
    return str(value)


def load_protocol(path: Path) -> pd.DataFrame:
    """Load ASVspoof 5 protocol file.

    ASVspoof 5 protocol files are whitespace-separated (not tab-separated)
    despite the .tsv extension.

    Args:
        path: Path to protocol file.

    Returns:
        DataFrame with protocol data.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PROTOCOL_COLUMNS,
        dtype=str,
    )
    return df


def get_codec_seed_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """Group samples by codec_seed for stratified splitting.

    This prevents leakage between coded variants of the same utterance.

    Args:
        df: Protocol DataFrame.

    Returns:
        Dictionary mapping codec_seed to list of indices.
    """
    return df.groupby("codec_seed").indices


def build_vocab(values: pd.Series) -> dict[str, int]:
    """Build vocabulary mapping from unique values.

    Args:
        values: Series of values to encode.

    Returns:
        Dictionary mapping value to integer id.
    """
    unique_values = sorted(values.unique())
    return {v: i for i, v in enumerate(unique_values)}


def save_vocab(vocab: dict, path: Path) -> None:
    """Save vocabulary to JSON file.

    Args:
        vocab: Vocabulary dictionary.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)


def load_vocab(path: Path) -> dict[str, int]:
    """Load vocabulary from JSON file.

    Args:
        path: Path to vocabulary file.

    Returns:
        Vocabulary dictionary.
    """
    with open(path) as f:
        return json.load(f)


class ASVspoof5Dataset(Dataset):
    """PyTorch Dataset for ASVspoof 5 Track 1.

    Args:
        manifest_path: Path to manifest (parquet) file.
        codec_vocab: CODEC vocabulary (value -> id).
        codec_q_vocab: CODEC_Q vocabulary (value -> id).
        max_duration_sec: Maximum audio duration in seconds.
        sample_rate: Expected sample rate.
        mode: 'train' for random crop, 'eval' for center crop.
        return_raw_length: If True, return original length before crop/pad.
        augmentor: Optional CodecAugmentor for synthetic domain augmentation.
        use_synthetic_labels: If True, return augmented domain labels (y_codec_aug, y_codec_q_aug).
    """

    def __init__(
        self,
        manifest_path: Path,
        codec_vocab: Optional[dict[str, int]] = None,
        codec_q_vocab: Optional[dict[str, int]] = None,
        max_duration_sec: float = 6.0,
        sample_rate: int = 16000,
        mode: Literal["train", "eval"] = "eval",
        return_raw_length: bool = False,
        augmentor: Optional["CodecAugmentor"] = None,
        use_synthetic_labels: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.max_samples = int(max_duration_sec * sample_rate)
        self.sample_rate = sample_rate
        self.mode = mode
        self.return_raw_length = return_raw_length
        self.augmentor = augmentor
        self.use_synthetic_labels = use_synthetic_labels

        # Pre-computed augmentation cache
        self._aug_cache_manifest = None
        self._aug_cache_index = None  # flac_stem → list of augmented file info
        if augmentor is not None and augmentor.cache_dir is not None:
            self._load_aug_cache(augmentor.cache_dir)

        # Load manifest
        self.df = pd.read_parquet(self.manifest_path)

        # Use provided vocabs or build from manifest
        if codec_vocab is not None:
            self.codec_vocab = codec_vocab
        else:
            self.codec_vocab = build_vocab(self.df["codec"])

        if codec_q_vocab is not None:
            self.codec_q_vocab = codec_q_vocab
        else:
            self.codec_q_vocab = build_vocab(self.df["codec_q"])

    def _load_aug_cache(self, cache_dir: Path) -> None:
        """Load pre-computed augmentation manifest for fast cache lookup.

        Supports two cache formats:
        - **Individual files** (version 1 / format="individual"): each augmentation
          is a separate file on disk. Direct file reads.
        - **Tar shards** (version 2 / format="tar"): augmentations are packed into
          tar files. Each DataLoader worker lazily opens its own TarFile handles
          for thread-safety.

        The manifest includes a ``codec_vocab`` that was built at pre-compute
        time. We validate it against the augmentor's live vocab to catch
        divergence early (e.g., codecs added/removed between pre-compute and
        training).
        """
        manifest_path = Path(cache_dir) / "manifest.json"
        if not manifest_path.exists():
            logger.debug(
                "No manifest.json in cache_dir %s — falling back to on-the-fly augmentation",
                cache_dir,
            )
            return

        with open(manifest_path) as f:
            self._aug_cache_manifest = json.load(f)

        # Validate codec vocab
        manifest_vocab = self._aug_cache_manifest.get("codec_vocab")
        if manifest_vocab is not None and self.augmentor is not None:
            live_vocab = self.augmentor.codec_vocab
            if manifest_vocab != live_vocab:
                logger.warning(
                    "Codec vocab mismatch between pre-computed manifest and "
                    "augmentor! manifest=%s, augmentor=%s. Domain label IDs "
                    "may be inconsistent. Re-run precompute_augmentations.py "
                    "with the same codec list.",
                    manifest_vocab,
                    live_vocab,
                )

        # Detect format
        self._aug_cache_format = self._aug_cache_manifest.get("format", "individual")

        # Build index: flac_stem → list of augmented file info
        self._aug_cache_index = {}
        for entry in self._aug_cache_manifest.get("entries", []):
            stem = entry["flac_stem"]
            self._aug_cache_index[stem] = entry["augmented_files"]

        # For tar format: build tar member offset index for fast random access.
        # We don't open tar files here — each DataLoader worker will lazily open
        # its own handles (TarFile is not thread-safe).
        self._tar_handles: dict[str, tarfile.TarFile] = {}  # per-thread lazy handles
        self._tar_handle_lock = threading.Lock()

        logger.info(
            "Loaded pre-computed augmentation cache (%s format): %d files, %d augmentations/file",
            self._aug_cache_format,
            len(self._aug_cache_index),
            self._aug_cache_manifest.get("augmentations_per_file", 0),
        )

    def _get_tar_handle(self, tar_path: str) -> tarfile.TarFile:
        """Get a TarFile handle for the current thread/worker.

        Each thread (DataLoader worker) gets its own TarFile handle since
        TarFile is not thread-safe. Handles are lazily opened on first access.

        Args:
            tar_path: Path to the tar file.

        Returns:
            Open TarFile handle for reading.
        """
        # Use (thread_id, tar_path) as key for per-thread handles
        tid = threading.get_ident()
        key = f"{tid}:{tar_path}"

        if key not in self._tar_handles:
            self._tar_handles[key] = tarfile.open(tar_path, "r")

        return self._tar_handles[key]

    def __len__(self) -> int:
        return len(self.df)

    def _get_cached_augmentation(
        self, audio_path: str, flac_file: str
    ) -> Optional[tuple["torch.Tensor", str, int]]:
        """Try to load a random pre-computed augmentation from cache.

        Supports both individual-file and tar-shard cache formats.

        Args:
            audio_path: Original audio file path.
            flac_file: FLAC filename (e.g., 'T_0001' or 'T_0001.flac').

        Returns:
            Tuple of (waveform, codec_name, quality) or None if cache miss.
        """
        if self._aug_cache_index is None:
            return None

        stem = Path(flac_file).stem

        aug_files = self._aug_cache_index.get(stem)
        if not aug_files:
            return None

        # Randomly pick one of the pre-computed augmentations
        choice = random.choice(aug_files)

        try:
            if self._aug_cache_format == "tar" and "tar_path" in choice:
                # Tar-shard mode: read from tar file
                tar_path = choice["tar_path"]
                internal_path = choice["internal_path"]
                tf = self._get_tar_handle(tar_path)
                member = tf.getmember(internal_path)
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    logger.debug("Cache miss (cannot extract): %s:%s", tar_path, internal_path)
                    return None
                audio_bytes = fileobj.read()
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                waveform = torch.from_numpy(data)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                if sr != self.sample_rate:
                    import torchaudio
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                return waveform, choice["codec"], choice["quality"]
            else:
                # Individual-file mode (backward compatible)
                aug_path = choice["path"]
                if not Path(aug_path).exists():
                    logger.debug("Cache miss (file not found): %s", aug_path)
                    return None
                waveform, sr = load_waveform(aug_path, target_sr=self.sample_rate)
                return waveform, choice["codec"], choice["quality"]
        except Exception as e:
            logger.debug("Failed to load cached augmentation: %s", e)
            return None

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        audio_path = row["audio_path"]
        waveform, sr = load_waveform(audio_path, target_sr=self.sample_rate)

        raw_length = waveform.shape[-1]

        # Apply codec augmentation (before cropping for better quality)
        y_codec_aug = 0  # NONE
        y_codec_q_aug = 0  # NONE
        applied_codec = "NONE"
        applied_quality = 0

        if self.augmentor is not None and self.mode == "train":
            if self._aug_cache_index is not None:
                # Pre-computed cache path: we handle codec_prob ourselves since
                # augmentor.augment() is bypassed. This avoids double-checking
                # the probability (augmentor.augment also checks codec_prob).
                if random.random() <= self.augmentor.config.codec_prob:
                    cached = self._get_cached_augmentation(audio_path, row["flac_file"])
                    if cached is not None:
                        waveform, applied_codec, applied_quality = cached
                        y_codec_aug, y_codec_q_aug = self.augmentor.get_domain_labels(
                            applied_codec, applied_quality
                        )
                    else:
                        # Cache miss — fall back to on-the-fly (rare).
                        # Use force=True because we already passed the
                        # codec_prob check above; without it, augmentor.augment()
                        # would roll codec_prob a second time.
                        waveform, applied_codec, applied_quality = self.augmentor.augment(
                            waveform, self.sample_rate, audio_path, force=True
                        )
                        y_codec_aug, y_codec_q_aug = self.augmentor.get_domain_labels(
                            applied_codec, applied_quality
                        )
                # else: keep original (NONE/0) — no augmentation for this sample
            else:
                # No pre-computed cache — use on-the-fly augmentation as before
                # (augmentor.augment handles codec_prob internally)
                waveform, applied_codec, applied_quality = self.augmentor.augment(
                    waveform, self.sample_rate, audio_path
                )
                y_codec_aug, y_codec_q_aug = self.augmentor.get_domain_labels(
                    applied_codec, applied_quality
                )

        # Crop or pad to fixed length
        waveform = crop_or_pad(waveform, self.max_samples, mode=self.mode)

        # Get labels (already encoded in manifest)
        y_task = int(row["y_task"])
        y_codec = int(row["y_codec"])
        y_codec_q = int(row["y_codec_q"])

        result = {
            "waveform": waveform,  # [1, T]
            "y_task": y_task,
            "y_codec": y_codec,
            "y_codec_q": y_codec_q,
            "flac_file": row["flac_file"],
            "speaker_id": row["speaker_id"],
            "codec_seed": row["codec_seed"],
            "codec": row["codec"],
            "codec_q": row["codec_q"],
        }

        # Add synthetic domain labels if using augmentation for DANN
        if self.use_synthetic_labels:
            result["y_codec_aug"] = y_codec_aug
            result["y_codec_q_aug"] = y_codec_q_aug
            result["applied_codec"] = applied_codec
            result["applied_quality"] = applied_quality

        if "attack_label" in row:
            result["attack_label"] = row["attack_label"]
        if "attack_tag" in row:
            result["attack_tag"] = row["attack_tag"]

        if self.return_raw_length:
            result["raw_length"] = raw_length

        return result

    @property
    def num_codecs(self) -> int:
        """Number of unique CODEC values."""
        return len(self.codec_vocab)

    @property
    def num_codec_qs(self) -> int:
        """Number of unique CODEC_Q values."""
        return len(self.codec_q_vocab)


def create_dataloader(
    manifest_path: Path,
    codec_vocab: dict[str, int],
    codec_q_vocab: dict[str, int],
    batch_size: int = 32,
    max_duration_sec: float = 6.0,
    sample_rate: int = 16000,
    mode: Literal["train", "eval"] = "eval",
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for ASVspoof5 dataset.

    Args:
        manifest_path: Path to manifest file.
        codec_vocab: CODEC vocabulary.
        codec_q_vocab: CODEC_Q vocabulary.
        batch_size: Batch size.
        max_duration_sec: Maximum audio duration.
        sample_rate: Sample rate.
        mode: 'train' or 'eval'.
        num_workers: Number of workers.
        shuffle: Whether to shuffle.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        DataLoader instance.
    """
    from .audio import AudioCollator

    dataset = ASVspoof5Dataset(
        manifest_path=manifest_path,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration_sec,
        sample_rate=sample_rate,
        mode=mode,
    )

    fixed_length = int(max_duration_sec * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode=mode)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=True,
    )
