#!/usr/bin/env python3
"""Pre-compute codec augmentations offline for DANN training.

Instead of running ffmpeg on-the-fly during training (~4h/epoch), pre-compute
all codec × quality augmentations so DANN training speed approaches ERM (~0.8h/epoch).

For each audio file in the training manifest, creates augmented versions:
  3 codecs × 5 quality levels = 15 augmented versions per file

Output structure:
  {output_dir}/
  ├── MP3/q1/  MP3/q2/  ...  MP3/q5/
  ├── AAC/q1/  AAC/q2/  ...  AAC/q5/
  ├── OPUS/q1/ OPUS/q2/ ... OPUS/q5/
  └── manifest.json

Usage:
    python scripts/precompute_augmentations.py \\
        --data-root $ASVSPOOF5_ROOT \\
        --output-dir /scratch/augmented_cache \\
        --codecs MP3 AAC OPUS \\
        --qualities 1 2 3 4 5 \\
        --num-workers 8 \\
        --split train
"""

import argparse
import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute codec augmentations for DANN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="ASVspoof5 data root (overrides ASVSPOOF5_ROOT env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for pre-computed augmentations",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=["MP3", "AAC", "OPUS"],
        help="Codecs to use (default: MP3 AAC OPUS)",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Quality levels to use (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "eval"],
        help="Data split to process (default: train)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Directory with manifest parquet files (default: data/manifests)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="flac",
        choices=["flac", "wav"],
        help="Output audio format (default: flac)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N files (for testing)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index (0-based) for parallel processing",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards",
    )
    parser.add_argument(
        "--merge-manifests",
        action="store_true",
        help="Merge per-shard manifests into final manifest.json (run after all shards complete)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with one file before full processing (recommended for first run)",
    )
    parser.add_argument(
        "--tar-shards",
        type=int,
        default=None,
        help=(
            "Pack augmented files into tar-based shards instead of individual files. "
            "Value is the number of audio files per shard. Use 0 for a single shard "
            "containing everything. This drastically reduces inode usage on HPC systems."
        ),
    )
    return parser.parse_args()


def check_ffmpeg() -> bool:
    """Check that ffmpeg is available."""
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        version_line = result.stdout.decode().split("\n")[0]
        logger.info(f"ffmpeg found: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_encoder_support(codecs: list[str]) -> list[str]:
    """Check which codecs are supported by ffmpeg."""
    import subprocess

    encoder_map = {
        "MP3": "libmp3lame",
        "AAC": "aac",
        "OPUS": "libopus",
        "SPEEX": "libspeex",
        "AMR": "libopencore_amrnb",
    }

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
        )
        encoder_output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Cannot query ffmpeg encoders")
        return []

    supported = []
    for codec in codecs:
        encoder = encoder_map.get(codec)
        if encoder and encoder in encoder_output:
            supported.append(codec)
            logger.info(f"  ✓ {codec} ({encoder}) supported")
        else:
            logger.warning(f"  ✗ {codec} ({encoder}) NOT supported — skipping")

    return supported


def run_smoke_test(
    audio_path: str,
    output_dir: Path,
    codecs: list[str],
    sample_rate: int,
    output_format: str,
) -> bool:
    """Run a quick smoke test to verify the pipeline works.

    Tests a single file with all codec/quality combinations to catch
    configuration issues before committing to a long run.

    Args:
        audio_path: Path to a test audio file.
        output_dir: Output directory for test files.
        codecs: List of codecs to test.
        sample_rate: Target sample rate.
        output_format: Output format (flac, wav).

    Returns:
        True if all tests passed, False otherwise.
    """
    import tempfile

    logger.info("")
    logger.info("=" * 50)
    logger.info("SMOKE TEST")
    logger.info("=" * 50)
    logger.info(f"Testing with: {audio_path}")

    test_dir = output_dir / "_smoke_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    errors = []

    stem = Path(audio_path).stem

    for codec in codecs:
        for quality in [1, 3, 5]:  # Test low, mid, high quality
            output_path = test_dir / codec / f"q{quality}" / f"{stem}.{output_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use the same temp path logic as process_single_file
            temp_path = output_path.with_suffix(f".{output_format}.tmp")

            success, error = apply_codec_to_file(
                input_path=audio_path,
                output_path=str(temp_path),
                codec=codec,
                quality=quality,
                sample_rate=sample_rate,
                output_format=output_format,
            )

            if success:
                # Rename temp to final
                os.replace(str(temp_path), str(output_path))

                # Verify output is valid audio
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(output_path))
                    if len(data) > 0 and sr > 0:
                        logger.info(f"  ✓ {codec}/q{quality}: OK ({len(data)} samples)")
                        passed += 1
                    else:
                        logger.error(f"  ✗ {codec}/q{quality}: Empty output")
                        failed += 1
                        errors.append(f"{codec}/q{quality}: Empty output")
                except Exception as e:
                    logger.error(f"  ✗ {codec}/q{quality}: Invalid output: {e}")
                    failed += 1
                    errors.append(f"{codec}/q{quality}: {e}")
            else:
                logger.error(f"  ✗ {codec}/q{quality}: {error[:100]}")
                failed += 1
                errors.append(f"{codec}/q{quality}: {error[:100]}")

                # Clean up failed temp file
                if temp_path.exists():
                    temp_path.unlink()

    # Clean up test directory
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

    logger.info("")
    logger.info(f"Smoke test: {passed} passed, {failed} failed")

    if failed > 0:
        logger.error("Smoke test FAILED! Fix the issues before running the full job.")
        for err in errors[:5]:
            logger.error(f"  - {err}")
        return False
    else:
        logger.info("Smoke test PASSED! Safe to proceed with full run.")
        return True


# Bitrate configurations per codec and quality level (kbps)
# Must match codec_augment.py exactly
CODEC_BITRATES = {
    "MP3": {1: 64, 2: 96, 3: 128, 4: 192, 5: 256},
    "AAC": {1: 32, 2: 64, 3: 96, 4: 128, 5: 192},
    "OPUS": {1: 12, 2: 24, 3: 48, 4: 64, 5: 96},
    "SPEEX": {1: 8, 2: 16, 3: 24, 4: 32, 5: 44},
    "AMR": {1: 6, 2: 9, 3: 12, 4: 18, 5: 23},
}

# ffmpeg encoder and format config (matches codec_augment.py)
CODEC_CONFIG = {
    "MP3": {"encoder": "libmp3lame", "format": "mp3", "ext": ".mp3"},
    "AAC": {"encoder": "aac", "format": "adts", "ext": ".aac"},
    "OPUS": {"encoder": "libopus", "format": "opus", "ext": ".opus"},
    "SPEEX": {"encoder": "libspeex", "format": "ogg", "ext": ".spx"},
    "AMR": {
        "encoder": "libopencore_amrnb",
        "format": "amr",
        "ext": ".amr",
        "sample_rate": 8000,
    },
}


def apply_codec_to_file(
    input_path: str,
    output_path: str,
    codec: str,
    quality: int,
    sample_rate: int = 16000,
    output_format: str = "flac",
) -> tuple[bool, str]:
    """Apply codec compression to an audio file.

    Encode input → codec format → decode back to output format.
    This matches the on-the-fly augmentation pipeline exactly.

    Args:
        input_path: Path to input audio file.
        output_path: Path to write output audio file.
        codec: Codec name (MP3, AAC, OPUS, etc.).
        quality: Quality level (1-5).
        sample_rate: Target sample rate.
        output_format: Output audio format (flac, wav). Required for temp files
            with non-standard extensions like .flac.tmp.

    Returns:
        Tuple of (success: bool, error_message: str).
    """
    import subprocess
    import tempfile

    bitrate = CODEC_BITRATES.get(codec, {}).get(quality)
    if bitrate is None:
        return False, f"Unknown codec/quality: {codec}/{quality}"

    config = CODEC_CONFIG.get(codec)
    if config is None:
        return False, f"Unknown codec config: {codec}"

    target_sr = config.get("sample_rate", sample_rate)
    tmp_encoded = None

    try:
        # Create temp file for the intermediate encoded format
        with tempfile.NamedTemporaryFile(
            suffix=config["ext"], delete=False
        ) as tmp:
            tmp_encoded = tmp.name

        # Step 1: Encode (input → codec format)
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(input_path),
            "-ar", str(target_sr),
            "-ac", "1",
            "-c:a", config["encoder"],
            "-b:a", f"{bitrate}k",
            "-f", config["format"],
            tmp_encoded,
        ]

        subprocess.run(
            encode_cmd,
            capture_output=True,
            check=True,
        )

        # Step 2: Decode back (codec format → output format)
        # Explicitly specify output format with -f since temp files may have
        # non-standard extensions like .flac.tmp that ffmpeg can't infer from.
        decode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", tmp_encoded,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", output_format,
            str(output_path),
        ]

        subprocess.run(
            decode_cmd,
            capture_output=True,
            check=True,
        )

        return True, ""

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()[:300] if e.stderr else "unknown error"
        return False, f"ffmpeg error: {stderr}"
    except Exception as e:
        return False, str(e)
    finally:
        if tmp_encoded is not None:
            try:
                os.unlink(tmp_encoded)
            except OSError:
                pass


def process_single_file(args_tuple: tuple) -> dict:
    """Process a single audio file for one codec/quality combination.

    Designed to be called from ProcessPoolExecutor. Takes a single tuple
    argument for compatibility with map().

    Args:
        args_tuple: (input_path, output_path, codec, quality, sample_rate, output_format)

    Returns:
        Dict with status info.
    """
    input_path, output_path, codec, quality, sample_rate, output_format = args_tuple

    output_path = Path(output_path)

    # Skip if output already exists (resume support)
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "codec": codec,
            "quality": quality,
            "status": "skipped",
            "error": "",
        }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then atomically rename (crash safety)
    temp_path = output_path.with_suffix(f".{output_format}.tmp")

    success, error = apply_codec_to_file(
        input_path=str(input_path),
        output_path=str(temp_path),
        codec=codec,
        quality=quality,
        sample_rate=sample_rate,
        output_format=output_format,
    )

    if success:
        # Atomic rename
        os.replace(str(temp_path), str(output_path))
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "codec": codec,
            "quality": quality,
            "status": "success",
            "error": "",
        }
    else:
        # Clean up temp file on failure
        try:
            if os.path.exists(str(temp_path)):
                os.unlink(str(temp_path))
        except OSError:
            pass
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "codec": codec,
            "quality": quality,
            "status": "failed",
            "error": error,
        }


def load_manifest(manifest_dir: Path, split: str) -> list[str]:
    """Load audio file paths from manifest.

    Args:
        manifest_dir: Directory containing manifest parquet files.
        split: Data split name.

    Returns:
        List of audio file paths.
    """
    import pandas as pd

    manifest_path = manifest_dir / f"{split}.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run `python scripts/prepare_asvspoof5.py` first to create manifests."
        )

    df = pd.read_parquet(manifest_path)
    logger.info(f"Loaded manifest: {manifest_path} ({len(df)} samples)")

    if "audio_path" not in df.columns:
        raise ValueError(
            f"Manifest missing 'audio_path' column. "
            f"Available columns: {list(df.columns)}"
        )

    audio_paths = df["audio_path"].tolist()

    return audio_paths


def build_task_list(
    audio_paths: list[str],
    output_dir: Path,
    codecs: list[str],
    qualities: list[int],
    sample_rate: int,
    output_format: str,
) -> list[tuple]:
    """Build the full list of (input, output, codec, quality) tasks.

    Args:
        audio_paths: List of input audio file paths.
        output_dir: Base output directory.
        codecs: List of codec names.
        qualities: List of quality levels.
        sample_rate: Target sample rate.
        output_format: Output file format extension.

    Returns:
        List of task tuples for process_single_file.
    """
    tasks = []
    for audio_path in audio_paths:
        # Use the flac filename (without directory) as the output filename
        stem = Path(audio_path).stem
        for codec in codecs:
            for quality in qualities:
                output_path = (
                    output_dir / codec / f"q{quality}" / f"{stem}.{output_format}"
                )
                tasks.append(
                    (audio_path, str(output_path), codec, quality, sample_rate, output_format)
                )
    return tasks


def build_manifest(
    audio_paths: list[str],
    output_dir: Path,
    codecs: list[str],
    qualities: list[int],
    output_format: str,
    tar_entries: list[dict] | None = None,
) -> dict:
    """Build the augmentation manifest mapping originals to augmented files.

    Args:
        audio_paths: List of original audio file paths.
        output_dir: Base output directory.
        codecs: List of codec names.
        qualities: List of quality levels.
        output_format: Output file format extension.
        tar_entries: If provided, use these pre-built entries (from tar-shard mode)
            instead of generating individual-file entries.

    Returns:
        Manifest dict with metadata and per-file mappings.
    """
    # Build codec label vocab.
    # IMPORTANT: This vocab MUST match the one built by
    # CodecAugmentor.codec_vocab (in codec_augment.py). Both use the same
    # scheme: NONE=0, then supported codecs in order starting at 1.
    # When the dataset loads this manifest via _load_aug_cache(), it will
    # validate that the manifest's codec_vocab matches the augmentor's
    # live vocab to catch any divergence early.
    codec_vocab = {"NONE": 0}
    for i, codec in enumerate(codecs, start=1):
        codec_vocab[codec] = i

    if tar_entries is not None:
        # Tar-shard mode: entries already built by process_tar_shard,
        # but we need to add codec_label/codec_q_label fields.
        for entry in tar_entries:
            for af in entry["augmented_files"]:
                af["codec_label"] = codec_vocab[af["codec"]]
                af["codec_q_label"] = f"{af['codec']}_Q{af['quality']}"
        entries = tar_entries
    else:
        entries = []
        for audio_path in audio_paths:
            stem = Path(audio_path).stem
            augmented_files = []
            for codec in codecs:
                for quality in qualities:
                    aug_path = str(
                        output_dir / codec / f"q{quality}" / f"{stem}.{output_format}"
                    )
                    augmented_files.append(
                        {
                            "path": aug_path,
                            "codec": codec,
                            "quality": quality,
                            "codec_label": codec_vocab[codec],
                            "codec_q_label": f"{codec}_Q{quality}",
                        }
                    )
            entries.append(
                {
                    "original_file": audio_path,
                    "flac_stem": stem,
                    "augmented_files": augmented_files,
                }
            )

    manifest = {
        "version": 2 if tar_entries is not None else 1,
        "format": "tar" if tar_entries is not None else "individual",
        "codecs": codecs,
        "qualities": qualities,
        "codec_vocab": codec_vocab,
        "num_originals": len(entries),
        "augmentations_per_file": len(codecs) * len(qualities),
        "total_augmented_files": len(entries) * len(codecs) * len(qualities),
        "entries": entries,
    }

    return manifest


def process_tar_shard(
    shard_index: int,
    audio_paths: list[str],
    output_dir: Path,
    codecs: list[str],
    qualities: list[int],
    sample_rate: int,
    output_format: str,
    num_workers: int,
) -> dict:
    """Process a group of audio files and pack all augmentations into a single tar.

    Each shard tar contains internal paths like: MP3/q1/stem.flac, AAC/q3/stem.flac, etc.
    This replaces individual file output, drastically reducing inode count.

    Args:
        shard_index: Index of this shard.
        audio_paths: Audio files assigned to this shard.
        output_dir: Base output directory (tar files go here).
        codecs: Codec names.
        qualities: Quality levels.
        sample_rate: Target sample rate.
        output_format: Output audio format (flac, wav).
        num_workers: Number of parallel workers for ffmpeg.

    Returns:
        Dict with shard stats (success, failed counts) and manifest entries.
    """
    tar_path = output_dir / f"shard_{shard_index:06d}.tar"
    temp_tar_path = tar_path.with_suffix(".tar.tmp")

    # Build tasks for all files in this shard — use a temp dir for intermediate files
    with tempfile.TemporaryDirectory(prefix=f"aug_shard_{shard_index}_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tasks = build_task_list(
            audio_paths=audio_paths,
            output_dir=tmp_dir,
            codecs=codecs,
            qualities=qualities,
            sample_rate=sample_rate,
            output_format=output_format,
        )

        results = {"success": 0, "failed": 0, "skipped": 0}
        errors = []

        # Process augmentations in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    results[result["status"]] += 1
                    if result["status"] == "failed":
                        errors.append(result)
                except Exception as e:
                    results["failed"] += 1
                    errors.append({"error": str(e)})

        # Pack everything into a tar file
        manifest_entries = []
        with tarfile.open(str(temp_tar_path), "w") as tar:
            for audio_path in audio_paths:
                stem = Path(audio_path).stem
                aug_files = []
                for codec in codecs:
                    for quality in qualities:
                        fname = f"{stem}.{output_format}"
                        internal_path = f"{codec}/q{quality}/{fname}"
                        fs_path = tmp_dir / codec / f"q{quality}" / fname

                        if fs_path.exists() and fs_path.stat().st_size > 0:
                            tar.add(str(fs_path), arcname=internal_path)
                            aug_files.append({
                                "tar_path": str(tar_path),
                                "internal_path": internal_path,
                                "codec": codec,
                                "quality": quality,
                            })

                if aug_files:
                    manifest_entries.append({
                        "original_file": audio_path,
                        "flac_stem": stem,
                        "augmented_files": aug_files,
                    })

    # Atomic rename
    os.replace(str(temp_tar_path), str(tar_path))

    return {
        "shard_index": shard_index,
        "tar_path": str(tar_path),
        "stats": results,
        "errors": errors,
        "entries": manifest_entries,
    }


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def main():
    args = parse_args()

    # Handle manifest merge mode (run after all shards complete)
    if args.merge_manifests:
        import re

        logger.info("Merging per-shard manifests...")
        shard_files = sorted(args.output_dir.glob("manifest_shard_*.json"))
        if not shard_files:
            logger.error(f"No shard manifests found in {args.output_dir}")
            sys.exit(1)

        # Validate all expected shards are present
        shard_indices = set()
        for sf in shard_files:
            match = re.match(r"manifest_shard_(\d+)\.json", sf.name)
            if match:
                shard_indices.add(int(match.group(1)))
        expected = set(range(max(shard_indices) + 1)) if shard_indices else set()
        missing = expected - shard_indices
        if missing:
            logger.warning(
                f"Missing shard manifests: {sorted(missing)}. "
                f"Found {len(shard_indices)}/{len(expected)}. "
                f"Proceeding with partial merge — training data may be incomplete."
            )

        merged_entries = []
        base_manifest = None
        for sf in shard_files:
            with open(sf) as f:
                shard_manifest = json.load(f)
            if base_manifest is None:
                base_manifest = shard_manifest
            merged_entries.extend(shard_manifest["entries"])
            logger.info(f"  {sf.name}: {len(shard_manifest['entries'])} entries")

        base_manifest["entries"] = merged_entries
        base_manifest["num_originals"] = len(merged_entries)
        base_manifest["total_augmented_files"] = (
            len(merged_entries) * base_manifest["augmentations_per_file"]
        )

        # Atomic write — use temp file + os.replace to prevent corruption
        manifest_path = args.output_dir / "manifest.json"
        temp_path = manifest_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(base_manifest, f, indent=2)
        os.replace(str(temp_path), str(manifest_path))
        logger.info(f"Merged manifest: {manifest_path} ({len(merged_entries)} entries from {len(shard_files)} shards)")
        return

    # Resolve data root
    if args.data_root:
        data_root = args.data_root
    else:
        data_root_env = os.environ.get("ASVSPOOF5_ROOT")
        if data_root_env:
            data_root = Path(data_root_env)
        else:
            logger.error(
                "No data root specified. Use --data-root or set ASVSPOOF5_ROOT env var."
            )
            sys.exit(1)

    # Resolve manifest directory
    if args.manifest_dir:
        manifest_dir = args.manifest_dir
    else:
        # Use the project's default manifest location
        project_root = Path(__file__).resolve().parent.parent
        manifest_dir = project_root / "data" / "manifests"

    logger.info("=" * 70)
    logger.info("Pre-compute Codec Augmentations for DANN Training")
    logger.info("=" * 70)
    logger.info(f"Data root:    {data_root}")
    logger.info(f"Manifest dir: {manifest_dir}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Split:        {args.split}")
    logger.info(f"Codecs:       {args.codecs}")
    logger.info(f"Qualities:    {args.qualities}")
    logger.info(f"Workers:      {args.num_workers}")
    logger.info(f"Sample rate:  {args.sample_rate}")
    logger.info(f"Format:       {args.output_format}")
    logger.info("")

    # Check ffmpeg
    if not check_ffmpeg():
        logger.error("ffmpeg not found. Install ffmpeg or load the module.")
        logger.error("On Snellius: module load FFmpeg/7.1.1-GCCcore-14.2.0")
        sys.exit(1)

    # Check codec support
    logger.info("Checking ffmpeg codec support:")
    supported_codecs = check_encoder_support(args.codecs)
    if not supported_codecs:
        logger.error("No requested codecs are supported by ffmpeg!")
        sys.exit(1)

    if len(supported_codecs) < len(args.codecs):
        logger.warning(
            f"Only {len(supported_codecs)}/{len(args.codecs)} codecs supported. "
            f"Proceeding with: {supported_codecs}"
        )

    # Load manifest
    audio_paths = load_manifest(manifest_dir, args.split)

    # Run smoke test if requested (recommended for first run)
    if args.smoke_test:
        if not audio_paths:
            logger.error("No audio files found for smoke test")
            sys.exit(1)
        smoke_ok = run_smoke_test(
            audio_path=audio_paths[0],
            output_dir=args.output_dir,
            codecs=supported_codecs,
            sample_rate=args.sample_rate,
            output_format=args.output_format,
        )
        if not smoke_ok:
            sys.exit(1)
        logger.info("")

    # Apply limit if specified
    if args.limit:
        audio_paths = audio_paths[: args.limit]
        logger.info(f"Limited to first {args.limit} files")

    # Shard the file list for parallel job arrays
    # Validate shard arguments are both provided or both omitted
    has_shard_index = args.shard_index is not None
    has_num_shards = args.num_shards is not None
    if has_shard_index != has_num_shards:
        logger.error("--shard-index and --num-shards must be used together")
        sys.exit(1)
    is_sharded = has_shard_index and has_num_shards
    if is_sharded:
        if args.num_shards <= 0:
            logger.error(f"--num-shards must be > 0, got {args.num_shards}")
            sys.exit(1)
        if not (0 <= args.shard_index < args.num_shards):
            logger.error(
                f"--shard-index must be in range [0, {args.num_shards}), "
                f"got {args.shard_index}"
            )
            sys.exit(1)
        total_files = len(audio_paths)
        shard_size = (total_files + args.num_shards - 1) // args.num_shards  # ceiling division
        start = args.shard_index * shard_size
        end = min(start + shard_size, total_files)
        audio_paths = audio_paths[start:end]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: files {start}-{end} ({len(audio_paths)} files)")

    # Build task list
    tasks = build_task_list(
        audio_paths=audio_paths,
        output_dir=args.output_dir,
        codecs=supported_codecs,
        qualities=args.qualities,
        sample_rate=args.sample_rate,
        output_format=args.output_format,
    )

    total_tasks = len(tasks)
    augs_per_file = len(supported_codecs) * len(args.qualities)

    logger.info(f"Audio files:            {len(audio_paths)}")
    logger.info(f"Codecs × Qualities:     {len(supported_codecs)} × {len(args.qualities)} = {augs_per_file}")
    logger.info(f"Total augmentations:    {total_tasks}")
    logger.info("")

    # Estimate disk usage
    # Training files are ~6s of 16kHz mono FLAC ≈ ~50-100KB per file
    # Augmented FLAC is similar size
    avg_file_kb = 80  # rough estimate
    estimated_total_kb = total_tasks * avg_file_kb
    logger.info(
        f"Estimated disk usage:   ~{format_size(estimated_total_kb * 1024)} "
        f"(assuming ~{avg_file_kb}KB/file)"
    )
    logger.info("")

    if args.dry_run:
        logger.info("[DRY RUN] Would process the above. Exiting.")
        # Show a few example tasks
        for task in tasks[:5]:
            logger.info(f"  {Path(task[0]).name} → {task[2]}/q{task[3]}")
        if len(tasks) > 5:
            logger.info(f"  ... and {len(tasks) - 5} more")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tar-shard mode ──────────────────────────────────────────────────
    if args.tar_shards is not None:
        start_time = time.time()
        files_per_shard = args.tar_shards if args.tar_shards > 0 else len(audio_paths)
        # Split audio_paths into shard groups
        shard_groups = [
            audio_paths[i : i + files_per_shard]
            for i in range(0, len(audio_paths), files_per_shard)
        ]
        logger.info(
            f"Tar-shard mode: {len(shard_groups)} shards, "
            f"~{files_per_shard} files/shard"
        )

        all_entries = []
        results = {"success": 0, "skipped": 0, "failed": 0}
        errors = []

        # When using SLURM array sharding (--shard-index), offset tar shard
        # names so each array task writes a unique file.
        shard_name_offset = args.shard_index if args.shard_index is not None else 0

        for si, group in enumerate(shard_groups):
            shard_id = shard_name_offset + si
            tar_path = args.output_dir / f"shard_{shard_id:06d}.tar"
            if tar_path.exists():
                # Resume: skip completed shards, load their entries from manifest
                shard_manifest_path = args.output_dir / f"manifest_shard_{shard_id:06d}.json"
                if shard_manifest_path.exists():
                    with open(shard_manifest_path) as f:
                        sm = json.load(f)
                    all_entries.extend(sm.get("entries", []))
                    logger.info(f"  Shard {shard_id}: skipped (already exists)")
                    results["skipped"] += len(group) * augs_per_file
                    continue

            logger.info(f"  Shard {shard_id}/{len(shard_groups)}: processing {len(group)} files...")
            shard_result = process_tar_shard(
                shard_index=shard_id,
                audio_paths=group,
                output_dir=args.output_dir,
                codecs=supported_codecs,
                qualities=args.qualities,
                sample_rate=args.sample_rate,
                output_format=args.output_format,
                num_workers=args.num_workers,
            )
            results["success"] += shard_result["stats"]["success"]
            results["failed"] += shard_result["stats"]["failed"]
            all_entries.extend(shard_result["entries"])
            errors.extend(shard_result["errors"])

            # Save per-shard manifest for resume support
            shard_manifest = build_manifest(
                audio_paths=group,
                output_dir=args.output_dir,
                codecs=supported_codecs,
                qualities=args.qualities,
                output_format=args.output_format,
                tar_entries=shard_result["entries"],
            )
            shard_manifest_path = args.output_dir / f"manifest_shard_{shard_id:06d}.json"
            with open(shard_manifest_path, "w") as f:
                json.dump(shard_manifest, f, indent=2)

        elapsed = time.time() - start_time

        # Build combined manifest
        logger.info("Building combined tar-shard manifest...")
        manifest = build_manifest(
            audio_paths=audio_paths,
            output_dir=args.output_dir,
            codecs=supported_codecs,
            qualities=args.qualities,
            output_format=args.output_format,
            tar_entries=all_entries,
        )
        if is_sharded:
            manifest_path = args.output_dir / f"manifest_shard_{args.shard_index}.json"
        else:
            manifest_path = args.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")

    else:
        # ── Individual-file mode (original behavior) ────────────────────
        # Process with multiprocessing
        try:
            from tqdm import tqdm
        except ImportError:
            logger.warning("tqdm not installed, using simple progress logging")
            tqdm = None

        start_time = time.time()
        results = {"success": 0, "skipped": 0, "failed": 0}
        errors = []

        logger.info(f"Starting processing with {args.num_workers} workers...")

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_single_file, task): task for task in tasks
            }

            if tqdm is not None:
                progress = tqdm(
                    as_completed(futures),
                    total=total_tasks,
                    desc="Augmenting",
                    unit="file",
                    ncols=100,
                )
            else:
                progress = as_completed(futures)

            completed = 0
            for future in progress:
                try:
                    result = future.result(timeout=120)
                    results[result["status"]] += 1

                    if result["status"] == "failed":
                        errors.append(result)
                        if len(errors) <= 10:
                            logger.warning(
                                f"Failed: {Path(result['input_path']).name} "
                                f"({result['codec']}/q{result['quality']}): "
                                f"{result['error'][:100]}"
                            )
                except Exception as e:
                    results["failed"] += 1
                    errors.append({"error": str(e)})
                    logger.warning(f"Worker exception: {e}")

                completed += 1

                # Log progress every 5000 tasks if no tqdm
                if tqdm is None and completed % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{total_tasks} "
                        f"({100 * completed / total_tasks:.1f}%) "
                        f"ETA: {eta / 60:.1f}min"
                    )

        elapsed = time.time() - start_time

        # Build and save manifest
        # When sharded, write per-shard manifests to avoid race conditions.
        # A separate merge step (--merge-manifests) combines them after all shards complete.
        logger.info("Building augmentation manifest...")
        manifest = build_manifest(
            audio_paths=audio_paths,
            output_dir=args.output_dir,
            codecs=supported_codecs,
            qualities=args.qualities,
            output_format=args.output_format,
        )
        if is_sharded:
            manifest_path = args.output_dir / f"manifest_shard_{args.shard_index}.json"
        else:
            manifest_path = args.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")

    # Measure actual disk usage
    actual_size = get_dir_size(args.output_dir)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total time:      {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info(f"Total tasks:     {total_tasks}")
    logger.info(f"  Succeeded:     {results['success']}")
    logger.info(f"  Skipped:       {results['skipped']} (already existed)")
    logger.info(f"  Failed:        {results['failed']}")
    logger.info(f"Throughput:      {total_tasks / elapsed:.1f} files/sec")
    logger.info(f"Disk usage:      {format_size(actual_size)}")
    logger.info(f"Output dir:      {args.output_dir}")
    logger.info(f"Manifest:        {manifest_path}")
    logger.info("")

    if results["failed"] > 0:
        logger.warning(
            f"{results['failed']} augmentations failed. "
            f"Re-run the script to retry (resume support will skip completed files)."
        )
        # Save error log
        error_log_path = args.output_dir / "errors.json"
        with open(error_log_path, "w") as f:
            json.dump(errors[:100], f, indent=2)  # Save first 100 errors
        logger.info(f"Error log saved: {error_log_path}")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Set cache_dir in your config:")
    logger.info(f"     augmentation.cache_dir: {args.output_dir}")
    logger.info("  2. Run training as usual:")
    logger.info("     python scripts/train.py --config configs/wavlm_dann.yaml")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
