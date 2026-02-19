"""Audio loading, cropping, padding, and batching utilities."""

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio


def load_waveform(
    path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """Load audio waveform using soundfile.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate (resamples if different).

    Returns:
        Tuple of (waveform [1, T], sample_rate).
    """
    # Use soundfile directly to avoid torchcodec issues
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    
    # Ensure shape is [C, T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        # soundfile returns [T, C], transpose to [C, T]
        waveform = waveform.T

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr


def crop_or_pad(
    waveform: torch.Tensor,
    n_samples: int,
    mode: Literal["train", "eval"] = "eval",
) -> torch.Tensor:
    """Crop or pad waveform to fixed length.

    Args:
        waveform: Audio tensor of shape [C, T] or [T].
        n_samples: Target number of samples.
        mode: 'train' for random crop, 'eval' for center crop.

    Returns:
        Waveform of shape [C, n_samples] or [n_samples].
    """
    squeeze = False
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        squeeze = True

    current_len = waveform.shape[-1]

    if current_len > n_samples:
        # Crop
        if mode == "train":
            start = torch.randint(0, current_len - n_samples + 1, (1,)).item()
        else:
            start = (current_len - n_samples) // 2
        waveform = waveform[..., start : start + n_samples]
    elif current_len < n_samples:
        # Pad with zeros
        pad_len = n_samples - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    if squeeze:
        waveform = waveform.squeeze(0)

    return waveform


def compute_attention_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Compute attention mask from lengths.

    Args:
        lengths: Sequence lengths of shape [B].
        max_len: Maximum sequence length.

    Returns:
        Boolean attention mask of shape [B, T] where True = valid.
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_audio_batch(
    samples: list[dict],
    fixed_length: Optional[int] = None,
    mode: Literal["train", "eval"] = "eval",
) -> dict:
    """Collate a batch of audio samples.

    Args:
        samples: List of sample dictionaries with keys:
            - waveform: torch.Tensor [1, T] or [T]
            - y_task: int
            - y_codec: int
            - y_codec_q: int
            - y_codec_aug: int (optional, for synthetic augmentation)
            - y_codec_q_aug: int (optional, for synthetic augmentation)
            - (optional metadata fields)
        fixed_length: If set, crop/pad all waveforms to this length.
        mode: 'train' for random crop, 'eval' for center crop.

    Returns:
        Batched dictionary with:
            - waveform: [B, T]
            - attention_mask: [B, T] (True = valid)
            - lengths: [B]
            - y_task: [B]
            - y_codec: [B]
            - y_codec_q: [B]
            - y_codec_aug: [B] (if present)
            - y_codec_q_aug: [B] (if present)
            - metadata: dict of lists
    """
    waveforms = []
    lengths = []
    y_task = []
    y_codec = []
    y_codec_q = []
    y_codec_aug = []
    y_codec_q_aug = []
    has_aug_labels = "y_codec_aug" in samples[0]

    # Collect metadata keys (exclude tensor fields)
    tensor_keys = {"waveform", "y_task", "y_codec", "y_codec_q", "y_codec_aug", "y_codec_q_aug"}
    metadata_keys = [k for k in samples[0].keys() if k not in tensor_keys]
    metadata = {k: [] for k in metadata_keys}

    for sample in samples:
        wav = sample["waveform"]

        # Ensure [1, T] shape
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Get original length before any processing
        original_len = wav.shape[-1]

        # Apply fixed length crop/pad if specified
        if fixed_length is not None:
            wav = crop_or_pad(wav, fixed_length, mode=mode)
            length = min(original_len, fixed_length)
        else:
            length = original_len

        # Squeeze to [T] for stacking
        waveforms.append(wav.squeeze(0))
        lengths.append(length)
        y_task.append(sample["y_task"])
        y_codec.append(sample["y_codec"])
        y_codec_q.append(sample["y_codec_q"])

        if has_aug_labels:
            y_codec_aug.append(sample["y_codec_aug"])
            y_codec_q_aug.append(sample["y_codec_q_aug"])

        for k in metadata_keys:
            metadata[k].append(sample.get(k))

    # Pad to batch max length if no fixed length
    if fixed_length is None:
        max_len = max(w.shape[0] for w in waveforms)
        padded = []
        for wav in waveforms:
            if wav.shape[0] < max_len:
                wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[0]))
            padded.append(wav)
        waveforms = padded
    else:
        max_len = fixed_length

    # Stack tensors
    waveform_batch = torch.stack(waveforms, dim=0)  # [B, T]
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    attention_mask = compute_attention_mask(lengths_tensor, max_len)

    result = {
        "waveform": waveform_batch,
        "attention_mask": attention_mask,
        "lengths": lengths_tensor,
        "y_task": torch.tensor(y_task, dtype=torch.long),
        "y_codec": torch.tensor(y_codec, dtype=torch.long),
        "y_codec_q": torch.tensor(y_codec_q, dtype=torch.long),
        "metadata": metadata,
    }

    if has_aug_labels:
        result["y_codec_aug"] = torch.tensor(y_codec_aug, dtype=torch.long)
        result["y_codec_q_aug"] = torch.tensor(y_codec_q_aug, dtype=torch.long)

    return result


class AudioCollator:
    """Callable collator for DataLoader.

    Args:
        fixed_length: Fixed waveform length in samples (None = dynamic).
        mode: 'train' for random crop, 'eval' for center crop.
    """

    def __init__(
        self,
        fixed_length: Optional[int] = None,
        mode: Literal["train", "eval"] = "eval",
    ):
        self.fixed_length = fixed_length
        self.mode = mode

    def __call__(self, samples: list[dict]) -> dict:
        return collate_audio_batch(
            samples,
            fixed_length=self.fixed_length,
            mode=self.mode,
        )
