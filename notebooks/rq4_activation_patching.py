#!/usr/bin/env python
# coding: utf-8

# # RQ4: Activation Patching for Domain Invariance
# 
# **Research Question:** Can activation patching reduce domain leakage without full retraining?
# 
# **Hypothesis:** By identifying layers where DANN diverges most from ERM (via CKA), we can "transplant" domain-invariant representations into ERM at inference time.
# 
# ## Approach
# 
# 1. **CKA Analysis** — Identify which layers show largest representation difference between ERM and DANN
# 2. **Activation Patching** — During inference, replace ERM activations at layer L with DANN activations
# 3. **Evaluation** — Measure domain probe accuracy and EER on patched models
# 
# ## Expected Outcome
# 
# If successful, this provides a lightweight method to improve domain robustness without expensive DANN training.

# ## Setup

# In[5]:


import json
import os
import sys
from pathlib import Path

# Resolve project paths from this script location instead of current working directory.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_root = project_root / "src"
for candidate_path in (src_root, project_root):
    candidate_path_str = str(candidate_path)
    if candidate_path_str not in sys.path:
        sys.path.insert(0, candidate_path_str)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Core utilities
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_manifest_path,
    get_manifests_dir,
    get_runs_dir,
    set_seed,
)

# Data loading - use create_dataloader (not get_dataloader)
from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset,
    AudioCollator,
    create_dataloader,
    load_vocab,
)

# Analysis tools
from asvspoof5_domain_invariant_cm.analysis import (
    compute_linear_cka,
    compare_representations,
    layerwise_probing,
    ActivationCache,
    register_hooks,
    remove_hooks,
)

# Evaluation metrics
from asvspoof5_domain_invariant_cm.evaluation import compute_eer, compute_min_dcf

# Model components
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)

set_seed(42)
device = get_device()
print(f"Using device: {device}")


def _get_env_int(name: str, default: int | None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


rq4_eval_split = os.environ.get("RQ4_EVAL_SPLIT", "dev")
rq4_max_eval_samples = _get_env_int("RQ4_MAX_EVAL_SAMPLES", 20000)
rq4_max_cka_samples = _get_env_int("RQ4_MAX_CKA_SAMPLES", 5000)
rq4_max_probe_samples = _get_env_int("RQ4_MAX_PROBE_SAMPLES", 5000)
rq4_probe_split = os.environ.get("RQ4_PROBE_SPLIT", rq4_eval_split)
rq4_output_prefix = os.environ.get("RQ4_OUTPUT_PREFIX", "rq4")
rq4_output_dir = Path(os.environ.get("RQ4_OUTPUT_DIR", "results"))
rq4_output_dir.mkdir(parents=True, exist_ok=True)
rq4_repr_cache_name = os.environ.get("RQ4_REPR_CACHE", f"{rq4_output_prefix}_repr_cache.npz")
rq4_metadata_name = os.environ.get("RQ4_METADATA_JSON", f"{rq4_output_prefix}_metadata.json")


def resolve_rq4_output_path(value: str) -> Path:
    """Resolve output path: relative filenames go under RQ4_OUTPUT_DIR."""
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    return rq4_output_dir / candidate
print(
    "RQ4 run config | "
    f"eval_split={rq4_eval_split} "
    f"probe_split={rq4_probe_split} "
    f"max_eval_samples={rq4_max_eval_samples} "
    f"max_cka_samples={rq4_max_cka_samples} "
    f"max_probe_samples={rq4_max_probe_samples} "
    f"output_dir={rq4_output_dir}"
)


# ## 1. Load Checkpoints
# 
# Load the ERM and DANN models (WavLM backbone).
# 
# We use a robust model loading function that handles architecture reconstruction from config.

# In[6]:


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint with proper architecture reconstruction.

    Adapted from scripts/evaluate.py - handles both ERM and DANN models,
    auto-detects discriminator dimensions from weights.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, config, codec_vocab, codec_q_vocab).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state_dict", {})

    run_dir = checkpoint_path.parent.parent
    run_codec_vocab_path = run_dir / "codec_vocab.json"
    run_codec_q_vocab_path = run_dir / "codec_q_vocab.json"
    fallback_codec_vocab_path = get_manifests_dir() / "codec_vocab.json"
    fallback_codec_q_vocab_path = get_manifests_dir() / "codec_q_vocab.json"

    codec_vocab_path = run_codec_vocab_path if run_codec_vocab_path.exists() else fallback_codec_vocab_path
    codec_q_vocab_path = run_codec_q_vocab_path if run_codec_q_vocab_path.exists() else fallback_codec_q_vocab_path

    if not codec_vocab_path.exists():
        raise FileNotFoundError(
            f"Missing codec_vocab.json. Checked {run_codec_vocab_path} and {fallback_codec_vocab_path}."
        )
    if not codec_q_vocab_path.exists():
        raise FileNotFoundError(
            f"Missing codec_q_vocab.json. Checked {run_codec_q_vocab_path} and {fallback_codec_q_vocab_path}."
        )

    codec_vocab = load_vocab(codec_vocab_path)
    codec_q_vocab = load_vocab(codec_q_vocab_path)

    # Build architecture from config
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,  # Always freeze for inference
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    proj_input_dim = backbone.hidden_size * 2 if pooling_method == "stats" else backbone.hidden_size

    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)
    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        # Auto-detect from weights if available
        disc_weight_key = "domain_discriminator.shared.0.weight"
        if disc_weight_key in state_dict:
            disc_input_dim = state_dict[disc_weight_key].shape[1]
        else:
            disc_input_dim = disc_cfg.get("input_dim", proj_input_dim)

        domain_discriminator = MultiHeadDomainDiscriminator(
            input_dim=disc_input_dim,
            num_codecs=len(codec_vocab),
            num_codec_qs=len(codec_q_vocab),
            hidden_dim=disc_cfg.get("hidden_dim", 512),
            dropout=disc_cfg.get("dropout", 0.1),
        )

        model = DANNModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
            domain_discriminator=domain_discriminator,
            lambda_=0.0,  # No GRL during inference
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config, codec_vocab, codec_q_vocab


# In[7]:


def resolve_checkpoint_path(*relative_parts: str) -> Path:
    """Resolve checkpoints from RUNS_DIR env or project runs directory."""
    runs_dir = Path(os.environ["RUNS_DIR"]) if "RUNS_DIR" in os.environ else get_runs_dir()
    return runs_dir.joinpath(*relative_parts)


# Parameterized pairwise setup (defaults keep prior ERM vs DANN behavior).
model_a_run = os.environ.get("RQ4_MODEL_A_RUN", "wavlm_erm")
model_a_ckpt = os.environ.get("RQ4_MODEL_A_CKPT", "best.pt")
model_b_run = os.environ.get("RQ4_MODEL_B_RUN", "wavlm_dann")
model_b_ckpt = os.environ.get("RQ4_MODEL_B_CKPT", "epoch_5_patched.pt")

MODEL_A_CHECKPOINT = resolve_checkpoint_path(model_a_run, "checkpoints", model_a_ckpt)
MODEL_B_CHECKPOINT = resolve_checkpoint_path(model_b_run, "checkpoints", model_b_ckpt)

print(f"Model A checkpoint: {MODEL_A_CHECKPOINT}")
print(f"Model B checkpoint: {MODEL_B_CHECKPOINT}")
print(f"Model A checkpoint exists: {MODEL_A_CHECKPOINT.exists()}")
print(f"Model B checkpoint exists: {MODEL_B_CHECKPOINT.exists()}")

if not MODEL_A_CHECKPOINT.exists() or not MODEL_B_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Missing model checkpoints. Configure RUNS_DIR and optionally "
        "RQ4_MODEL_A_RUN/RQ4_MODEL_A_CKPT/RQ4_MODEL_B_RUN/RQ4_MODEL_B_CKPT."
    )

# Load models
print("\nLoading model A...")
model_a_model, model_a_config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(MODEL_A_CHECKPOINT, device)
model_a_method = model_a_config.get("training", {}).get("method", "erm")
print(f"Model A method: {model_a_method}")

print("\nLoading model B...")
model_b_model, model_b_config, _, _ = load_model_from_checkpoint(MODEL_B_CHECKPOINT, device)
model_b_method = model_b_config.get("training", {}).get("method", "dann")
print(f"Model B method: {model_b_method}")

supported_methods = {"erm", "dann"}
if model_a_method not in supported_methods or model_b_method not in supported_methods:
    raise NotImplementedError(
        "This notebook currently supports training.method in {'erm', 'dann'} only. "
        f"Got model_a={model_a_method}, model_b={model_b_method}."
    )

model_a_name = os.environ.get("RQ4_MODEL_A_LABEL", f"A:{model_a_method.upper()}")
model_b_name = os.environ.get("RQ4_MODEL_B_LABEL", f"B:{model_b_method.upper()}")
patched_model_name = f"Patched {model_a_name}"

print(f"\nModel labels: {model_a_name} vs {model_b_name}")
print("DANN domain objective is applied on pre-projection pooled features.")
print("With a frozen backbone, strongest differences are expected after backbone hidden states.")

# Backward-compatible aliases for existing downstream variable names.
erm_model, erm_config = model_a_model, model_a_config
dann_model, dann_config = model_b_model, model_b_config

print(f"\nCodec vocab size: {len(codec_vocab)}")
print(f"Codec Q vocab size: {len(codec_q_vocab)}")


# ## 2. CKA Analysis
# 
# Centered Kernel Alignment (CKA) measures representational similarity between layers.
# We compute CKA between ERM and DANN at each layer to identify where they diverge most.
# 
# The library's `compute_linear_cka` function already handles numerical stability with epsilon.

# In[ ]:


def create_eval_dataloader(
    split: str = "dev",
    codec_vocab: dict = None,
    codec_q_vocab: dict = None,
    config: dict = None,
    max_samples: int = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create dataloader for CKA analysis and evaluation.

    Args:
        split: Data split ('dev' or 'eval').
        codec_vocab: CODEC vocabulary.
        codec_q_vocab: CODEC_Q vocabulary.
        config: Model config dict.
        max_samples: Maximum samples to use (None for all).
        batch_size: Batch size.
        num_workers: Number of workers.
        seed: Random seed for subset selection.

    Returns:
        DataLoader instance.
    """
    audio_cfg = config.get("audio", {}) if config else {}
    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)

    manifest_path = get_manifest_path(split)
    print(f"Loading manifest from: {manifest_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run scripts/prepare_asvspoof5.py after setting ASVSPOOF5_ROOT."
        )

    dataset = ASVspoof5Dataset(
        manifest_path=manifest_path,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    print(f"Dataset size: {len(dataset)}")

    # Subsample if needed
    if max_samples and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subsampled to {len(dataset)} samples")

    fixed_length = int(max_duration * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


# In[ ]:


def _get_selected_layer_indices(backbone: torch.nn.Module, total_layers: int) -> list[int]:
    """Resolve which hidden-state indices participate in backbone layer mixing."""
    selection = getattr(backbone, "layer_selection", "weighted")
    k = int(getattr(backbone, "k", total_layers))
    explicit_indices = getattr(backbone, "layer_indices", None)

    if selection == "first_k":
        return list(range(min(k, total_layers)))
    if selection == "last_k":
        start = max(0, total_layers - k)
        return list(range(start, total_layers))
    if selection == "specific" and explicit_indices:
        return [int(idx) for idx in explicit_indices if 0 <= int(idx) < total_layers]

    # weighted / fallback: use all transformer layers
    return list(range(total_layers))


def _get_model_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Return a backbone for representation extraction.

    Supports both plain models (ERM/DANN) and PatchedModel wrappers.
    """
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "base_model") and hasattr(model.base_model, "backbone"):
        return model.base_model.backbone
    raise RuntimeError(
        f"Model of type {type(model).__name__} does not expose a compatible backbone."
    )


@torch.no_grad()
def extract_layer_representations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = None,
    representation: str = "layer_contrib",
) -> dict:
    """Extract model representations for CKA.

    Supported representations: hidden_states, mixed, repr, layer_contrib.
    """
    model.eval()
    layer_reps = {}
    total_batches = num_batches if num_batches else len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"Extracting ({representation})")):
        if num_batches and batch_idx >= num_batches:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)
        all_hidden_states = outputs.get("all_hidden_states", [])
        if not all_hidden_states:
            raise RuntimeError("Model did not return all_hidden_states. Check model forward method.")

        if representation == "hidden_states":
            for layer_idx, hidden_state in enumerate(all_hidden_states):
                pooled = hidden_state.mean(dim=1)
                layer_reps.setdefault(layer_idx, []).append(pooled.cpu())

        elif representation == "repr":
            if "repr" not in outputs:
                raise RuntimeError("Model did not return repr. Cannot compute CKA with representation='repr'.")
            layer_reps.setdefault("repr", []).append(outputs["repr"].cpu())

        elif representation == "mixed":
            backbone = _get_model_backbone(model)
            mixed, _ = backbone(waveform, attention_mask)
            layer_reps.setdefault("mixed", []).append(mixed.mean(dim=1).cpu())

        elif representation == "layer_contrib":
            backbone = _get_model_backbone(model)
            total_layers = len(all_hidden_states)
            selected_indices = _get_selected_layer_indices(backbone, total_layers)
            selected_states = [all_hidden_states[idx] for idx in selected_indices]

            layer_pooling = getattr(backbone, "layer_pooling", None)
            if layer_pooling is None or not hasattr(layer_pooling, "weights"):
                raise RuntimeError("Backbone missing layer_pooling.weights needed for layer_contrib extraction.")

            weights = torch.softmax(layer_pooling.weights.detach(), dim=0)
            if weights.numel() != len(selected_states):
                raise RuntimeError(
                    "Layer weight count does not match selected states: "
                    f"weights={weights.numel()} states={len(selected_states)}"
                )

            for local_idx, (layer_idx, hidden_state) in enumerate(zip(selected_indices, selected_states)):
                contribution = hidden_state * weights[local_idx]
                pooled = contribution.mean(dim=1)
                layer_reps.setdefault(int(layer_idx), []).append(pooled.cpu())

        else:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                "Expected one of: hidden_states, mixed, repr, layer_contrib"
            )

    result = {}
    for key, rep_list in layer_reps.items():
        if rep_list:
            result[key] = torch.cat(rep_list, dim=0)

    print(f"Extracted representations for {len(result)} keys: {sorted(result.keys(), key=lambda x: str(x))}")
    return result


# In[ ]:


# Select representation used by CKA and downstream patch-layer selection.
# Use layer_contrib by default; hidden_states is usually trivial (CKA~1.0) with frozen backbones.
cka_representation = "layer_contrib"
valid_representations = {"hidden_states", "mixed", "repr", "layer_contrib"}
if cka_representation not in valid_representations:
    raise ValueError(f"Invalid cka_representation={cka_representation}. Choose from {sorted(valid_representations)}")

# Intervention mode controls
available_intervention_modes = [
    "layer_patch_hidden",
    "layer_patch_mixed",
    "layer_patch_repr",
    "pool_weight_transplant",
]
run_all_modes = os.environ.get("RQ4_RUN_ALL_MODES", "true").lower() == "true"
configured_mode = os.environ.get("RQ4_INTERVENTION_MODE", "layer_patch_hidden")
if configured_mode not in available_intervention_modes:
    raise ValueError(
        f"Invalid RQ4_INTERVENTION_MODE={configured_mode}. "
        f"Choose from {available_intervention_modes}"
    )
selected_intervention_modes = (
    list(available_intervention_modes) if run_all_modes else [configured_mode]
)
print(f"Selected intervention modes: {selected_intervention_modes}")

if cka_representation == "hidden_states":
    is_frozen_base = bool(erm_config.get("backbone", {}).get("freeze", True))
    is_frozen_donor = bool(dann_config.get("backbone", {}).get("freeze", True))
    if is_frozen_base and is_frozen_donor:
        print(
            "Warning: hidden_states with frozen SSL backbones typically yields CKA~1.0. "
            "Use 'layer_contrib', 'mixed', or 'repr' for a more informative comparison."
        )

# Create dataloader for CKA analysis (use subset for speed)
print(f"Creating evaluation dataloader for CKA analysis ({cka_representation})...")
eval_loader = create_eval_dataloader(
    split=rq4_eval_split,
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=rq4_max_cka_samples,
    batch_size=32,
)


# In[ ]:


# Extract representations from model A
print(f"Extracting {model_a_name} representations ({cka_representation})...")
erm_reps = extract_layer_representations(
    erm_model,
    eval_loader,
    device,
    representation=cka_representation,
)
print(f"Extracted {len(erm_reps)} keys, shapes: {[(k, v.shape) for k, v in erm_reps.items()]}")


# In[ ]:


# Reset dataloader for model B
eval_loader = create_eval_dataloader(
    split=rq4_eval_split,
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=dann_config,
    max_samples=rq4_max_cka_samples,
    batch_size=32,
)

# Extract representations from model B
print(f"Extracting {model_b_name} representations ({cka_representation})...")
dann_reps = extract_layer_representations(
    dann_model,
    eval_loader,
    device,
    representation=cka_representation,
)
print(f"Extracted {len(dann_reps)} keys, shapes: {[(k, v.shape) for k, v in dann_reps.items()]}")


# In[ ]:


# Compute CKA per representation key
print(f"Computing CKA between {model_a_name} and {model_b_name} using '{cka_representation}'...")
cka_results = compare_representations(
    erm_layers={k: v.numpy() for k, v in erm_reps.items()},
    dann_layers={k: v.numpy() for k, v in dann_reps.items()},
)

print("\nCKA Results:")
print(f"Model pair: {model_a_name} vs {model_b_name}")
print(f"Representation mode: {cka_representation}")
print(f"Mean CKA: {cka_results['mean_cka']:.4f}")
print(f"Min CKA:  {cka_results['min_cka']:.4f}")
print(f"Max CKA:  {cka_results['max_cka']:.4f}")
print(f"Most different key: {cka_results['most_different_layer']}")

print("\nPer-key CKA:")
for key in sorted(cka_results['per_layer'].keys(), key=lambda x: str(x)):
    cka = cka_results['per_layer'][key]['cka']
    print(f"  Key {key}: CKA = {cka:.4f}")

cka_layer_keys = [k for k in cka_results['per_layer'].keys() if isinstance(k, int)]
if len(cka_layer_keys) < 3:
    raise RuntimeError(
        "Activation patching needs at least 3 integer transformer-layer keys from CKA. "
        f"Current keys: {sorted(cka_results['per_layer'].keys(), key=lambda x: str(x))}. "
        "Use cka_representation='hidden_states' or 'layer_contrib'."
    )


# In[ ]:


# Plot CKA scores for integer layer keys (needed for activation patching)
layers = sorted([k for k in cka_results['per_layer'].keys() if isinstance(k, int)])
if len(layers) < 3:
    raise RuntimeError(
        "Cannot plot patch-layer CKA bars because fewer than 3 integer layer keys were found. "
        "Use cka_representation='hidden_states' or 'layer_contrib'."
    )

cka_scores = [cka_results['per_layer'][l]['cka'] for l in layers]

plt.figure(figsize=(12, 5))
bars = plt.bar(layers, cka_scores, color='steelblue', edgecolor='black')

# Highlight the most divergent layers
sorted_by_cka = sorted(layers, key=lambda l: cka_results['per_layer'][l]['cka'])
divergent_layers = sorted_by_cka[:3]
for i, l in enumerate(layers):
    if l in divergent_layers:
        bars[i].set_color('tomato')

plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('CKA Similarity', fontsize=12)
plt.title(
    f"{model_a_name} vs {model_b_name} Representation Similarity (CKA, {cka_representation})\nRed = Most Divergent Layers",
    fontsize=14,
)
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='High similarity threshold')
plt.ylim(0, 1.05)
plt.xticks(layers)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nMost divergent layers (lowest CKA): {divergent_layers}")
cka_value_strings = [f"{cka_results['per_layer'][l]['cka']:.4f}" for l in divergent_layers]
print(f"CKA values: {cka_value_strings}")


# ## 3. Activation Patching
# 
# Replace ERM activations at specific layers with DANN activations during inference.
# 
# We use forward hooks to:
# 1. Capture DANN activations at target layers
# 2. Replace ERM activations with DANN activations at those layers

# In[ ]:


class PatchedModel(torch.nn.Module):
    """Model that patches activations from a donor model at specified layers.

    Hooks into the backbone's internal transformer layers to replace
    hidden states from model_a with those from model_b.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        donor_model: torch.nn.Module,
        patch_layers: list[int],
    ):
        """
        Args:
            base_model: The model to run inference on.
            donor_model: The model to take activations from.
            patch_layers: List of layer indices to patch (0-indexed transformer layers).
        """
        super().__init__()
        self.base_model = base_model
        self.donor_model = donor_model
        self.patch_layers = set(patch_layers)

        # Keep deterministic execution while patching.
        self.base_model.eval()
        self.donor_model.eval()

        self._donor_activations = {}
        self._patching_active = False
        self._handles = []
        self._setup_hooks()

    def _get_layer_name(self, layer_idx: int) -> str:
        """Get the full module name for a transformer layer.

        For WavLM: backbone.model.encoder.layers.{layer_idx}
        Note: layer_idx 0 corresponds to the first transformer layer (after CNN encoder)
        """
        return f"backbone.model.encoder.layers.{layer_idx}"

    def _setup_hooks(self):
        """Setup forward hooks to capture and replace activations."""
        for layer_idx in self.patch_layers:
            layer_name = self._get_layer_name(layer_idx)

            # Find module in donor
            donor_module = dict(self.donor_model.named_modules()).get(layer_name)
            if donor_module is None:
                raise ValueError(f"Layer {layer_name} not found in donor model")

            def make_capture_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self._donor_activations[idx] = output[0].detach()
                    else:
                        self._donor_activations[idx] = output.detach()

                return hook

            handle = donor_module.register_forward_hook(make_capture_hook(layer_idx))
            self._handles.append(handle)

        # Register patching hooks on base model
        for layer_idx in self.patch_layers:
            layer_name = self._get_layer_name(layer_idx)

            base_module = dict(self.base_model.named_modules()).get(layer_name)
            if base_module is None:
                raise ValueError(f"Layer {layer_name} not found in base model")

            def make_patch_hook(idx):
                def hook(module, input, output):
                    if not self._patching_active:
                        return output
                    if idx not in self._donor_activations:
                        return output

                    donor_act = self._donor_activations[idx]
                    base_hidden = output[0] if isinstance(output, tuple) else output
                    if donor_act.shape != base_hidden.shape:
                        raise RuntimeError(
                            "Activation shape mismatch during patching at layer "
                            f"{idx}: donor={tuple(donor_act.shape)} vs base={tuple(base_hidden.shape)}"
                        )

                    if isinstance(output, tuple):
                        # Keep tuple tail (including position_bias) from the base forward.
                        # Recomputing position_bias in later layers can fail for WavLM blocks
                        # that do not own relative-bias embeddings.
                        return (donor_act, *output[1:])
                    return donor_act

                return hook

            handle = base_module.register_forward_hook(make_patch_hook(layer_idx))
            self._handles.append(handle)

        print(f"Registered {len(self._handles)} hooks for patching layers {sorted(self.patch_layers)}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        """Forward pass with activation patching.

        1. Run donor model to capture activations
        2. Run base model with patched activations (via hooks)
        """
        self._donor_activations.clear()

        # Run donor to capture activations
        with torch.no_grad():
            _ = self.donor_model(waveform, attention_mask, lengths)

        missing_layers = sorted(self.patch_layers.difference(self._donor_activations.keys()))
        if missing_layers:
            raise RuntimeError(f"Missing donor activations for patch layers: {missing_layers}")

        # Run base model (hooks patch only while this flag is enabled)
        self._patching_active = True
        try:
            output = self.base_model(waveform, attention_mask, lengths)
        finally:
            self._patching_active = False
            self._donor_activations.clear()

        return output

    def __del__(self):
        """Cleanup hooks on deletion."""
        # Interpreter shutdown can leave torch internals partially torn down.
        try:
            self.remove_hooks()
        except Exception:
            pass


class MixedPatchedModel(torch.nn.Module):
    """Replace base mixed representation with donor mixed representation."""

    def __init__(self, base_model: torch.nn.Module, donor_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        self.donor_model = donor_model
        self.base_model.eval()
        self.donor_model.eval()

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        mixed_base, all_hidden_states_base = self.base_model.backbone(waveform, attention_mask)
        mixed_donor, _ = self.donor_model.backbone(waveform, attention_mask)
        if mixed_base.shape != mixed_donor.shape:
            raise RuntimeError(
                "Mixed representation shape mismatch: "
                f"base={tuple(mixed_base.shape)} donor={tuple(mixed_donor.shape)}"
            )
        pooled = self.base_model.pooling(mixed_donor, lengths)
        repr_ = self.base_model.projection(pooled)
        task_logits = self.base_model.task_head(repr_)
        return {
            "task_logits": task_logits,
            "repr": repr_,
            "all_hidden_states": all_hidden_states_base,
        }


class ReprPatchedModel(torch.nn.Module):
    """Replace base repr with donor repr before task head."""

    def __init__(self, base_model: torch.nn.Module, donor_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        self.donor_model = donor_model
        self.base_model.eval()
        self.donor_model.eval()

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        base_outputs = self.base_model(waveform, attention_mask, lengths)
        donor_outputs = self.donor_model(waveform, attention_mask, lengths)
        base_repr = base_outputs["repr"]
        donor_repr = donor_outputs["repr"]
        if base_repr.shape != donor_repr.shape:
            raise RuntimeError(
                "Repr shape mismatch: "
                f"base={tuple(base_repr.shape)} donor={tuple(donor_repr.shape)}"
            )
        task_logits = self.base_model.task_head(donor_repr)
        return {
            "task_logits": task_logits,
            "repr": donor_repr,
            "all_hidden_states": base_outputs["all_hidden_states"],
        }


def compute_mode_cka(
    base_model: torch.nn.Module,
    donor_model: torch.nn.Module,
    base_config: dict,
    donor_config: dict,
    representation: str,
    codec_vocab: dict,
    codec_q_vocab: dict,
    device: torch.device,
    split: str,
    max_samples: int = 5000,
) -> dict:
    """Compute CKA for a chosen representation space."""
    mode_loader = create_eval_dataloader(
        split=split,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        config=base_config,
        max_samples=max_samples,
        batch_size=32,
    )
    base_reps = extract_layer_representations(
        base_model,
        mode_loader,
        device,
        representation=representation,
    )
    mode_loader = create_eval_dataloader(
        split=split,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        config=donor_config,
        max_samples=max_samples,
        batch_size=32,
    )
    donor_reps = extract_layer_representations(
        donor_model,
        mode_loader,
        device,
        representation=representation,
    )
    return compare_representations(
        erm_layers={k: v.numpy() for k, v in base_reps.items()},
        dann_layers={k: v.numpy() for k, v in donor_reps.items()},
    )


def select_divergent_transformer_layers(base_model: torch.nn.Module, cka_results: dict, top_k: int = 3) -> list[int]:
    """Select top-k lowest CKA transformer layers that exist in base model."""
    cka_per_layer = cka_results["per_layer"]
    sorted_layers = sorted(
        [k for k in cka_per_layer.keys() if isinstance(k, int)],
        key=lambda k: cka_per_layer[k]["cka"],
    )
    available_transformer_layers = {
        int(name.split(".")[-1])
        for name, _ in base_model.named_modules()
        if name.startswith("backbone.model.encoder.layers.") and name.split(".")[-1].isdigit()
    }
    selected = [layer_idx for layer_idx in sorted_layers if layer_idx in available_transformer_layers][:top_k]
    if len(selected) < top_k:
        raise RuntimeError(
            f"Expected at least {top_k} patchable transformer layers, found {len(selected)}. "
            f"Available layers: {sorted(available_transformer_layers)}"
        )
    return selected

# In[ ]:


def make_intervention_model(mode: str, divergent_layers: list[int]) -> torch.nn.Module:
    """Build intervention model for a given mode."""
    if mode == "layer_patch_hidden":
        model = PatchedModel(
            base_model=erm_model,
            donor_model=dann_model,
            patch_layers=divergent_layers,
        )
        model.eval()
        return model
    if mode == "layer_patch_mixed":
        model = MixedPatchedModel(base_model=erm_model, donor_model=dann_model)
        model.eval()
        return model
    if mode == "layer_patch_repr":
        model = ReprPatchedModel(base_model=erm_model, donor_model=dann_model)
        model.eval()
        return model
    if mode == "pool_weight_transplant":
        return erm_model
    raise ValueError(f"Unknown intervention mode: {mode}")

# In[ ]:


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute EER and collect predictions for analysis.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        device: Computation device.

    Returns:
        dict with 'eer', 'min_dcf', 'scores', 'labels'
    """
    model.eval()
    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        waveform = batch['waveform'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lengths = batch['lengths'].to(device)
        # Defensive access - handle case where y_task might not be present
        labels = batch.get('y_task')
        if labels is None:
            raise ValueError(
                "Batch missing 'y_task' field. Ensure dataloader/collator "
                "returns task labels in eval mode."
            )

        outputs = model(waveform, attention_mask, lengths)

        # Score convention in this codebase: higher score = more likely bonafide
        # Labels: 0 = bonafide, 1 = spoof
        probs = torch.softmax(outputs['task_logits'], dim=-1)
        scores = probs[:, 0].cpu().numpy()  # P(bonafide)

        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    eer, threshold = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)

    return {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': threshold,
        'scores': scores,
        'labels': labels,
    }


# In[ ]:


# Create fresh dataloader for evaluation (full dev set or subset)
print("Creating evaluation dataloader...")
eval_loader = create_eval_dataloader(
    split=rq4_eval_split,
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=rq4_max_eval_samples,
    batch_size=32,
)


# In[ ]:


# Evaluate model A baseline
print(f"Evaluating {model_a_name}...")
erm_results = evaluate_model(erm_model, eval_loader, device)
print(f"{model_a_name} EER: {erm_results['eer']:.2%}, minDCF: {erm_results['min_dcf']:.4f}")


# In[ ]:


# Recreate dataloader for model B
eval_loader = create_eval_dataloader(
    split=rq4_eval_split,
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=dann_config,
    max_samples=rq4_max_eval_samples,
    batch_size=32,
)

# Evaluate model B baseline
print(f"Evaluating {model_b_name}...")
dann_results = evaluate_model(dann_model, eval_loader, device)
print(f"{model_b_name} EER: {dann_results['eer']:.2%}, minDCF: {dann_results['min_dcf']:.4f}")


# In[ ]:


# Baseline-only comparison; intervention modes are evaluated in ablation loop below.
print("\n" + "=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)
print(f"{'Model':<20} {'EER':>10} {'minDCF':>10}")
print("-" * 60)
print(f"{model_a_name:<20} {erm_results['eer']:>9.2%} {erm_results['min_dcf']:>10.4f}")
print(f"{model_b_name:<20} {dann_results['eer']:>9.2%} {dann_results['min_dcf']:>10.4f}")
print("=" * 60)

# In[ ]:


def extract_representations_for_probing(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 5000,
    representation: str = "hidden_states",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Extract representations and domain labels for probing.

    Supported representations: hidden_states, mixed, repr, layer_contrib.
    """
    model.eval()
    layer_reps = {}
    all_codec = []
    all_codec_q = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting for probing ({representation})"):
            if max_samples and n_samples >= max_samples:
                break

            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            outputs = model(waveform, attention_mask, lengths)
            all_hidden_states = outputs.get("all_hidden_states", [])
            batch_size = waveform.shape[0]

            if representation == "hidden_states":
                for layer_idx, hidden_state in enumerate(all_hidden_states):
                    pooled = hidden_state.mean(dim=1).cpu().numpy()
                    layer_reps.setdefault(layer_idx, []).append(pooled)

            elif representation == "repr":
                if "repr" not in outputs:
                    raise RuntimeError("Model did not return repr. Cannot probe representation='repr'.")
                layer_reps.setdefault("repr", []).append(outputs["repr"].cpu().numpy())

            elif representation == "mixed":
                backbone = _get_model_backbone(model)
                mixed, _ = backbone(waveform, attention_mask)
                layer_reps.setdefault("mixed", []).append(mixed.mean(dim=1).cpu().numpy())

            elif representation == "layer_contrib":
                backbone = _get_model_backbone(model)
                total_layers = len(all_hidden_states)
                selected_indices = _get_selected_layer_indices(backbone, total_layers)
                selected_states = [all_hidden_states[idx] for idx in selected_indices]

                layer_pooling = getattr(backbone, "layer_pooling", None)
                if layer_pooling is None or not hasattr(layer_pooling, "weights"):
                    raise RuntimeError("Backbone missing layer_pooling.weights needed for layer_contrib probing.")

                weights = torch.softmax(layer_pooling.weights.detach(), dim=0)
                if weights.numel() != len(selected_states):
                    raise RuntimeError(
                        "Layer weight count does not match selected states: "
                        f"weights={weights.numel()} states={len(selected_states)}"
                    )

                for local_idx, (layer_idx, hidden_state) in enumerate(zip(selected_indices, selected_states)):
                    contribution = hidden_state * weights[local_idx]
                    pooled = contribution.mean(dim=1).cpu().numpy()
                    layer_reps.setdefault(int(layer_idx), []).append(pooled)

            else:
                raise ValueError(
                    f"Unknown representation '{representation}'. "
                    "Expected one of: hidden_states, mixed, repr, layer_contrib"
                )

            # Defensive access - y_codec/y_codec_q may not be present in all eval modes
            y_codec = batch.get("y_codec")
            y_codec_q = batch.get("y_codec_q")
            if y_codec is not None:
                all_codec.append(y_codec.numpy())
            if y_codec_q is not None:
                all_codec_q.append(y_codec_q.numpy())

            n_samples += batch_size

    for key in list(layer_reps.keys()):
        layer_reps[key] = np.concatenate(layer_reps[key], axis=0)
        if max_samples:
            layer_reps[key] = layer_reps[key][:max_samples]

    if not all_codec:
        raise ValueError("No y_codec labels were found in batches. Cannot run codec probing.")
    if not all_codec_q:
        raise ValueError("No y_codec_q labels were found in batches. Cannot run codec_q probing.")

    all_codec = np.concatenate(all_codec)
    all_codec_q = np.concatenate(all_codec_q)
    if max_samples:
        all_codec = all_codec[:max_samples]
        all_codec_q = all_codec_q[:max_samples]

    return layer_reps, all_codec, all_codec_q


# In[ ]:


# Select representation for domain probing.
# Keep hidden_states to preserve layer-wise leakage profile.
probe_representation = "hidden_states"
probe_split = rq4_probe_split

# Create dataloader for probing
probe_loader = create_eval_dataloader(
    split=probe_split,
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=rq4_max_probe_samples,
    batch_size=32,
)

# Extract model A representations for probing
print(f"Extracting {model_a_name} representations for probing ({probe_representation})...")
erm_reps_probe, codec_labels, codec_q_labels = extract_representations_for_probing(
    erm_model,
    probe_loader,
    device,
    max_samples=rq4_max_probe_samples,
    representation=probe_representation,
)
print(f"Extracted representation keys: {list(erm_reps_probe.keys())[:10]}")
codec_unique = int(np.unique(codec_labels).shape[0])
codec_q_unique = int(np.unique(codec_q_labels).shape[0])
print(
    f"Probe split: {probe_split} | "
    f"CODEC unique={codec_unique} | CODEC_Q unique={codec_q_unique}"
)
def choose_probe_target(codec_labels: np.ndarray, codec_q_labels: np.ndarray, split: str) -> tuple[str, np.ndarray]:
    """Pick probe target that has at least 2 classes."""
    codec_unique = int(np.unique(codec_labels).shape[0])
    codec_q_unique = int(np.unique(codec_q_labels).shape[0])
    print(
        f"Probe split: {split} | "
        f"CODEC unique={codec_unique} | CODEC_Q unique={codec_q_unique}"
    )
    if codec_unique >= 2:
        return "CODEC", codec_labels
    if codec_q_unique >= 2:
        print("Warning: CODEC labels are single-class; falling back to CODEC_Q probing.")
        return "CODEC_Q", codec_q_labels
    raise ValueError(
        "Both CODEC and CODEC_Q labels are single-class on the selected probe split, "
        f"so probing is undefined. split={split}, codec_unique={codec_unique}, "
        f"codec_q_unique={codec_q_unique}. Try setting RQ4_PROBE_SPLIT to another split."
    )


def to_probe_inputs(representations: dict) -> dict[int, np.ndarray]:
    """Convert representation dict into layerwise_probing-compatible integer keys."""
    int_keys = sorted([k for k in representations.keys() if isinstance(k, int)])
    if int_keys:
        return {k: representations[k] for k in int_keys}
    ordered_keys = sorted(representations.keys(), key=lambda x: str(x))
    return {idx: representations[key] for idx, key in enumerate(ordered_keys)}


def run_probe_for_model(
    model: torch.nn.Module,
    representation: str,
    split: str,
    max_samples: int,
) -> dict:
    """Extract representations and run domain probing for a model."""
    probe_loader_local = create_eval_dataloader(
        split=split,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        config=erm_config,
        max_samples=max_samples,
        batch_size=32,
    )
    model_reps, codec_labels_local, codec_q_labels_local = extract_representations_for_probing(
        model,
        probe_loader_local,
        device,
        max_samples=max_samples,
        representation=representation,
    )
    probe_target_name_local, probe_labels_local = choose_probe_target(
        codec_labels_local,
        codec_q_labels_local,
        split,
    )
    probe_inputs = to_probe_inputs(model_reps)
    if not probe_inputs:
        raise RuntimeError(
            "No valid probing inputs extracted. "
            f"representation={representation}, keys={list(model_reps.keys())}"
        )
    probe_results = layerwise_probing(
        probe_inputs,
        probe_labels_local,
        classifier="logistic",
        cv_folds=5,
        seed=42,
    )
    return {
        "target_name": probe_target_name_local,
        "labels": probe_labels_local,
        "inputs": probe_inputs,
        "results": probe_results,
    }


def copy_layer_pooling_weights_from_donor(base_model: torch.nn.Module, donor_model: torch.nn.Module) -> tuple[torch.Tensor, dict]:
    """Copy donor layer-pooling weights into base model; return original for restore."""
    base_pool = getattr(base_model.backbone, "layer_pooling", None)
    donor_pool = getattr(donor_model.backbone, "layer_pooling", None)
    if base_pool is None or donor_pool is None:
        raise RuntimeError("Missing layer_pooling in base/donor backbone for pool_weight_transplant mode.")
    base_weights = getattr(base_pool, "weights", None)
    donor_weights = getattr(donor_pool, "weights", None)
    if base_weights is None or donor_weights is None:
        raise RuntimeError("Missing layer_pooling.weights in base/donor backbone.")
    if base_weights.shape != donor_weights.shape:
        raise RuntimeError(
            "Layer-pooling weight shape mismatch: "
            f"base={tuple(base_weights.shape)} donor={tuple(donor_weights.shape)}"
        )
    original = base_weights.detach().clone()
    with torch.no_grad():
        base_weights.copy_(donor_weights.detach())
    delta = (donor_weights.detach() - original).cpu().numpy()
    stats = {
        "l2_delta": float(np.linalg.norm(delta)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "num_weights": int(delta.shape[0]),
    }
    return original, stats


mode_to_cka_representation = {
    "layer_patch_hidden": "hidden_states",
    "layer_patch_mixed": "mixed",
    "layer_patch_repr": "repr",
    "pool_weight_transplant": "layer_contrib",
}
mode_to_probe_representation = {
    "layer_patch_hidden": "hidden_states",
    "layer_patch_mixed": "mixed",
    "layer_patch_repr": "repr",
    "pool_weight_transplant": "layer_contrib",
}

mode_results = []
mode_cka_rows = []
baseline_probe_cache = {}
created_patch_models = []
mode_eval_cache: dict[str, dict[str, np.ndarray]] = {}

for mode in selected_intervention_modes:
    print("\n" + "=" * 100)
    print(f"RUNNING INTERVENTION MODE: {mode}")
    print("=" * 100)

    mode_cka_rep = mode_to_cka_representation[mode]
    mode_probe_rep = mode_to_probe_representation[mode]
    mode_cka = compute_mode_cka(
        erm_model,
        dann_model,
        erm_config,
        dann_config,
        mode_cka_rep,
        codec_vocab,
        codec_q_vocab,
        device,
        split=rq4_eval_split,
        max_samples=rq4_max_cka_samples,
    )
    print(
        f"Mode {mode} CKA ({mode_cka_rep}): "
        f"mean={mode_cka['mean_cka']:.4f} min={mode_cka['min_cka']:.4f} max={mode_cka['max_cka']:.4f}"
    )
    for layer_key, layer_data in mode_cka["per_layer"].items():
        mode_cka_rows.append(
            {
                "mode": mode,
                "representation_mode": mode_cka_rep,
                "layer_key": str(layer_key),
                "cka": float(layer_data["cka"]),
            }
        )

    divergent_layers = []
    intervention_notes = ""
    intervention_model = None
    transplanted_original_weights = None
    transplant_stats = None

    if mode == "layer_patch_hidden":
        divergent_layers = select_divergent_transformer_layers(erm_model, mode_cka, top_k=3)
        intervention_model = make_intervention_model(mode, divergent_layers)
        created_patch_models.append(intervention_model)
        intervention_notes = f"layers={divergent_layers}"
    elif mode in {"layer_patch_mixed", "layer_patch_repr"}:
        intervention_model = make_intervention_model(mode, divergent_layers)
        created_patch_models.append(intervention_model)
        intervention_notes = f"global_{mode_cka_rep}_patch"
    elif mode == "pool_weight_transplant":
        intervention_model = make_intervention_model(mode, divergent_layers)
        transplanted_original_weights, transplant_stats = copy_layer_pooling_weights_from_donor(
            erm_model,
            dann_model,
        )
        intervention_notes = (
            "pool_weight_transplant "
            f"(l2_delta={transplant_stats['l2_delta']:.6f}, "
            f"max_abs_delta={transplant_stats['max_abs_delta']:.6f})"
        )
    else:
        raise ValueError(f"Unsupported intervention mode: {mode}")

    mode_eval_loader = create_eval_dataloader(
        split=rq4_eval_split,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        config=erm_config,
        max_samples=rq4_max_eval_samples,
        batch_size=32,
    )
    try:
        mode_eval = evaluate_model(intervention_model, mode_eval_loader, device)
        print(f"Mode {mode} EER: {mode_eval['eer']:.2%}, minDCF: {mode_eval['min_dcf']:.4f}")

        # Baseline probe cache by representation to keep fair deltas.
        if mode_probe_rep not in baseline_probe_cache:
            baseline_probe_cache[mode_probe_rep] = run_probe_for_model(
                erm_model,
                representation=mode_probe_rep,
                split=probe_split,
                max_samples=rq4_max_probe_samples,
            )
        baseline_probe = baseline_probe_cache[mode_probe_rep]
        mode_probe = run_probe_for_model(
            intervention_model,
            representation=mode_probe_rep,
            split=probe_split,
            max_samples=rq4_max_probe_samples,
        )
    finally:
        if transplanted_original_weights is not None:
            with torch.no_grad():
                erm_model.backbone.layer_pooling.weights.copy_(transplanted_original_weights)

    baseline_probe_acc = float(baseline_probe["results"]["max_leakage_accuracy"])
    mode_probe_acc = float(mode_probe["results"]["max_leakage_accuracy"])
    probe_target_unique = int(np.unique(mode_probe["labels"]).shape[0])
    probe_chance_acc = 1.0 / probe_target_unique if probe_target_unique > 0 else 0.0
    mode_eval_cache[mode] = {
        "scores": np.asarray(mode_eval["scores"]),
        "labels": np.asarray(mode_eval["labels"]),
    }
    mode_results.append(
        {
            "mode": mode,
            "model_label": f"{patched_model_name}:{mode}",
            "eer": float(mode_eval["eer"]),
            "min_dcf": float(mode_eval["min_dcf"]),
            "eval_split": rq4_eval_split,
            "probe_split": probe_split,
            "max_eval_samples": rq4_max_eval_samples if rq4_max_eval_samples is not None else -1,
            "max_cka_samples": rq4_max_cka_samples if rq4_max_cka_samples is not None else -1,
            "max_probe_samples": rq4_max_probe_samples if rq4_max_probe_samples is not None else -1,
            "probe_representation": mode_probe_rep,
            "probe_target": mode_probe["target_name"],
            "codec_unique": int(codec_unique),
            "codec_q_unique": int(codec_q_unique),
            "probe_target_unique": probe_target_unique,
            "probe_chance_acc": probe_chance_acc,
            "probe_chance_pct": probe_chance_acc * 100.0,
            "max_probe_acc": mode_probe_acc,
            "probe_num_samples": int(mode_probe["labels"].shape[0]),
            "max_leakage_layer": mode_probe["results"]["max_leakage_layer"],
            "delta_eer_vs_base": float(mode_eval["eer"] - erm_results["eer"]),
            "delta_min_dcf_vs_base": float(mode_eval["min_dcf"] - erm_results["min_dcf"]),
            "delta_probe_vs_base": float(mode_probe_acc - baseline_probe_acc),
            "notes": intervention_notes,
        }
    )

mode_results_df = pd.DataFrame(mode_results)
mode_results_df = mode_results_df.sort_values(["delta_eer_vs_base", "delta_probe_vs_base"], ascending=[True, True])

print("\n" + "=" * 120)
print("INTERVENTION ABLATION RESULTS")
print("=" * 120)
if not mode_results_df.empty:
    print(mode_results_df.to_string(index=False))
else:
    print("No intervention modes were executed.")
print("=" * 120)

# Balanced rule: no regression on both EER and minDCF, then best leakage reduction.
eligible_df = mode_results_df[
    (mode_results_df["delta_eer_vs_base"] <= 0.0) &
    (mode_results_df["delta_min_dcf_vs_base"] <= 0.0)
]
if not eligible_df.empty:
    recommended_row = eligible_df.sort_values("delta_probe_vs_base", ascending=True).iloc[0]
    recommendation_reason = "no-regression candidate with strongest leakage reduction"
else:
    recommended_row = mode_results_df.sort_values(
        ["delta_eer_vs_base", "delta_min_dcf_vs_base", "delta_probe_vs_base"],
        ascending=[True, True, True],
    ).iloc[0]
    recommendation_reason = "best trade-off (all candidates had some regression)"

print("\nRecommended mode:")
print(
    f"- mode={recommended_row['mode']} | "
    f"delta_eer_vs_base={recommended_row['delta_eer_vs_base']:+.4f} | "
    f"delta_min_dcf_vs_base={recommended_row['delta_min_dcf_vs_base']:+.4f} | "
    f"delta_probe_vs_base={recommended_row['delta_probe_vs_base']:+.4f} "
    f"({recommendation_reason})"
)

# Save results to files
results_path = resolve_rq4_output_path(f"{rq4_output_prefix}_results_summary.csv")
mode_results_df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")

cka_df = pd.DataFrame(mode_cka_rows)
cka_path = resolve_rq4_output_path(f"{rq4_output_prefix}_cka_results.csv")
cka_df.to_csv(cka_path, index=False)
print(f"CKA results saved to {cka_path}")

# Save score-level artifacts for bootstrap/significance tests.
stats_cache_path = resolve_rq4_output_path(f"{rq4_output_prefix}_stats_cache.npz")
stats_cache_payload = {
    "baseline_scores": np.asarray(erm_results["scores"]),
    "baseline_labels": np.asarray(erm_results["labels"]),
    "dann_scores": np.asarray(dann_results["scores"]),
    "dann_labels": np.asarray(dann_results["labels"]),
    "modes": np.array(sorted(mode_eval_cache.keys()), dtype=object),
}
for mode_key, mode_payload in mode_eval_cache.items():
    stats_cache_payload[f"{mode_key}__scores"] = mode_payload["scores"]
    stats_cache_payload[f"{mode_key}__labels"] = mode_payload["labels"]
np.savez_compressed(stats_cache_path, **stats_cache_payload)
print(f"Saved stats cache: {stats_cache_path}")

# Save projection representations for DR visualizations (ERM vs DANN).
erm_repr_probe = run_probe_for_model(
    erm_model,
    representation="repr",
    split=probe_split,
    max_samples=rq4_max_probe_samples,
)
dann_repr_probe = run_probe_for_model(
    dann_model,
    representation="repr",
    split=probe_split,
    max_samples=rq4_max_probe_samples,
)
repr_cache_path = resolve_rq4_output_path(rq4_repr_cache_name)
np.savez_compressed(
    repr_cache_path,
    erm_repr=erm_repr_probe["inputs"][0],
    dann_repr=dann_repr_probe["inputs"][0],
    erm_labels=erm_repr_probe["labels"],
    dann_labels=dann_repr_probe["labels"],
    probe_target=np.array([erm_repr_probe["target_name"]], dtype=object),
    probe_split=np.array([probe_split], dtype=object),
)
print(f"Saved representation cache: {repr_cache_path}")

metadata = {
    "eval_split": rq4_eval_split,
    "probe_split": probe_split,
    "max_eval_samples": rq4_max_eval_samples,
    "max_cka_samples": rq4_max_cka_samples,
    "max_probe_samples": rq4_max_probe_samples,
    "codec_unique": int(codec_unique),
    "codec_q_unique": int(codec_q_unique),
    "result_rows": int(mode_results_df.shape[0]),
    "results_csv": str(results_path),
    "cka_csv": str(cka_path),
    "stats_cache": str(stats_cache_path),
    "repr_cache": str(repr_cache_path),
}
metadata_path = resolve_rq4_output_path(rq4_metadata_name)
with metadata_path.open("w", encoding="utf-8") as metadata_file:
    json.dump(metadata, metadata_file, indent=2)
print(f"Saved RQ4 metadata: {metadata_path}")


# ## Conclusions
# 
# ### Key Findings
# 
# 1. **CKA Analysis**
#    - Compared model pair: `model_a_name` vs `model_b_name`
#    - Representation mode for CKA: `cka_representation`
#    - Most divergent patchable layers: `divergent_layers`
# 
# 2. **Performance Impact**
#    - Base model metric: `erm_results`
#    - Donor model metric: `dann_results`
#    - Patched model metric: `patched_results`
# 
# 3. **Domain Invariance**
#    - Base model max probe leakage: `erm_codec_probes['max_leakage_accuracy']`
#    - Patched model max probe leakage: `patched_codec_probes['max_leakage_accuracy']`
# 
# ### Scope Notes
# 
# - This notebook currently supports checkpoints whose `training.method` is `erm` or `dann`.
# - In this codebase, DANN applies the domain objective to pre-projection pooled features.
# - With frozen backbones, differences are expected mostly in post-backbone representations.
# 
# ### Trade-offs
# 
# - **Computational cost:** Patching runs donor + base during inference (roughly 2x forward passes).
# - **Flexibility:** You can target specific layers without retraining.
# - **Simplicity:** Useful as a lightweight intervention once donor checkpoints exist.
# 
# ### Future Work
# 
# 1. **Selective patching:** Patch only samples likely to benefit.
# 2. **Partial patching:** Interpolate between base and donor activations.
# 3. **Distillation:** Train one model to mimic patched behavior.
# 4. **Multi-layer analysis:** Study interactions across patch sets.

# In[ ]:


# Cleanup
cleaned_hooks = 0
for model in created_patch_models:
    if isinstance(model, PatchedModel):
        model.remove_hooks()
        cleaned_hooks += 1
if cleaned_hooks:
    print(f"Hooks cleaned up for {cleaned_hooks} hook-based intervention model(s).")
else:
    print("No hook-based intervention models found; skipping hook cleanup.")

