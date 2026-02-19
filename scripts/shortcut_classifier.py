#!/usr/bin/env python3
"""Shortcut classifier to bound confound effect (ASVspoof5 §4.2).

This script trains a simple classifier using only the 5 "shortcut" features
identified in the ASVspoof5 paper:
- Peak amplitude
- Leading non-speech duration
- Trailing non-speech duration
- Total duration
- Total energy

If this classifier performs well on train but poorly on dev/eval, it provides
evidence that shortcuts are train-specific and don't explain the OOD gap.

Usage:
    python scripts/shortcut_classifier.py \
        --train-manifest manifests/train.parquet \
        --dev-manifest manifests/dev.parquet \
        --eval-manifest manifests/eval.parquet \
        --data-root $ASVSPOOF5_ROOT \
        --output results/shortcut_classifier.json

References:
    ASVspoof5 Paper §4.2: "Shortcut learning"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_shortcut_features(
    audio_path: Path,
    sample_rate: int = 16000,
    silence_threshold_ratio: float = 0.01,
) -> Optional[dict]:
    """Extract the 5 shortcut features from ASVspoof5 paper.
    
    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate.
        silence_threshold_ratio: Threshold for silence detection as ratio of peak.
        
    Returns:
        Dictionary of features, or None if extraction fails.
    """
    try:
        waveform, sr = sf.read(audio_path)
        if sr != sample_rate:
            # Resample using scipy
            num_samples = int(len(waveform) * sample_rate / sr)
            waveform = signal.resample(waveform, num_samples)
        
        # Handle stereo by taking mean
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        # Handle edge case of empty/silent audio
        if len(waveform) == 0 or np.abs(waveform).max() == 0:
            return None
        
        # 1. Peak amplitude
        peak_amplitude = float(np.abs(waveform).max())
        
        # 2. Total energy (RMS-based)
        total_energy = float(np.sqrt(np.mean(waveform ** 2)))
        
        # 3. Total duration
        total_duration = len(waveform) / sample_rate
        
        # 4/5. Leading/trailing silence (threshold-based VAD)
        threshold = silence_threshold_ratio * peak_amplitude
        above_thresh = np.abs(waveform) > threshold
        
        if above_thresh.any():
            first_sound = np.argmax(above_thresh)
            last_sound = len(waveform) - np.argmax(above_thresh[::-1]) - 1
            leading_silence = first_sound / sample_rate
            trailing_silence = (len(waveform) - 1 - last_sound) / sample_rate
        else:
            # All silence
            leading_silence = total_duration
            trailing_silence = 0.0
        
        return {
            'peak_amplitude': peak_amplitude,
            'total_energy': total_energy,
            'total_duration': total_duration,
            'leading_silence': leading_silence,
            'trailing_silence': trailing_silence,
        }
    except Exception as e:
        logger.warning(f"Failed to extract features from {audio_path}: {e}")
        return None


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer)


def main():
    parser = argparse.ArgumentParser(
        description="Train shortcut classifier to bound confound effect"
    )
    parser.add_argument('--train-manifest', type=Path, required=True,
                        help='Path to training manifest (parquet)')
    parser.add_argument('--dev-manifest', type=Path, required=True,
                        help='Path to dev manifest (parquet)')
    parser.add_argument('--eval-manifest', type=Path, default=None,
                        help='Path to eval manifest (parquet, optional)')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root directory for audio files')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Max samples per split (for speed)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Collect splits to process
    splits = [
        ('train', args.train_manifest),
        ('dev', args.dev_manifest),
    ]
    if args.eval_manifest and args.eval_manifest.exists():
        splits.append(('eval', args.eval_manifest))
    
    results = {}
    
    # Extract features for each split
    for split_name, manifest_path in splits:
        logger.info(f"Extracting features for {split_name}...")
        
        # Support both parquet and protocol txt files
        if manifest_path.suffix == '.parquet':
            df = pd.read_parquet(manifest_path)
        elif manifest_path.suffix in ['.txt', '.tsv']:
            # ASVspoof5 protocol format (whitespace-separated)
            df = pd.read_csv(manifest_path, sep=r'\s+', header=None,
                           names=['speaker_id', 'flac_file', 'gender', 'attack_label', 'key'])
        else:
            df = pd.read_parquet(manifest_path)  # Default to parquet
        
        if len(df) > args.max_samples:
            df = df.sample(n=args.max_samples, random_state=args.seed)
            logger.info(f"  Sampled {args.max_samples} from {len(df)} samples")
        
        features = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            # Handle different path column names (parquet vs protocol)
            if 'audio_path' in row:
                # Absolute path from our manifests
                audio_path = Path(row['audio_path'])
            elif 'path' in row:
                audio_path = args.data_root / row['path']
            elif 'flac_file' in row:
                audio_path = args.data_root / row['flac_file']
            elif 'file' in row:
                audio_path = args.data_root / row['file']
            else:
                logger.warning(f"No path column found in manifest. Columns: {list(row.index)}")
                continue
            
            if not audio_path.exists():
                # Try alternative path structures
                for alt in ['flac_T', 'flac_D', 'flac_E_eval']:
                    alt_path = args.data_root / alt / audio_path.name
                    if alt_path.exists():
                        audio_path = alt_path
                        break
            
            if not audio_path.exists():
                continue
            
            feat = extract_shortcut_features(audio_path)
            if feat is None:
                continue
            
            features.append(list(feat.values()))
            
            # Handle different label formats (parquet vs protocol)
            if 'y_task' in row:
                # Our manifests use y_task: 0=bonafide, 1=spoof
                label = int(row['y_task'])
            elif 'label' in row:
                label = int(row['label'])
            elif 'key' in row:
                # ASVspoof5 protocol: bonafide=0, spoof=1
                label = 0 if str(row['key']).lower() == 'bonafide' else 1
            elif 'attack_label' in row:
                label = 0 if str(row['attack_label']) == '-' else 1
            else:
                logger.warning(f"No label column found. Columns: {list(row.index)}")
                continue
            
            labels.append(label)
        
        results[split_name] = {
            'X': np.array(features),
            'y': np.array(labels),
            'n_samples': len(labels),
        }
        logger.info(f"  Extracted {len(labels)} samples")
    
    # Train classifier on train set
    logger.info("\nTraining shortcut classifier...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(results['train']['X'])
    y_train = results['train']['y']
    
    clf = LogisticRegression(max_iter=1000, random_state=args.seed, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate on all splits
    print("\n" + "=" * 60)
    print("SHORTCUT CLASSIFIER RESULTS")
    print("=" * 60)
    print("\nFeatures: peak_amplitude, total_energy, duration, leading_silence, trailing_silence")
    print("Model: Logistic Regression (balanced class weights)\n")
    
    eval_results = {}
    
    for split_name in results.keys():
        X = scaler.transform(results[split_name]['X'])
        y = results[split_name]['y']
        
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        eer = compute_eer(y, y_prob)
        
        eval_results[split_name] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'eer': float(eer),
            'n_samples': int(results[split_name]['n_samples']),
        }
        
        print(f"{split_name:>5}: Acc={acc:.1%}, AUC={auc:.3f}, EER={eer:.1%} (n={results[split_name]['n_samples']})")
    
    # Feature importances
    print("\nFeature coefficients (importance):")
    feature_names = ['peak_amp', 'energy', 'duration', 'lead_sil', 'trail_sil']
    coef_dict = {}
    for name, coef in zip(feature_names, clf.coef_[0]):
        print(f"  {name:>10}: {coef:+.3f}")
        coef_dict[name] = float(coef)
    
    # Analysis
    print("\n" + "-" * 60)
    print("ANALYSIS")
    print("-" * 60)
    
    train_acc = eval_results['train']['accuracy']
    dev_acc = eval_results['dev']['accuracy']
    
    if train_acc > 0.6 and dev_acc < 0.55:
        print("✓ Shortcuts are TRAIN-SPECIFIC: high train acc, low dev acc")
        print("  → OOD gap is NOT primarily explained by shortcuts")
    elif train_acc > 0.6 and dev_acc > 0.55:
        print("⚠ Shortcuts generalize somewhat: both train and dev acc elevated")
        print("  → Shortcuts may partially explain OOD gap")
    else:
        print("✓ Shortcuts have LIMITED predictive power overall")
        print("  → Shortcut confounds are minimal")
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            'eval_results': eval_results,
            'feature_coefficients': coef_dict,
            'config': {
                'max_samples': args.max_samples,
                'seed': args.seed,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
