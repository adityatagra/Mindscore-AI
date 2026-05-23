"""
Audio dataset loader and advanced feature extractor.

Extracts a rich 34-dimensional feature vector per audio clip:
  - 13 MFCC means + 13 MFCC stds  (spectral envelope)
  - 1 RMS energy mean              (loudness)
  - 1 zero crossing rate mean      (noisiness / voicing)
  - 1 spectral centroid mean       (brightness)
  - 1 spectral rolloff mean        (frequency distribution)
  - 4 spectral contrast bands      (dynamic range)

These capture vocal stress markers: pitch variation, speech rate,
energy fluctuation, and spectral tilt that correlate with emotional state.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

FEATURE_DIM = 34


def extract_audio_features(file_path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract a comprehensive 34-dimensional feature vector from an audio file.

    Requires at least config.AUDIO_CONFIG['min_duration_sec'] seconds of audio.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for audio. Run: pip install librosa soundfile")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(file_path, sr=sr)

    duration = len(y) / sr
    min_required = config.AUDIO_CONFIG.get("min_duration_sec", 10)
    max_allowed = config.AUDIO_CONFIG.get("max_duration_sec")
    if duration < min_required:
        raise ValueError(
            f"Audio too short: {duration:.1f}s (minimum {min_required}s required). "
            "Please provide a longer recording for accurate analysis."
        )
    if max_allowed and duration > max_allowed:
        raise ValueError(
            f"Audio too long: {duration:.1f}s (maximum {max_allowed}s allowed)."
        )

    features = []

    # MFCC (26 features: 13 means + 13 stds)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # RMS energy (1 feature)
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))

    # Zero crossing rate (1 feature)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))

    # Spectral centroid (1 feature)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(centroid)))

    # Spectral rolloff (1 feature)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(rolloff)))

    # Spectral contrast - 4 bands (4 features)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
    features.extend(np.mean(contrast, axis=1))

    return np.array(features, dtype=np.float64)


def load_and_preprocess(cfg: dict = None) -> pd.DataFrame:
    """Load audio files, extract features, and save as CSV."""
    cfg = cfg or config.AUDIO_CONFIG
    processed_path = cfg["processed_csv"]

    if os.path.exists(processed_path):
        print(f"[AudioLoader] Loading cached features from {processed_path}")
        return pd.read_csv(processed_path)

    meta_path = cfg["metadata_csv"]
    audio_dir = cfg["raw_dir"]

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Audio metadata CSV not found at {meta_path}.\n"
            f"Provide a CSV with columns: '{cfg['file_column']}', '{cfg['label_column']}'."
        )

    meta = pd.read_csv(meta_path)
    file_col = cfg["file_column"]
    label_col = cfg["label_column"]

    records = []
    for _, row in meta.iterrows():
        fpath = os.path.join(audio_dir, row[file_col])
        if not os.path.exists(fpath):
            continue
        try:
            feats = extract_audio_features(fpath, sr=cfg["sample_rate"], n_mfcc=cfg["n_mfcc"])
            record = {f"feat_{i}": feats[i] for i in range(len(feats))}
            record[label_col] = row[label_col]
            records.append(record)
        except Exception as e:
            print(f"[AudioLoader] Skipping {fpath}: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[AudioLoader] Saved features to {processed_path} ({len(df)} samples)")
    return df


if __name__ == "__main__":
    if config.AUDIO_CONFIG["enabled"]:
        print(f"Audio feature dimensionality: {FEATURE_DIM}")
        print("Audio module is enabled. Provide data or use generate_dummy_data.py")
    else:
        print("Audio module disabled. Set AUDIO_CONFIG['enabled'] = True in config.py")
