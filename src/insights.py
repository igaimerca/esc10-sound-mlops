"""Per-class acoustic feature statistics used by the notebook and the dashboard."""
import json

import librosa
import numpy as np
import pandas as pd

from .config import DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from .preprocessing import HOP, N_FFT, SR, list_files, load_audio

FEATURES = ["rms_energy", "spectral_centroid_hz", "zero_crossing_rate", "spectral_rolloff_hz", "active_ratio"]


def clip_features(y):
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    return {
        "rms_energy": float(np.sqrt(np.mean(y**2))),
        "spectral_centroid_hz": float(librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP).mean()),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP).mean()),
        "spectral_rolloff_hz": float(librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP).mean()),
        "active_ratio": float((rms > 0.1 * rms.max()).mean()) if rms.max() > 0 else 0.0,
    }


def feature_table():
    rows = []
    for split, root in (("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)):
        paths, labels = list_files(root)
        for p, label in zip(paths, labels):
            rows.append({"file": p.name, "label": label, "split": split, **clip_features(load_audio(p))})
    return pd.DataFrame(rows)


def build_insights(out=DATA_DIR / "insights.json"):
    df = feature_table()
    df.to_csv(DATA_DIR / "audio_features.csv", index=False)
    summary = df.groupby("label")[FEATURES].agg(["mean", "std"])
    payload = {
        "class_counts": df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict(orient="index"),
        "features": {
            f: {
                "mean": summary[(f, "mean")].round(5).to_dict(),
                "std": summary[(f, "std")].round(5).to_dict(),
            }
            for f in FEATURES
        },
    }
    out.write_text(json.dumps(payload))
    return df, payload


if __name__ == "__main__":
    df, _ = build_insights()
    print(df.groupby("label")[FEATURES].mean().round(3))
