import numpy as np
import librosa

SR = 16000
SECONDS = 5
SAMPLES = SR * SECONDS
N_MELS = 64
N_FFT = 1024
HOP = 512
N_FRAMES = 1 + SAMPLES // HOP
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)


def fix_length(y):
    if len(y) >= SAMPLES:
        return y[:SAMPLES]
    return np.pad(y, (0, SAMPLES - len(y)))


def load_audio(src):
    """Load a path or file-like object as mono 16 kHz audio of exactly 5 seconds."""
    try:
        y, _ = librosa.load(src, sr=SR, mono=True, duration=SECONDS)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc
    if y.size < SR // 10:
        raise ValueError("Audio is shorter than 0.1 seconds")
    return fix_length(y).astype(np.float32)


def log_mel(y):
    """Log-mel spectrogram, standardised per clip so no statistics are fitted on data."""
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=SR // 2
    )
    db = librosa.power_to_db(m, ref=1.0, top_db=80.0)
    db = (db - db.mean()) / (db.std() + 1e-6)
    return db[..., None].astype(np.float32)


def augment_waveform(y, rng):
    shift = int(rng.integers(-SR, SR))
    y = np.roll(y, shift)
    y = y * rng.uniform(0.6, 1.4)
    snr_db = rng.uniform(20, 35)
    noise_power = (np.mean(y**2) + 1e-10) / (10 ** (snr_db / 10))
    y = y + rng.normal(0, np.sqrt(noise_power), y.shape)
    return y.astype(np.float32)


def list_files(root):
    """Return (paths, labels) for root/<class>/*.wav."""
    from pathlib import Path

    root = Path(root)
    paths, labels = [], []
    if not root.exists():
        return paths, labels
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.glob("*.wav")):
            paths.append(f)
            labels.append(cls_dir.name)
    return paths, labels


def build_features(paths, labels, class_names, augment_copies=0, seed=42):
    """Feature matrix for the given files; augment_copies adds randomly augmented copies."""
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(class_names)}
    X, y = [], []
    for path, label in zip(paths, labels):
        wav = load_audio(path)
        X.append(log_mel(wav))
        y.append(idx[label])
        for _ in range(augment_copies):
            X.append(log_mel(augment_waveform(wav, rng)))
            y.append(idx[label])
    if not X:
        return np.zeros((0, *INPUT_SHAPE), np.float32), np.zeros((0,), np.int64)
    return np.stack(X), np.array(y, dtype=np.int64)
