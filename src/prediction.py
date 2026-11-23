import io
import json
import os
import tempfile
import threading

import numpy as np

from .config import BASE_MODEL, CURRENT_POINTER, LABELS_PATH, MODEL_DIR
from .preprocessing import load_audio, log_mel


def read_labels():
    return json.loads(LABELS_PATH.read_text())


def read_pointer():
    if CURRENT_POINTER.exists():
        pointer = json.loads(CURRENT_POINTER.read_text())
        pointer["path"] = str(MODEL_DIR / pointer["path"])
        return pointer
    return {"version": 0, "path": str(BASE_MODEL)}


def write_pointer(version, path, metrics):
    CURRENT_POINTER.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CURRENT_POINTER.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump({"version": version, "path": os.path.relpath(path, MODEL_DIR), "metrics": metrics}, f)
    os.replace(tmp, CURRENT_POINTER)


class Predictor:
    """Serves the model named by models/current.json and reloads it when the pointer changes,
    so every container sharing the models folder picks up a retrained model."""

    def __init__(self):
        self._lock = threading.Lock()
        self._mtime = None
        self._model = None
        self.version = None
        self.classes = None

    def _ensure_loaded(self):
        mtime = CURRENT_POINTER.stat().st_mtime_ns if CURRENT_POINTER.exists() else 0
        if self._model is not None and mtime == self._mtime:
            return
        with self._lock:
            if self._model is not None and mtime == self._mtime:
                return
            from .model import load_model

            pointer = read_pointer()
            self._model = load_model(pointer["path"])
            self.classes = read_labels()
            self.version = pointer["version"]
            self._mtime = mtime

    def warm_up(self):
        """Load the model and run one dummy inference so the first real request is not slow."""
        from .preprocessing import INPUT_SHAPE

        self.predict_features(np.zeros(INPUT_SHAPE, np.float32))

    def predict_features(self, features):
        self._ensure_loaded()
        model, classes, version = self._model, self.classes, self.version
        return model.predict(features[None], verbose=0)[0], classes, version

    def predict(self, source, top_k=3):
        """source: path or file-like holding audio. Returns the predicted class and top-k scores."""
        probs, classes, version = self.predict_features(log_mel(load_audio(source)))
        order = np.argsort(probs)[::-1]
        return {
            "label": classes[order[0]],
            "confidence": float(probs[order[0]]),
            "top": [{"label": classes[i], "confidence": float(probs[i])} for i in order[:top_k]],
            "model_version": version,
        }

    def predict_bytes(self, data, top_k=3):
        return self.predict(io.BytesIO(data), top_k)
