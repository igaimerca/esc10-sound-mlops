import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"

BASE_MODEL = MODEL_DIR / "esc10_cnn.h5"
LABELS_PATH = MODEL_DIR / "labels.json"
CURRENT_POINTER = MODEL_DIR / "current.json"
VERSIONS_DIR = MODEL_DIR / "versions"

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", 30))
PROMOTION_TOLERANCE = 0.02
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 100
