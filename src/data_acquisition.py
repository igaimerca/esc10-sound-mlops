"""Download ESC-50 and build the ESC-10 train/val/test folders.

Split by the dataset's predefined folds (clips cut from the same recording share a fold),
so no source recording appears on both sides of a split:
  folds 1-3 -> data/train, fold 4 -> data/val, fold 5 -> data/test
"""
import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, LABELS_PATH, MODEL_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
SPLIT_BY_FOLD = {1: TRAIN_DIR, 2: TRAIN_DIR, 3: TRAIN_DIR, 4: VAL_DIR, 5: TEST_DIR}


def acquire(zip_path=None):
    if zip_path:
        raw = Path(zip_path).read_bytes()
    else:
        print("Downloading ESC-50 (~640 MB)...")
        raw = urllib.request.urlopen(URL).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    meta = pd.read_csv(zf.open("ESC-50-master/meta/esc50.csv"))
    meta = meta[meta.esc10].reset_index(drop=True)

    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
    for row in meta.itertuples():
        dest = SPLIT_BY_FOLD[row.fold] / row.category
        dest.mkdir(parents=True, exist_ok=True)
        (dest / row.filename).write_bytes(zf.read(f"ESC-50-master/audio/{row.filename}"))

    meta[["filename", "category", "fold", "src_file"]].to_csv(DATA_DIR / "manifest.csv", index=False)
    classes = sorted(meta.category.unique())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(classes))
    print(f"{len(meta)} clips, {len(classes)} classes -> {DATA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="Path to an already downloaded ESC-50 master.zip")
    acquire(ap.parse_args().zip)
