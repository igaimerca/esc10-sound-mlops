import json
import shutil
import threading
import traceback
import uuid
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf

from . import db as dbm
from .config import (
    BASE_MODEL,
    PROMOTION_TOLERANCE,
    RETRAIN_THRESHOLD,
    TEST_DIR,
    TRAIN_DIR,
    UPLOAD_DIR,
    VAL_DIR,
    VERSIONS_DIR,
)
from .preprocessing import SR, build_features, list_files, load_audio
from .prediction import read_labels, read_pointer, write_pointer

STALE_AFTER = timedelta(hours=1)


def save_upload(data, filename, label):
    """Decode, normalise to 16 kHz mono 5 s and store an uploaded clip; returns the DB id."""
    import io

    wav = load_audio(io.BytesIO(data))
    dest_dir = UPLOAD_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}.wav"
    sf.write(dest, wav, SR, subtype="PCM_16")
    try:
        with dbm.db() as conn:
            cur = conn.execute(
                "INSERT INTO uploads (filename, stored_path, label) VALUES (?, ?, ?)",
                (filename[:200], str(dest), label),
            )
            return cur.lastrowid
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def pending_count():
    with dbm.db() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads WHERE used_in_version IS NULL").fetchone()[0]


def total_uploads():
    with dbm.db() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def _claim_run(trigger):
    """Atomically register a run and claim the pending uploads; None if one is already running."""
    conn = dbm.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cutoff = (datetime.utcnow() - STALE_AFTER).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE uploads SET used_in_version=NULL WHERE used_in_version IN "
            "(SELECT version FROM retrain_runs WHERE status='running' AND started_at < ?)",
            (cutoff,),
        )
        conn.execute(
            "UPDATE retrain_runs SET status='failed', finished_at=datetime('now'), "
            "message='stale run' WHERE status='running' AND started_at < ?",
            (cutoff,),
        )
        if conn.execute("SELECT 1 FROM retrain_runs WHERE status='running'").fetchone():
            conn.execute("ROLLBACK")
            return None
        version = (conn.execute("SELECT MAX(version) FROM retrain_runs").fetchone()[0] or 0) + 1
        version = max(version, read_pointer()["version"] + 1)
        n_new = conn.execute(
            "UPDATE uploads SET used_in_version=? WHERE used_in_version IS NULL", (version,)
        ).rowcount
        conn.execute(
            "INSERT INTO retrain_runs (version, status, trigger, n_new_files) VALUES (?, 'running', ?, ?)",
            (version, trigger, n_new),
        )
        conn.execute("COMMIT")
        return version, n_new
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _finish(version, status, message, metrics=None):
    with dbm.db() as conn:
        conn.execute(
            "UPDATE retrain_runs SET status=?, finished_at=datetime('now'), message=?, metrics=? WHERE version=?",
            (status, message, json.dumps(metrics) if metrics else None, version),
        )


def _release_claim(version):
    with dbm.db() as conn:
        conn.execute("UPDATE uploads SET used_in_version=NULL WHERE used_in_version=?", (version,))


def _run(version):
    from .model import compile_model, evaluate, fit, load_model

    new_path = VERSIONS_DIR / f"v{version}.h5"
    try:
        classes = read_labels()
        pointer = read_pointer()
        base_path = pointer["path"] if pointer["path"] else str(BASE_MODEL)

        train_p, train_l = list_files(TRAIN_DIR)
        up_p, up_l = list_files(UPLOAD_DIR)
        Xtr, ytr = build_features(train_p + up_p, train_l + up_l, classes, augment_copies=2, seed=version)
        Xva, yva = build_features(*list_files(VAL_DIR), classes)
        Xte, yte = build_features(*list_files(TEST_DIR), classes)

        old = load_model(base_path)
        old_val = evaluate(compile_model(old), Xva, yva, classes)
        old_test = evaluate(old, Xte, yte, classes)

        model = load_model(base_path)
        compile_model(model, "adam", 1e-4)
        history = fit(model, Xtr, ytr, Xva, yva, epochs=30, patience=6)
        new_val = evaluate(model, Xva, yva, classes)
        new_test = evaluate(model, Xte, yte, classes)

        promote = new_val["f1_macro"] >= old_val["f1_macro"] - PROMOTION_TOLERANCE
        metrics = {
            "n_train_clips": int(len(Xtr)),
            "n_uploaded_files": len(up_p),
            "epochs_run": len(history.history["loss"]),
            "previous": {"val_f1": old_val["f1_macro"], "test": _summary(old_test)},
            "new": {"val_f1": new_val["f1_macro"], "test": _summary(new_test)},
        }
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(new_path)
        if promote:
            write_pointer(version, new_path, metrics["new"]["test"])
            _finish(version, "promoted", "New model is live", metrics)
        else:
            _finish(
                version,
                "rejected",
                f"Validation F1 {new_val['f1_macro']:.3f} fell below the live model's {old_val['f1_macro']:.3f}; kept the live model",
                metrics,
            )
    except Exception as exc:
        traceback.print_exc()
        new_path.unlink(missing_ok=True)
        _release_claim(version)
        _finish(version, "failed", f"{type(exc).__name__}: {exc}")


def _summary(m):
    return {k: m[k] for k in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro")}


def start_retrain(trigger="manual", background=True):
    claimed = _claim_run(trigger)
    if claimed is None:
        return None
    version, _ = claimed
    if background:
        threading.Thread(target=_run, args=(version,), daemon=True).start()
    else:
        _run(version)
    return version


def maybe_auto_retrain():
    if pending_count() >= RETRAIN_THRESHOLD:
        return start_retrain("auto")
    return None


def latest_run():
    with dbm.db() as conn:
        row = conn.execute("SELECT * FROM retrain_runs ORDER BY version DESC LIMIT 1").fetchone()
    return _row(row)


def run_history(limit=10):
    with dbm.db() as conn:
        rows = conn.execute("SELECT * FROM retrain_runs ORDER BY version DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def _row(row):
    if row is None:
        return None
    d = dict(row)
    d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
    return d
