import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src import retrain
from src.config import (
    DATA_DIR,
    MAX_FILES_PER_UPLOAD,
    MAX_UPLOAD_BYTES,
    RETRAIN_THRESHOLD,
    TEST_DIR,
)
from src.db import init_db
from src.preprocessing import build_features, list_files
from src.prediction import Predictor, read_labels, read_pointer

STATIC = Path(__file__).parent / "static"
STARTED = time.time()

app = FastAPI(title="ESC-10 Sound Classifier", version="1.0")
predictor = Predictor()
stats = {"predictions": 0}


@app.on_event("startup")
def startup():
    init_db()
    predictor.warm_up()


async def _read_limited(upload: UploadFile):
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename} is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    return data


@app.get("/api/health")
def health():
    pointer = read_pointer()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED, 1),
        "started_at": STARTED,
        "model_version": pointer["version"],
        "predictions_served": stats["predictions"],
        "pending_uploads": retrain.pending_count(),
        "retrain_threshold": RETRAIN_THRESHOLD,
    }


@app.get("/api/classes")
def classes():
    return read_labels()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await _read_limited(file)
    try:
        result = await run_in_threadpool(predictor.predict_bytes, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    stats["predictions"] += 1
    return result


@app.post("/api/upload")
async def upload(label: str = Form(...), files: list[UploadFile] = File(...)):
    if label not in read_labels():
        raise HTTPException(422, f"label must be one of {read_labels()}")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(413, f"at most {MAX_FILES_PER_UPLOAD} files per upload")
    saved, rejected = 0, []
    for f in files:
        try:
            await run_in_threadpool(retrain.save_upload, await _read_limited(f), f.filename or "clip.wav", label)
            saved += 1
        except HTTPException as exc:
            rejected.append({"file": f.filename, "reason": exc.detail})
        except ValueError as exc:
            rejected.append({"file": f.filename, "reason": str(exc)})
    auto = retrain.maybe_auto_retrain() if saved else None
    return {
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": retrain.pending_count(),
        "auto_retrain_started": auto is not None,
        "retrain_version": auto,
    }


@app.post("/api/retrain")
def trigger_retrain():
    version = retrain.start_retrain("manual")
    if version is None:
        raise HTTPException(409, "A retraining run is already in progress")
    return {"started": True, "version": version}


@app.get("/api/retrain/status")
def retrain_status():
    return {"latest": retrain.latest_run(), "history": retrain.run_history(), "pending_uploads": retrain.pending_count()}


@app.get("/api/evaluate")
def evaluate_live():
    from src.model import evaluate, load_model

    classes_ = read_labels()
    pointer = read_pointer()
    X, y = build_features(*list_files(TEST_DIR), classes_)
    m = evaluate(load_model(pointer["path"]), X, y, classes_)
    return {"model_version": pointer["version"], "n_test_clips": int(len(y)), **m}


@app.get("/api/insights")
def insights():
    path = DATA_DIR / "insights.json"
    if not path.exists():
        raise HTTPException(404, "insights.json missing; run `python -m src.insights`")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
