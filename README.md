---
title: ESC-10 Sound Classifier
emoji: 🔊
sdk: docker
app_port: 8000
pinned: false
---

# ESC-10 Environmental Sound Classifier: an end-to-end MLOps pipeline

Classify a 5-second audio clip into one of 10 everyday sounds (chainsaw, clock tick, crackling fire, crying baby, dog, helicopter, rain, rooster, sea waves, sneezing). The project covers data acquisition, preprocessing, model training and evaluation, an API, a web dashboard with upload and one-click retraining, Docker deployment and a Locust load test.

| | |
|---|---|
| **Live app** | `<PASTE PUBLIC URL HERE>` |
| **Video demo** | `<PASTE YOUTUBE LINK HERE>` |
| **GitHub** | `<PASTE REPO URL HERE>` |

## Results at a glance

Held-out test fold (80 clips never used for training or model selection):

| Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 86.3% | 0.361 | 0.891 | 0.863 | 0.868 |

Baselines on the same test fold: majority class 10%, MFCC + logistic regression 72.5% accuracy. Details, six optimisation experiments, confusion matrix, ROC curves and error analysis are in [notebook/esc10_audio_classifier.ipynb](notebook/esc10_audio_classifier.ipynb).

## Project structure

```
├── README.md
├── notebook/esc10_audio_classifier.ipynb   # preprocessing, EDA, experiments, evaluation (executed)
├── src/
│   ├── preprocessing.py   # decode, 16 kHz mono, 5 s, log-mel, augmentation
│   ├── model.py           # CNN, optimisers, early stopping, metrics
│   ├── prediction.py      # model loading with hot reload + single-clip prediction
│   ├── retrain.py         # upload storage, retraining job, promotion gate, auto trigger
│   ├── data_acquisition.py, insights.py, db.py, config.py
├── app/                   # FastAPI service + dashboard (app/static/index.html)
├── data/train, val, test  # ESC-10 clips, split by fold (see below)
├── models/esc10_cnn.h5    # trained base model (+ labels.json, base_metrics.json)
├── locustfile.py, scripts/run_load_tests.sh, results/   # flood simulation
├── Dockerfile, docker-compose.yml, nginx.conf
└── tests/
```

## The pipeline

1. **Data acquisition**: `python -m src.data_acquisition` downloads ESC-50 and keeps the 400 ESC-10 clips (40 per class).
2. **Split**: by ESC-50's predefined folds (1-3 train, 4 validation, 5 test). Clips cut from the same source recording always share a fold, so a recording never appears on both sides of a split.
3. **Preprocessing** (identical in training, API and retraining): mono, 16 kHz, fixed 5 s, 64-band log-mel spectrogram, per-clip standardisation (no dataset statistics are fitted, so nothing can leak). Training-only augmentation: time shift, gain, noise, SpecAugment.
4. **Model**: a small custom CNN (86k parameters), L2 + dropout + batch norm, Adam, early stopping on validation loss, learning-rate reduction on plateau. Six experiments compare regularisation, optimiser (Adam, SGD with Nesterov, RMSprop) and learning rate.
5. **Evaluation**: accuracy, loss, macro precision, recall and F1, per-class report, confusion matrix, one-vs-rest ROC.
6. **Retraining**: users upload labelled clips, which are decoded, normalised and stored (files on disk, rows in SQLite). Retraining fine-tunes the **current live model** on the original training folds plus all uploads, then compares it with the live model on the validation fold. It is promoted only if validation F1 is within 0.02 of the old one, so a bad batch of uploads cannot silently replace a good model. The base file `models/esc10_cnn.h5` is never overwritten; new versions go to `models/versions/` and `models/current.json` points at the live one, which all containers reload automatically.
7. **Triggers**: the "Retrain now" button (`POST /api/retrain`) and an automatic trigger when `RETRAIN_THRESHOLD` (default 30) new clips are pending. Only one run can be active at a time.

## Run it locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.data_acquisition        # downloads ESC-50 (~640 MB), builds data/train|val|test
python -m src.insights                # writes data/insights.json for the dashboard
jupyter nbconvert --to notebook --execute --inplace notebook/esc10_audio_classifier.ipynb   # optional: retrains the base model (~15 min on CPU)

uvicorn app.main:app --port 8000      # open http://localhost:8000
pytest
```

The repository already contains the trained model and the data, so the app starts without running the first three commands.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | up-time, model version, predictions served, pending uploads |
| `POST /api/predict` | one audio file, returns label, confidence, top 3 |
| `POST /api/upload` | form field `label` + many `files` (.wav), stored for retraining |
| `POST /api/retrain` | start retraining (409 if one is already running) |
| `GET /api/retrain/status` | latest run and history with old vs new metrics |
| `GET /api/evaluate` | evaluate the live model on the test clips |
| `GET /api/insights` | per-class feature statistics for the charts |

## Docker and deployment

```bash
docker compose up --build --scale api=3   # nginx load balancer on http://localhost:8080
```

**Public deployment (Hugging Face Spaces, free):** create a new Space with the **Docker** SDK, push this repository to it (the YAML block at the top of this README configures it), and open the Space URL. A Space has 16 GB of RAM, which this app needs: the container uses about 0.5 GB when idle and about 1.3 GB while retraining, so Render's 512 MB plans are too small. Files written inside a free Space (uploads, retrained versions) are lost when the Space restarts.

## Flood request simulation (Locust)

`locustfile.py` posts random test clips to `/api/predict` (plus occasional health checks). Setup: 50 concurrent users, spawn rate 10/s, 60 seconds per run, each API container limited to 1 CPU, nginx round-robin in front, load generator on the same 8-CPU machine. Reproduce with `scripts/run_load_tests.sh 50 10 60`.

| Containers | Requests | Failures | Median (ms) | 95th pct (ms) | 99th pct (ms) | Avg (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | 314 | 0 | 8100 | 10000 | 11000 | 7740 | 5.3 |
| 2 | 545 | 0 | 5000 | 11000 | 13000 | 4536 | 9.2 |
| 3 | 684 | 0 | 2300 | 9200 | 11000 | 3650 | 11.5 |
| 4 | 844 | 0 | 1600 | 9400 | 13000 | 2897 | 14.3 |

How to read it:
- Prediction is CPU-bound (decoding, spectrogram, CNN), so one container tops out at about 5 requests/s. At 50 users that means a queue and an 8-second median.
- Adding containers raised throughput 2.7x (5.3 to 14.3 req/s) and cut the median latency 5x (8.1 s to 1.6 s). There were no failed requests at any size.
- The gain is less than linear because Locust and all containers share one machine, and the 95th percentile barely moved because early requests still queue while the users ramp up.
- An earlier run (kept in `results_run1/`) showed a 2-container anomaly with a 30 s 95th percentile. The cause was cold start: each container loaded the model on its first request. The model is now loaded at startup, which fixed it.

## Limitations

- 80 test clips, so one clip is 1.25 points of accuracy; validation (93.8%) is an optimistic estimate because the model was selected on it.
- Closed-set: any sound is forced into one of 10 classes.
- Training clips are Freesound recordings; real microphones will differ.
- Uploaded labels are trusted. A wrong label will degrade the model, and the validation gate is the only protection.

## Data and references

ESC-50: Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia (CC BY-NC licence). SpecAugment: Park et al. (2019).
