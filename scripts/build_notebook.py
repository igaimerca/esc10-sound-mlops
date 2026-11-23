import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s, tag=None: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Environmental Sound Classification (ESC-10): an end-to-end ML pipeline

**Author:** Aime Igirimpuhwe

**Task.** Classify a 5-second audio clip into one of 10 everyday sound classes (dog, rooster, rain, sea waves, crackling fire, crying baby, sneezing, helicopter, chainsaw, clock tick). The model is deployed behind a FastAPI service with a web dashboard where a user can predict one clip, upload new labelled clips in bulk, and trigger retraining.

**Why this matters.** Sound classification is non-tabular data, and it is the building block for noise monitoring, smart-home alerts and wildlife monitoring. This notebook covers data acquisition, preprocessing, EDA, model training with several optimisation techniques, evaluation and error analysis. The same modules (`src/preprocessing.py`, `src/model.py`, `src/prediction.py`) are used by the deployed service, so what is evaluated here is what runs in production.

**Dataset.** ESC-50 (Piczak, 2015), restricted to its 10-class ESC-10 subset: 400 clips, 40 per class, from Freesound recordings.""")

md("## 1. Setup")
code("""import sys, json, random, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import librosa, librosa.display
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from src.config import *
from src.preprocessing import *
from src.model import *
from src import insights

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
sns.set_theme(style="whitegrid")
CLASSES = json.loads(LABELS_PATH.read_text())
print("TensorFlow", tf.__version__, "| classes:", CLASSES)""")

md("""## 2. Data acquisition and split

`src/data_acquisition.py` downloads ESC-50 and keeps the ESC-10 clips. ESC-50 ships with five predefined folds, and clips taken from the same original Freesound recording are always placed in the same fold. I split **by fold**, not by random clip:

| Split | Folds | Folder | Use |
|---|---|---|---|
| train | 1, 2, 3 | `data/train` | fit weights |
| val | 4 | `data/val` | early stopping, model selection, retraining promotion gate |
| test | 5 | `data/test` | reported once for the final model |

A random clip-level split would put near-duplicate pieces of one recording on both sides and inflate the test score. Splitting by fold avoids that. Nothing is fitted on the data during preprocessing (see section 3), so there is no leakage through normalisation statistics either.""")
code("""counts = {}
for name, root in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
    _, labels = list_files(root)
    counts[name] = pd.Series(labels).value_counts()
split_table = pd.DataFrame(counts).loc[CLASSES]
split_table.loc["TOTAL"] = split_table.sum()
split_table""")

md("""## 3. Preprocessing

Each clip goes through exactly the same steps in training, in the API and in retraining (`src/preprocessing.py`):

1. Decode, convert to **mono**, resample to **16 kHz**.
2. Fix the length to **5 s** (crop longer clips, zero-pad shorter ones), which is ESC-50's native length.
3. **Log-mel spectrogram**: 64 mel bands, FFT 1024, hop 512, converted to dB with an 80 dB floor, giving a 64x157 "image".
4. **Per-clip standardisation** (subtract the clip's own mean, divide by its own std). No dataset statistics are fitted, so nothing can leak from validation or test, and the model becomes robust to microphone gain.

**Augmentation (training data only).** Waveform level: random circular time shift (up to 1 s), gain 0.6-1.4x, additive noise at 20-35 dB SNR. This adds 3 extra copies per training clip. Spectrogram level: SpecAugment, i.e. a random frequency mask and a random time mask on every batch.""")
code("""y = load_audio(list_files(TRAIN_DIR)[0][0])
print("waveform:", y.shape, "->", "log-mel:", log_mel(y).shape)

fig, axes = plt.subplots(2, 5, figsize=(20, 6))
for ax, cls in zip(axes.ravel(), CLASSES):
    p = list_files(TRAIN_DIR)[0][list_files(TRAIN_DIR)[1].index(cls)]
    S = log_mel(load_audio(p))[..., 0]
    librosa.display.specshow(S, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title(cls)
plt.suptitle("Log-mel spectrogram of one training clip per class (after preprocessing)", y=1.02)
plt.tight_layout(); plt.show()""")

md("## 4. Exploratory analysis: what does each sound look like?")
code("""fig, axes = plt.subplots(2, 5, figsize=(20, 5), sharey=True)
for ax, cls in zip(axes.ravel(), CLASSES):
    p = list_files(TRAIN_DIR)[0][list_files(TRAIN_DIR)[1].index(cls)]
    w = load_audio(p)
    ax.plot(np.arange(len(w)) / SR, w, lw=0.4)
    ax.set_title(cls)
plt.suptitle("Waveform of one training clip per class", y=1.02); plt.tight_layout(); plt.show()""")
code("""df, _ = insights.build_insights()
feats = insights.FEATURES
df.groupby("label")[feats].mean().round(3)""")
code("""fig, axes = plt.subplots(1, 3, figsize=(21, 5))
for ax, f in zip(axes, ["active_ratio", "spectral_centroid_hz", "rms_energy"]):
    order = df.groupby("label")[f].median().sort_values().index
    sns.boxplot(data=df, y="label", x=f, order=order, ax=ax, color="#4c78a8")
    ax.set_title(f); ax.set_ylabel("")
if True:
    axes[2].set_xscale("log")
plt.tight_layout(); plt.show()""")
md("TAG:EDA_OBS")
code("""corr = df[feats].corr()
plt.figure(figsize=(6, 5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0); plt.title("Feature correlation"); plt.show()""")

md("""## 5. Baselines

A model is only good relative to something simple. Two baselines on the test fold:

* **Majority class**: always predict one class, so accuracy is 10% with balanced classes.
* **Logistic regression on summary features**: mean and std of 20 MFCCs per clip, scaled with a scaler **fitted on the training folds only**.""")
code("""def mfcc_stats(paths):
    out = []
    for p in paths:
        m = librosa.feature.mfcc(y=load_audio(p), sr=SR, n_mfcc=20)
        out.append(np.r_[m.mean(1), m.std(1)])
    return np.array(out)

idx = {c: i for i, c in enumerate(CLASSES)}
tr_p, tr_l = list_files(TRAIN_DIR); va_p, va_l = list_files(VAL_DIR); te_p, te_l = list_files(TEST_DIR)
Ftr, Fva, Fte = mfcc_stats(tr_p), mfcc_stats(va_p), mfcc_stats(te_p)
ytr_b, yva_b, yte_b = [np.array([idx[l] for l in L]) for L in (tr_l, va_l, te_l)]
scaler = StandardScaler().fit(Ftr)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Ftr), ytr_b)
pred = lr.predict(scaler.transform(Fte))
from sklearn.metrics import accuracy_score, f1_score
baseline = {"majority class": 1 / len(CLASSES),
            "MFCC + logistic regression (accuracy)": accuracy_score(yte_b, pred),
            "MFCC + logistic regression (macro F1)": f1_score(yte_b, pred, average="macro")}
pd.Series(baseline, name="test").round(3).to_frame()""")

md("""## 6. Building the training sets

Features are computed once. The training folds are expanded with 3 augmented copies per clip, while validation and test stay untouched.""")
code("""Xtr, ytr = build_features(tr_p, tr_l, CLASSES, augment_copies=3, seed=SEED)
Xva, yva = build_features(va_p, va_l, CLASSES)
Xte, yte = build_features(te_p, te_l, CLASSES)
print("train", Xtr.shape, "| val", Xva.shape, "| test", Xte.shape)""")

md("""## 7. Model: a custom CNN on log-mel spectrograms

Four Conv-BatchNorm-ReLU-MaxPool blocks (16-32-64-96 filters), global average pooling, dropout, a 64-unit dense layer and a 10-way softmax. It is deliberately small (86k parameters) so that it trains on a CPU and fits a free cloud tier. This model is also the **pre-trained starting point** that the retraining pipeline fine-tunes.

**Optimisation techniques used:** L2 weight decay, dropout, batch normalisation, data augmentation (waveform and SpecAugment), early stopping on validation loss with best-weight restore, learning-rate reduction on plateau, and a comparison of three optimisers (Adam, SGD with Nesterov momentum, RMSprop) plus learning-rate and regularisation-strength tuning.""")
code("""build_model(len(CLASSES)).summary()""")

md("""## 8. Experiments

Each run changes one thing from a reference configuration. Models are compared on the **validation fold**; the test fold is not touched until section 9.""")
code("""EXPERIMENTS = {
    "e1 adam, no regularisation":        dict(l2=0,    dropout=0.0, spec=False, opt="adam",    lr=1e-3),
    "e2 adam + L2 + dropout + SpecAug":  dict(l2=1e-4, dropout=0.4, spec=True,  opt="adam",    lr=1e-3),
    "e3 sgd (nesterov) + reg":           dict(l2=1e-4, dropout=0.4, spec=True,  opt="sgd",     lr=2e-2),
    "e4 rmsprop + reg":                  dict(l2=1e-4, dropout=0.4, spec=True,  opt="rmsprop", lr=1e-3),
    "e5 adam, lr 3e-4":                  dict(l2=1e-4, dropout=0.4, spec=True,  opt="adam",    lr=3e-4),
    "e6 adam, stronger reg (l2 1e-3, drop 0.5)": dict(l2=1e-3, dropout=0.5, spec=True, opt="adam", lr=1e-3),
}
runs, models, histories = [], {}, {}
for name, cfg in EXPERIMENTS.items():
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
    m = compile_model(build_model(len(CLASSES), cfg["l2"], cfg["dropout"]), cfg["opt"], cfg["lr"])
    h = fit(m, Xtr, ytr, Xva, yva, epochs=80, patience=12, spec_augment=cfg["spec"])
    ev_tr, ev_va = evaluate(m, Xtr, ytr, CLASSES), evaluate(m, Xva, yva, CLASSES)
    runs.append({"experiment": name, "epochs run": len(h.history["loss"]),
                 "train acc": ev_tr["accuracy"], "val acc": ev_va["accuracy"], "val loss": ev_va["loss"],
                 "val precision": ev_va["precision_macro"], "val recall": ev_va["recall_macro"], "val F1": ev_va["f1_macro"]})
    models[name], histories[name] = m, h.history
    print(name, "-> val acc %.3f, val F1 %.3f" % (ev_va["accuracy"], ev_va["f1_macro"]))
results = pd.DataFrame(runs).set_index("experiment")
results.round(3)""")
code("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for name, h in histories.items():
    axes[0].plot(h["val_loss"], label=name); axes[1].plot(h["val_accuracy"], label=name)
axes[0].set_title("Validation loss"); axes[1].set_title("Validation accuracy")
for a in axes: a.set_xlabel("epoch")
axes[0].legend(fontsize=7); plt.tight_layout(); plt.show()""")
md("TAG:EXP_OBS")

md("""## 9. Final model on the held-out test fold

The experiment with the best validation macro-F1 is selected, and its test-fold results are reported **once**. Reported metrics: accuracy, loss, macro precision, macro recall, macro F1, plus a per-class report, confusion matrix and ROC curves.""")
code("""best_name = results["val F1"].idxmax()
best = models[best_name]
print("Selected:", best_name)
final = evaluate(best, Xte, yte, CLASSES)
pd.Series({k: final[k] for k in ["accuracy", "loss", "precision_macro", "recall_macro", "f1_macro"]}, name="test").round(4).to_frame()""")
code("""probs = best.predict(Xte, verbose=0); pred = probs.argmax(1)
print(classification_report(yte, pred, target_names=CLASSES, digits=3))""")
code("""from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
h = histories[best_name]
axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="val"); axes[0].set_title("Loss"); axes[0].legend()
axes[0].plot([], [])
cm = np.array(final["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], cbar=False)
axes[1].set_title("Confusion matrix (test)"); axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
Y = label_binarize(yte, classes=range(len(CLASSES)))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i]); axes[2].plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=0.8); axes[2].set_title("One-vs-rest ROC (test)"); axes[2].legend(fontsize=7)
plt.tight_layout(); plt.show()""")
code("""off = [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
pd.DataFrame(sorted(off, key=lambda t: -t[2]), columns=["true", "predicted as", "count"]).head(8)""")
md("TAG:FINAL_OBS")

md("""## 10. Save the model and a prediction function

The selected model is saved as `models/esc10_cnn.h5` (the base model, never overwritten: retraining writes new versions under `models/versions/` and moves a pointer, `models/current.json`). The label list is saved next to it. `src/prediction.py` loads them and predicts from any audio file.""")
code("""MODEL_DIR.mkdir(exist_ok=True)
best.save(BASE_MODEL)
LABELS_PATH.write_text(json.dumps(CLASSES))
json.dump({"selected": best_name, "test": {k: final[k] for k in ["accuracy","loss","precision_macro","recall_macro","f1_macro"]},
           "per_class": final["per_class"], "confusion_matrix": final["confusion_matrix"]}, open(MODEL_DIR / "base_metrics.json", "w"), indent=1)
results.round(4).to_csv(MODEL_DIR / "experiments.csv")
(CURRENT_POINTER).unlink(missing_ok=True)
print("saved", BASE_MODEL, f"({BASE_MODEL.stat().st_size/1e6:.2f} MB)")""")
code("""from src.prediction import Predictor
predictor = Predictor()
sample = te_p[7]
print("true:", te_l[7])
predictor.predict(sample)""")
code("""ok = sum(predictor.predict(p)["label"] == l for p, l in zip(te_p, te_l))
print(f"Prediction function reproduces test accuracy: {ok}/{len(te_p)} = {ok/len(te_p):.4f} (notebook: {final['accuracy']:.4f})")""")

md("""## 11. Limitations

* **Small test set.** 80 clips (8 per class) means one clip is 1.25 points of accuracy; treat differences of a point or two as noise.
* **Closed set.** The model always outputs one of 10 classes. A dishwasher or a car horn will be labelled as one of them, sometimes with high confidence.
* **Silence padding.** ESC-50 pads short events with digital silence, and the model may partly use the silence as a cue (see the active-ratio analysis). Uploaded clips are padded the same way so the behaviour is consistent, but this can differ from real continuous recordings.
* **Recording conditions.** All training clips are Freesound recordings; microphones and rooms in the wild will differ.""")

md("## 12. References\n\n* Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *Proceedings of the 23rd ACM International Conference on Multimedia*.\n* Park, D. S. et al. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech*.")

nb.cells = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
Path("notebook/esc10_audio_classifier.ipynb").write_text(nbf.writes(nb))
