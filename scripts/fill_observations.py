import nbformat as nbf
p = "notebook/esc10_audio_classifier.ipynb"
nb = nbf.read(p, 4)

OBS = {
"TAG:EDA_OBS": """**What the three features tell us**

1. **Active ratio separates continuous from impulsive sounds.** Rain (1.00), helicopter (0.95), sea waves (0.96) and chainsaw (0.94) are active for almost the whole clip. Sneezing (0.13), dog (0.32) and rooster (0.39) are short events padded with silence. A model can use this temporal shape alone to split the 10 classes into two groups, so the main difficulty lies inside each group.
2. **Spectral centroid separates bright from dull sounds.** Rain is the brightest (about 2.7 kHz), then chainsaw (2.4 kHz) and crying baby (2.3 kHz). Dog, rooster, helicopter and sneezing all sit between 1.1 and 1.2 kHz. Those four low-centroid classes overlap, so confusions (e.g. sneezing and dog) are most likely there.
3. **RMS energy varies by a factor of 10 across classes** (clock tick 0.016, helicopter 0.160), but it depends on how the clip was recorded, not only on the sound. The per-clip standardisation in section 3 removes this cue on purpose, which makes the model robust to microphone gain.

Also, features such as centroid and zero-crossing rate are strongly correlated (heatmap below), so they carry overlapping information about "brightness".""",

"TAG:EXP_OBS": """**My observations**

* **Regularisation was the biggest single gain.** With no regularisation (e1) the model memorises the training set (train accuracy 100%, validation 88.8%, validation loss 0.32). Adding L2, dropout and SpecAugment (e2) lifts validation accuracy to 93.8% and halves the gap between train and validation.
* **Optimiser matters less than regularisation.** Adam (e2, F1 0.937) and SGD with Nesterov momentum (e3, F1 0.927) end close together; SGD stopped earlier with lower training accuracy (94.2%), meaning it is under-fitted rather than over-fitted. RMSprop (e4, F1 0.899) was the weakest.
* **A lower learning rate (e5, 3e-4) did not help** (validation loss 0.35, higher than e2's 0.19): the model was still improving slowly when early stopping ended it.
* **Stronger regularisation (e6) gave the best validation F1 (0.938), but only by 0.001 over e2.** With 160 validation clips, one clip is 0.6 points, so e2 and e6 are statistically tied. I still pick e6 by the rule I fixed in advance (highest validation F1), and I report that honestly rather than choosing after seeing the test fold.""",

"TAG:FINAL_OBS": """**My observations on the test fold**

* **Headline: 86.3% accuracy, macro F1 0.868, loss 0.361** (69 of 80 clips correct). The MFCC + logistic regression baseline reaches 72.5% accuracy and the majority baseline 10%, so the CNN is a clear improvement (+13.8 points).
* **Validation 93.8% vs test 86.3%.** The 7.5 point drop is expected: I selected the model on validation, which makes that score slightly optimistic, and the test fold has only 80 clips (about +/- 7.5 points at 95% confidence). I treat 86% as the honest estimate.
* **Easy classes:** crying baby and rooster are perfect (F1 1.00), and rain and sea waves are at 0.93.
* **Hard classes:** helicopter has the lowest precision (0.58) because it absorbs errors from other classes (2 chainsaws, 1 sea wave and 1 dog were labelled helicopter). Both chainsaw and helicopter are steady, low-pitched engine-like noises, which fits the feature analysis. Crackling fire (F1 0.75) is confused with clock tick and helicopter, and sneezing is mistaken for dog twice, which also matches the overlap of short, low-centroid sounds found in section 4.
* **Next steps if more accuracy is needed:** more labelled clips per class (the retraining pipeline in this project exists for exactly that), or a pre-trained audio embedding such as YAMNet as the starting point.""",
}
for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.strip() in OBS:
        cell.source = OBS[cell.source.strip()]
nbf.write(nb, p)

b = open("scripts/build_notebook.py").read()
for tag in OBS:
    assert tag in b
open("notebook/.observations_filled", "w").write("ok")
