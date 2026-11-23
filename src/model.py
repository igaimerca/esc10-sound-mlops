import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .preprocessing import INPUT_SHAPE, N_FRAMES, N_MELS


def build_model(n_classes, l2=1e-4, dropout=0.4):
    reg = tf.keras.regularizers.l2(l2)
    inp = tf.keras.Input(INPUT_SHAPE)
    x = inp
    for filters in (16, 32, 64, 96):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="esc10_cnn")


def make_optimizer(name, lr):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(lr)
    raise ValueError(f"Unknown optimizer {name}")


def compile_model(model, optimizer="adam", lr=1e-3):
    model.compile(
        optimizer=make_optimizer(optimizer, lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _spec_augment(x, y):
    f = tf.random.uniform([], 0, 9, tf.int32)
    f0 = tf.random.uniform([], 0, N_MELS - f, tf.int32)
    t = tf.random.uniform([], 0, 25, tf.int32)
    t0 = tf.random.uniform([], 0, N_FRAMES - t, tf.int32)
    mel_idx = tf.range(N_MELS)[:, None, None]
    time_idx = tf.range(N_FRAMES)[None, :, None]
    mask = ((mel_idx >= f0) & (mel_idx < f0 + f)) | ((time_idx >= t0) & (time_idx < t0 + t))
    return tf.where(mask, tf.zeros_like(x), x), y


def make_dataset(X, y, batch_size=32, training=False, spec_augment=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        if spec_augment:
            ds = ds.map(_spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def fit(model, Xtr, ytr, Xva, yva, epochs=60, batch_size=32, patience=10, spec_augment=True, verbose=0):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), min_lr=1e-6
        ),
    ]
    return model.fit(
        make_dataset(Xtr, ytr, batch_size, training=True, spec_augment=spec_augment),
        validation_data=make_dataset(Xva, yva, batch_size),
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )


def evaluate(model, X, y, class_names):
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(1)
    loss = float(tf.keras.losses.sparse_categorical_crossentropy(y, probs).numpy().mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(len(class_names)), average="macro", zero_division=0
    )
    pc_p, pc_r, pc_f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(class_names)), zero_division=0
    )
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            c: {"precision": float(pc_p[i]), "recall": float(pc_r[i]), "f1": float(pc_f1[i]), "support": int(support[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(class_names))).tolist(),
    }


def load_model(path):
    return tf.keras.models.load_model(path, compile=False)
