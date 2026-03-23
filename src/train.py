import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 64    # small = fast
EPOCHS     = 3
BATCH_SIZE = 16
MAX_IMGS   = 100   # 100 sketches + 100 pokemon = 200 samples total

# ── Data ──────────────────────────────────────────────────────────────────────
def load_images(folder: str, label: int) -> tuple:
    images, labels = [], []
    files = sorted(Path(folder).glob("*.jpg"))[:MAX_IMGS]
    for f in files:
        img = tf.keras.utils.load_img(f, target_size=(IMG_SIZE, IMG_SIZE))
        images.append(tf.keras.utils.img_to_array(img) / 255.0)
        labels.append(label)
    return images, labels


print("Loading data...")
imgs_a, lbls_a = load_images("data/sketch2pokemon/trainA", 0)   # sketches  → 0
imgs_b, lbls_b = load_images("data/sketch2pokemon/trainB", 1)   # pokemon   → 1

X = np.array(imgs_a + imgs_b, dtype=np.float32)
y = np.array(lbls_a + lbls_b, dtype=np.float32)

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
print(f"Dataset: {len(X)} images  ({IMG_SIZE}x{IMG_SIZE})")

# ── MLflow ────────────────────────────────────────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment4_Salma")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    mlflow.log_params({
        "img_size":   IMG_SIZE,
        "epochs":     EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_imgs":   MAX_IMGS,
    })

    # ── Model ─────────────────────────────────────────────────────────────────
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="sketch_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    val_accuracy = float(history.history["val_accuracy"][-1])
    val_loss     = float(history.history["val_loss"][-1])

    mlflow.log_metrics({"val_accuracy": val_accuracy, "val_loss": val_loss})
    print(f"\nFinal val_accuracy: {val_accuracy:.4f}")

    # ── Export run ID ─────────────────────────────────────────────────────────
    with open("model_info.txt", "w") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"val_accuracy={val_accuracy:.4f}\n")
        f.write(f"val_loss={val_loss:.4f}\n")

    print("model_info.txt written.")