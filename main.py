"""
Heart Disease Prediction from ECG Images — End-to-End Pipeline
---------------------------------------------------------------
- Preprocess: grayscale, resize(128x128), normalize, wavelet-based enhancement
- Models: RandomForest, DecisionTree, CNN (Keras)
- Evaluation: accuracy, precision, recall, f1, confusion matrix
- Explainability: SHAP (trees), Grad-CAM++ (CNN)
- Outputs saved under ./artifacts/
Dataset expectation:
    data/
      normal/   *.png|*.jpg
      disease/  *.png|*.jpg
Run:
    python main.py
    python main.py --epochs 8 --batch 32 --img 128
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

# Reproducibility
import random
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Image & plotting
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Wavelets
import pywt

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels   # ✅ added
import joblib

# DL
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Explainability
import shap

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
(ARTIFACTS/"models").mkdir(parents=True, exist_ok=True)
(ARTIFACTS/"plots").mkdir(parents=True, exist_ok=True)
(ARTIFACTS/"results").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utility: data loading
# ----------------------------
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_images(root: Path) -> List[Path]:
    files = []
    for cls in ["normal", "disease"]:
        d = root/cls
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in SUPPORTED_EXT:
                files.append(p)
    return files

# ----------------------------
# Preprocessing pipeline
# ----------------------------
def preprocess_image(path: Path, target_size: int = 128) -> np.ndarray:
    """Read -> grayscale -> resize -> normalize -> wavelet enhance -> return (H,W) float32 [0,1]"""
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((target_size, target_size), Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    # Wavelet denoise/enhance (simple soft-threshold on detail coefficients)
    coeffs2 = pywt.dwt2(x, 'db2')
    cA, (cH, cV, cD) = coeffs2
    # soft threshold
    def soft(a, t): 
        return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)
    tH = np.median(np.abs(cH)) * 1.2
    tV = np.median(np.abs(cV)) * 1.2
    tD = np.median(np.abs(cD)) * 1.2
    cH2, cV2, cD2 = soft(cH, tH), soft(cV, tV), soft(cD, tD)
    x_rec = pywt.idwt2((cA, (cH2, cV2, cD2)), 'db2')
    # Clip to [0,1]
    x_rec = np.clip(x_rec, 0.0, 1.0)
    return x_rec.astype(np.float32)

def load_dataset(root="data", img_size=128) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    files = find_images(root)
    if not files:
        raise SystemExit(f"No images found under {root}. Expected data/normal and data/disease with images.")
    X, y = [], []
    for p in files:
        cls = p.parent.name.lower()
        label = 0 if cls == "normal" else 1
        X.append(preprocess_image(p, img_size))
        y.append(label)
    X = np.stack(X, axis=0)  # (N, H, W)
    y = np.array(y, dtype=np.int32)
    return X, y

# ----------------------------
# Models
# ----------------------------
def build_cnn(input_shape=(128,128,1)) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------
# Grad-CAM++ for CNN
# ----------------------------
def grad_cam_plus(model: tf.keras.Model, img: np.ndarray, layer_name: str=None) -> np.ndarray:
    """
    Minimal Grad-CAM++ for binary classifier.
    img: (H,W) float32 in [0,1]
    returns heatmap (H,W) in [0,1]
    """
    x = img[None, ..., None]  # (1,H,W,1)
    if layer_name is None:
        # pick last conv layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
        conv_out, preds = grad_model(x)
        loss = preds[:, 0]
        grads = tape1.gradient(loss, conv_out)
        grads2 = tape2.gradient(loss, conv_out)
        grads3 = tape3.gradient(loss, conv_out)
    # Grad-CAM++ weights
    numerator = grads2
    denominator = 2.0 * grads2 + tf.reduce_sum(conv_out * grads3, axis=(1,2), keepdims=True)
    denominator = tf.where(denominator!=0, denominator, tf.ones_like(denominator))
    alphas = numerator / denominator
    weights = tf.nn.relu(grads) * alphas
    cam = tf.reduce_sum(weights * conv_out, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (img.shape[0], img.shape[1])).numpy()[...,0]
    return cam

# ----------------------------
# Train & Evaluate
# ----------------------------
def evaluate_and_log(y_true, y_pred, name: str, out_csv: Path):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # ✅ FIXED PART: handle cases where only 1 class exists
    labels_present = unique_labels(y_true, y_pred)
    target_map = {0: "Normal", 1: "Disease"}
    target_names = [target_map[l] for l in labels_present]

    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    row = dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1)
    df = pd.DataFrame([row])
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    # Save confusion matrix plot
    fig = plt.figure(figsize=(4,4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0,1], ["Normal","Disease"])
    plt.yticks([0,1], ["Normal","Disease"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(ARTIFACTS/"plots"/f"cm_{name}.png", dpi=150)
    plt.close(fig)
    # Save classification report
    pd.DataFrame(report).to_csv(ARTIFACTS/"results"/f"classification_report_{name}.csv")
    print(f"[{name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--img", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.data, args.img)  # (N,H,W), (N,)
    print(f"Loaded: {X.shape[0]} images, size {X.shape[1]}x{X.shape[2]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )

    # ---------------- Random Forest ----------------
    print("Training RandomForest...")
    Xtr_flat = X_train.reshape(len(X_train), -1)
    Xte_flat = X_test.reshape(len(X_test), -1)
    scaler = StandardScaler(with_mean=False)  # images already 0-1, keep sparse-safe
    Xtr_f = scaler.fit_transform(Xtr_flat)
    Xte_f = scaler.transform(Xte_flat)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RNG_SEED, n_jobs=-1
    )
    rf.fit(Xtr_f, y_train)
    joblib.dump((rf, scaler), ARTIFACTS/"models"/"random_forest.pkl")
    y_pred_rf = rf.predict(Xte_f)
    evaluate_and_log(y_test, y_pred_rf, "RandomForest", ARTIFACTS/"results"/"evaluation.csv")

    # SHAP for RF
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xte_f[:100])
        # Mean absolute SHAP per feature then reshape to image
        mean_abs = np.mean(np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values), axis=0)
        heat = mean_abs.reshape(args.img, args.img)
        plt.figure(figsize=(4,4))
        plt.imshow(heat)
        plt.title("RF SHAP mean | ECG pixels")
        plt.axis("off")
        plt.savefig(ARTIFACTS/"plots"/"rf_shap_heat.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"SHAP (RF) skipped: {e}")

    # ---------------- Decision Tree ----------------
    print("Training DecisionTree...")
    dt = DecisionTreeClassifier(random_state=RNG_SEED)
    dt.fit(Xtr_f, y_train)
    joblib.dump((dt, scaler), ARTIFACTS/"models"/"decision_tree.pkl")
    y_pred_dt = dt.predict(Xte_f)
    evaluate_and_log(y_test, y_pred_dt, "DecisionTree", ARTIFACTS/"results"/"evaluation.csv")

    # ---------------- CNN ----------------
    print("Training CNN...")
    Xtr_cnn = X_train[..., None]  # add channel
    Xte_cnn = X_test[..., None]
    model = build_cnn(input_shape=(args.img, args.img, 1))
    ckpt_path = ARTIFACTS/"models"/"cnn_model.keras"
    cb = [
        callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_accuracy", mode="max"),
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    ]
    history = model.fit(
        Xtr_cnn, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=cb
    )
    model.save(ckpt_path)

    # Evaluate CNN
    y_prob = model.predict(Xte_cnn).ravel()
    y_pred_cnn = (y_prob >= 0.5).astype(int)
    evaluate_and_log(y_test, y_pred_cnn, "CNN", ARTIFACTS/"results"/"evaluation.csv")

    # Grad-CAM++ for a few test images
    print("Generating Grad-CAM++ for CNN...")
    n_vis = min(5, len(X_test))
    for i in range(n_vis):
        img = X_test[i]
        heat = grad_cam_plus(model, img)
        # overlay
        overlay = (img - img.min()) / (img.max() - img.min() + 1e-8)
        heat_rgb = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
        base = (overlay*255).astype(np.uint8)
        base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(base_rgb, 0.6, heat_rgb, 0.4, 0)
        out_path = ARTIFACTS/"plots"/f"gradcampp_{i}.png"
        cv2.imwrite(str(out_path), mix)

    # ---------------- Summary ----------------
    df = pd.read_csv(ARTIFACTS/"results"/"evaluation.csv")
    print("\n=== Model Comparison ===")
    print(df.to_string(index=False))
    df.to_csv(ARTIFACTS/"results"/"evaluation.csv", index=False)
    print(f"\nArtifacts saved to: {ARTIFACTS.resolve()}")

if __name__ == "__main__":
    main()
