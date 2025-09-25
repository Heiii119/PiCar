#!/usr/bin/env python3
import os
import sys
import glob
import json
import csv
from collections import OrderedDict
import datetime

# Use non-interactive backend (safe over SSH/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIGURE YOUR SESSION
# =========================
DEFAULT_SESSIONS_ROOT = os.path.expanduser("~/PiCar/data")
SESSION_PREFIX = "session_"

# If None, you will be prompted to choose existing session or create new
session_dir = None  # or set e.g. "/home/pi/PiCar/data/session_20250925_222453"

# =========================
# TRAINING PLACEHOLDER (REPLACE)
# =========================
def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(16, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    steering = layers.Dense(1, name="steering")(x)
    throttle = layers.Dense(1, name="throttle")(x)

    model = keras.Model(inputs, [steering, throttle])
    model.compile(
        optimizer="adam",
        loss={"steering": "mse", "throttle": "mse"},
        metrics={"steering": ["mae"], "throttle": ["mae"]},
    )
    return model

def load_datasets():
    import numpy as np
    import tensorflow as tf

    def make_data(n):
        x = np.random.rand(n, 64, 64, 3).astype("float32")
        y1 = np.random.uniform(-1, 1, size=(n, 1)).astype("float32")
        y2 = np.random.uniform(0, 1, size=(n, 1)).astype("float32")
        return x, {"steering": y1, "throttle": y2}

    x_tr, y_tr = make_data(512)
    x_va, y_va = make_data(128)
    train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(32).prefetch(2)
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(32).prefetch(2)
    return train_ds, val_ds

# =========================
# UTILS: HISTORY SAVE/LOAD
# =========================
def save_training_history(history_obj, out_dir, filename_base="history", also_csv=True):
    os.makedirs(out_dir, exist_ok=True)
    hist = getattr(history_obj, "history", None)
    if hist is None or not isinstance(hist, dict):
        raise ValueError("Invalid History object: no .history dict found")
    cleaned = {k: [float(x) for x in hist.get(k, [])] for k in hist.keys()}
    json_path = os.path.join(out_dir, f"{filename_base}.json")
    with open(json_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"[ok] Saved JSON history to {json_path}")
    if also_csv:
        max_len = max((len(v) for v in cleaned.values()), default=0)
        fieldnames = ["epoch"] + list(cleaned.keys())
        csv_path = os.path.join(out_dir, f"{filename_base}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(max_len):
                row = OrderedDict()
                row["epoch"] = i + 1
                for k, series in cleaned.items():
                    row[k] = series[i] if i < len(series) else ""
                writer.writerow(row)
        print(f"[ok] Saved CSV history to {csv_path}")

def load_history_json(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: list(v) for k, v in data.items()}
    except Exception as e:
        print(f"[error] Failed to read JSON {json_path}: {e}")
    return None

def load_history_csv(csv_path):
    try:
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return None
        all_keys = [k for k in rows[0].keys() if k]
        epoch_like = {k for k in all_keys if k.lower() in ("epoch", "epochs", "step", "steps")}
        keys = [k for k in all_keys if k not in epoch_like]
        series = {k: [] for k in keys}
        numeric_keys = set()
        for r in rows:
            for k in keys:
                v = r.get(k, "")
                if v == "" or v is None:
                    continue
                try:
                    series[k].append(float(v))
                    numeric_keys.add(k)
                except ValueError:
                    pass
        series = {k: v for k, v in series.items() if k in numeric_keys}
        if not series:
            print(f"[warn] CSV has no numeric metric columns. Columns detected: {all_keys}")
            return None
        return series
    except Exception as e:
        print(f"[error] Failed to read CSV {csv_path}: {e}")
        return None

def find_external_history(session_dir):
    print(f"[debug] Scanning for history files in: {session_dir}")
    try:
        for name in sorted(os.listdir(session_dir)):
            print("  -", name)
    except Exception as e:
        print(f"[error] Cannot list {session_dir}: {e}")
    candidates_json = [
        os.path.join(session_dir, "history.json"),
        os.path.join(session_dir, "training_history.json"),
    ]
    candidates_csv = [
        os.path.join(session_dir, "history.csv"),
        os.path.join(session_dir, "training_history.csv"),
        os.path.join(session_dir, "labels.csv"),
    ]
    for p in candidates_json:
        if os.path.isfile(p):
            print(f"[info] Found external history JSON: {p}")
            hist = load_history_json(p)
            if hist:
                return hist, os.path.splitext(os.path.basename(p))[0]
            else:
                print(f"[warn] JSON exists but could not be parsed: {p}")
    for p in candidates_csv:
        if os.path.isfile(p):
            print(f"[info] Found CSV candidate: {p}")
            hist = load_history_csv(p)
            if hist:
                return hist, os.path.splitext(os.path.basename(p))[0]
            else:
                print(f"[warn] CSV exists but yielded no numeric metrics: {p}")
    return None, None

# =========================
# PLOTTING (now saving into a timestamped folder under data)
# =========================
def plot_curves_to_folder(history, output_folder, out_prefix="history"):
    os.makedirs(output_folder, exist_ok=True)
    if not history:
        print("[warn] No history dict to plot.")
        return []
    first_series = next((v for v in history.values() if hasattr(v, "__len__")), [])
    epochs = range(1, len(first_series) + 1)
    saved = []

    def save_fig(name):
        path = os.path.join(output_folder, f"{out_prefix}_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)
        print("[ok] Saved:", path)

    # Loss
    if "loss" in history:
        plt.figure()
        plt.plot(epochs, history["loss"], label="train loss")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
        plt.legend()
        save_fig("loss")

    # Accuracy-like metrics
    acc_keys = [k for k in history.keys() if "acc" in k or "accuracy" in k]
    if acc_keys:
        base = sorted(set(k.replace("val_", "") for k in acc_keys))
        for m in base:
            train_k = m
            val_k = f"val_{m}"
            if train_k not in history and val_k not in history:
                continue
            plt.figure()
            if train_k in history:
                plt.plot(epochs, history[train_k], label=f"train {m}")
            if val_k in history:
                plt.plot(epochs, history[val_k], label=f"val {m}")
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.title(f"Training/Validation {m}")
            plt.legend()
            save_fig(m)

    # Other numeric series
    for k, v in history.items():
        if k in ("loss", "val_loss") or "acc" in k or "accuracy" in k:
            continue
        if hasattr(v, "__len__"):
            plt.figure()
            plt.plot(range(1, len(v) + 1), v, label=k)
            plt.xlabel("Epoch")
            plt.ylabel(k)
            plt.title(k)
            plt.legend()
            safe_k = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in k)
            save_fig(safe_k)

    if not saved:
        print("[warn] No standard keys (loss/accuracy) found in history.")
    return saved

# =========================
# SESSION MANAGEMENT
# =========================
def ensure_session_dir(session_dir):
    if session_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(DEFAULT_SESSIONS_ROOT, f"{SESSION_PREFIX}{ts}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def find_sessions(root, prefix=SESSION_PREFIX):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return []
    try:
        names = sorted(os.listdir(root))
    except Exception:
        return []
    sessions = []
    for name in names:
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith(prefix):
            sessions.append(full)
    return sessions

def pick_session_interactive(sessions):
    print("Available sessions:")
    for i, s in enumerate(sessions, 1):
        print(f" [{i}] {s}")
    while True:
        choice = input(f"Choose a session [1-{len(sessions)}]: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]
        print("Invalid choice, try again.")

# =========================
# MAIN
# =========================
def main():
    global session_dir
    os.makedirs(DEFAULT_SESSIONS_ROOT, exist_ok=True)

    # Choose/create session
    if session_dir:
        session_dir = os.path.expanduser(session_dir)
        os.makedirs(session_dir, exist_ok=True)
    else:
        existing = find_sessions(DEFAULT_SESSIONS_ROOT)
        if existing:
            print("Do you want to use an existing session or create a new one?")
            print("  [1] Choose existing")
            print("  [2] Create new")
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                session_dir = pick_session_interactive(existing)
            else:
                session_dir = ensure_session_dir(None)
        else:
            session_dir = ensure_session_dir(None)

    print(f"[info] Using session directory: {session_dir}")

    # Build/train (replace with your real code)
    model = build_model()
    train_ds, val_ds = load_datasets()
    EPOCHS = 10  # adjust
    print(f"[info] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    # Save model and history in session
    model_path = os.path.join(session_dir, "model.keras")
    print(f"[info] Saving model to {model_path}")
    from tensorflow import keras  # ensure keras import for save
    model.save(model_path)

    print("[info] Saving training history (JSON/CSV) ...")
    save_training_history(history, out_dir=session_dir, filename_base="history", also_csv=True)

    # Decide output folder under data root: history_YYYYMMDD_HHMMSS
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(DEFAULT_SESSIONS_ROOT, f"history_{ts}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"[info] Plot images will be saved into: {output_folder}")

    # Prefer just-saved history.json, fallback to other files (including labels.csv)
    hist_json = os.path.join(session_dir, "history.json")
    if os.path.isfile(hist_json):
        found_history = load_history_json(hist_json)
    else:
        print("[info] No history.json found; searching other files...")
        found_history, _ = find_external_history(session_dir)

    if not found_history:
        print("[warn] No training history found. Ensure validation_data is set for val_loss.")
        sys.exit(0)

    # Save all plots into the timestamped folder, using "history" as prefix
    saved_paths = plot_curves_to_folder(found_history, output_folder=output_folder, out_prefix="history")

    print("[done] Training and plotting complete.")
    print("Saved files:")
    for p in saved_paths:
        print(" -", p)
    print(f"Folder: {output_folder}")

if __name__ == "__main__":
    main()
