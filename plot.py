#!/usr/bin/env python3
import os
import json
import glob
import sys

# Use non-interactive backend (safe over SSH/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_keras_file(data_dir="/data"):
    files = glob.glob(os.path.join(data_dir, "*.keras"))
    if not files:
        print(f"No .keras files found in {data_dir}")
        sys.exit(1)
    if len(files) > 1:
        print("Multiple .keras files found:")
        for f in files:
            print(" -", f)
        print("Specify one with: python plot_keras_history.py /data/your_model.keras")
        sys.exit(1)
    return files[0]

def load_model(path):
    # Lazy import to avoid heavy load when just listing files
    import tensorflow as tf
    from tensorflow import keras
    print(f"Loading model from: {path}")
    model = keras.models.load_model(path)
    return model

def extract_history_from_model(model):
    """
    Try several places:
    - model.history (rare after loading)
    - model._training_history (custom attr)
    - model.get_config() metadata (custom embedding)
    Returns dict or None.
    """
    # 1) Direct attribute
    hist = getattr(model, "history", None)
    if hist and hasattr(hist, "history"):
        return hist.history

    # 2) Custom attribute dict
    if hasattr(model, "_training_history") and isinstance(model._training_history, dict):
        return model._training_history

    # 3) Config-embedded
    try:
        cfg = model.get_config()
        if isinstance(cfg, dict):
            for key in ("training_history", "history", "fit_history"):
                if key in cfg and isinstance(cfg[key], dict):
                    return cfg[key]
    except Exception:
        pass

    return None

def plot_curves(history, out_prefix="model"):
    """
    history is a dict from Keras History.history
    Keys usually include: loss, val_loss, accuracy, val_accuracy, etc.
    """
    if not history:
        print("No history dict to plot.")
        return []

    # Determine epoch count from the first metric list
    first_series = next((v for v in history.values() if hasattr(v, "__len__")), [])
    epochs = range(1, len(first_series) + 1)

    saved = []

    # Loss curves
    if "loss" in history:
        plt.figure()
        plt.plot(epochs, history["loss"], label="train loss")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
        plt.legend()
        loss_png = f"{out_prefix}_loss.png"
        plt.savefig(loss_png, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(loss_png)
        print("Saved:", loss_png)

    # Accuracy curves (several possible metric names)
    acc_keys = [k for k in history.keys() if "acc" in k or "accuracy" in k]
    if acc_keys:
        base_metrics = set(k.replace("val_", "") for k in acc_keys)
        for m in sorted(base_metrics):
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
            acc_png = f"{out_prefix}_{m}.png"
            plt.savefig(acc_png, dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(acc_png)
            print("Saved:", acc_png)

    if not saved:
        print("No standard keys (loss/accuracy) found in history to plot.")
    return saved

def main():
    # Optional arg: explicit model path
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"Path not found: {model_path}")
            sys.exit(1)
    else:
        model_path = find_keras_file("/data")

    model = load_model(model_path)
    history = extract_history_from_model(model)

    if history is None:
        print("No embedded training history found in the model file.")
        print("If you have a saved history JSON/CSV, share its path and I’ll adapt the script.")
        return

    # Basic validation of history dict
    if not isinstance(history, dict):
        print("History found but not a dict. Type:", type(history))
        return

    # Warn if series lengths differ
    lengths = {k: len(v) for k, v in history.items() if hasattr(v, "__len__")}
    if lengths:
        n = max(lengths.values())
        mismatched = [k for k, l in lengths.items() if l != n]
        if mismatched:
            print("Warning: history series have different lengths:", lengths)

    out_prefix = os.path.splitext(os.path.basename(model_path))[0]
    plot_curves(history, out_prefix=out_prefix)

if __name__ == "__main__":
    main()
