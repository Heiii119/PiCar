#!/usr/bin/env python3
import os
import sys
import glob
import json
import csv

# Use non-interactive backend (safe over SSH/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hardcoded default to your sessions root so you can just run the script
DEFAULT_SESSIONS_ROOT = os.path.expanduser("~/PiCar/data")
SESSION_PREFIX = "session_"

def find_sessions(root, prefix=SESSION_PREFIX):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        print(f"[error] Sessions root not found: {root}")
        return []
    sessions = []
    try:
        names = sorted(os.listdir(root))
    except Exception as e:
        print(f"[error] Cannot list directory {root}: {e}")
        return []
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

def find_keras_in_session(session_dir):
    keras_files = glob.glob(os.path.join(session_dir, "*.keras"))
    if not keras_files:
        print(f"[error] No .keras files found in: {session_dir}")
        return None
    if len(keras_files) > 1:
        print("[info] Multiple .keras files found; using the first. To choose a specific file, move others out temporarily.")
        for f in keras_files:
            print("  -", f)
    return keras_files[0]

def load_model(path):
    from tensorflow import keras
    print(f"[info] Loading model: {path}")
    return keras.models.load_model(path)

def extract_history_from_model(model):
    # Try common places where history may be stored
    hist = getattr(model, "history", None)
    if hist is not None and hasattr(hist, "history"):
        return hist.history
    if hasattr(model, "_training_history") and isinstance(model._training_history, dict):
        return model._training_history
    try:
        cfg = model.get_config()
        if isinstance(cfg, dict):
            for key in ("training_history", "history", "fit_history"):
                if key in cfg and isinstance(cfg[key], dict):
                    return cfg[key]
    except Exception:
        pass
    return None

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
        # Build dict of series from columns (skip 'epoch' if present)
        keys = [k for k in rows[0].keys() if k and k.lower() != "epoch"]
        series = {k: [] for k in keys}
        for r in rows:
            for k in keys:
                v = r.get(k, "")
                if v == "" or v is None:
                    # pad with last value or skip; here we try to cast or leave empty
                    continue
                try:
                    series[k].append(float(v))
                except ValueError:
                    # Non-numeric; skip
                    pass
        return series
    except Exception as e:
        print(f"[error] Failed to read CSV {csv_path}: {e}")
        return None

def find_external_history(session_dir):
    # Prefer JSON, then CSV
    candidates_json = [
        os.path.join(session_dir, "history.json"),
        os.path.join(session_dir, "training_history.json"),
    ]
    candidates_csv = [
        os.path.join(session_dir, "history.csv"),
        os.path.join(session_dir, "training_history.csv"),
    ]
    for p in candidates_json:
        if os.path.isfile(p):
            print(f"[info] Found external history JSON: {p}")
            hist = load_history_json(p)
            if hist:
                return hist, os.path.splitext(os.path.basename(p))[0]
    for p in candidates_csv:
        if os.path.isfile(p):
            print(f"[info] Found external history CSV: {p}")
            hist = load_history_csv(p)
            if hist:
                return hist, os.path.splitext(os.path.basename(p))[0]
    return None, None

def plot_curves(history, out_prefix="model"):
    if not history:
        print("[warn] No history dict to plot.")
        return []

    # Determine epochs length from the first series
    first_series = next((v for v in history.values() if hasattr(v, "__len__")), [])
    epochs = range(1, len(first_series) + 1)
    saved = []

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
        out = f"{out_prefix}_loss.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out)
        print("[ok] Saved:", out)

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
            out = f"{out_prefix}_{m}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(out)
            print("[ok] Saved:", out)

    # Plot any other numeric series
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
            out = f"{out_prefix}_{k}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(out)
            print("[ok] Saved:", out)

    if not saved:
        print("[warn] No standard keys (loss/accuracy) found in history.")
    return saved

def main():
    # Optional: user can still override by passing a path or using SESSIONS_ROOT
    env_root = os.environ.get("SESSIONS_ROOT")
    if len(sys.argv) > 1:
        sessions_root = os.path.expanduser(sys.argv[1])
    elif env_root:
        sessions_root = os.path.expanduser(env_root)
    else:
        sessions_root = DEFAULT_SESSIONS_ROOT

    sessions = find_sessions(sessions_root)
    if not sessions:
        print(f"[error] No session directories found under {sessions_root} (expected names like {SESSION_PREFIX}YYYYMMDD_HHMMSS).")
        print("Hint: Place your session_* folders under this path, or run with a custom root, e.g.:")
        print(f"  python3 {os.path.basename(__file__)} /path/to/your/data")
        print("or set env var:")
        print(f"  SESSIONS_ROOT=/path/to/your/data python3 {os.path.basename(__file__)}")
        sys.exit(1)

    session_dir = pick_session_interactive(sessions)

    # Try to load model and history from the model file
    model_path = find_keras_in_session(session_dir)
    found_history = None
    out_prefix = None
    if model_path is not None:
        try:
            model = load_model(model_path)
            found_history = extract_history_from_model(model)
            if found_history:
                out_prefix = os.path.splitext(os.path.basename(model_path))[0]
        except Exception as e:
            print(f"[warn] Could not load model or extract history: {e}")

    # If no embedded history, look for external JSON/CSV in the session folder
    if not found_history:
        print("[info] Model has no embedded history; searching for history.json or history.csv...")
        found_history, base = find_external_history(session_dir)
        if found_history and not out_prefix:
            out_prefix = base

    if not found_history:
        print("[warn] No training history found. Add a history.json or history.csv in the session folder and re-run.")
        print("Tip: See the 'save_training_history' helper function below to generate history.json after training.")
        sys.exit(0)

    # Basic sanity check for inconsistent lengths
    if isinstance(found_history, dict):
        lengths = {k: len(v) for k, v in found_history.items() if hasattr(v, '__len__')}
        if lengths:
            nmax = max(lengths.values())
            mismatched = [k for k, l in lengths.items() if l != nmax]
            if mismatched:
                print("[warn] History series have different lengths:", lengths)

    if not out_prefix:
        out_prefix = "model"

    print(f"[info] Saving plots as {out_prefix}_*.png in current directory.")
    plot_curves(found_history, out_prefix=out_prefix)

if __name__ == "__main__":
    main()
