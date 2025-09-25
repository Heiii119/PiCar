#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="plot-venv"

echo "[info] Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>/dev/null || echo 'not found'))"
echo "[info] Creating virtual environment at: $VENV_DIR"

# Ensure venv module is available
if ! $PYTHON_BIN -m venv --help >/dev/null 2>&1; then
  echo "[info] python3-venv not found; installing it (requires sudo)..."
  if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y python3-venv
  else
    echo "[error] The venv module is unavailable and apt is not present. Install a Python venv tool, then retry."
    exit 1
  fi
fi

$PYTHON_BIN -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[info] Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

echo "[info] Installing required packages into $VENV_DIR..."
# Matplotlib for plotting, TensorFlow + Keras to load .keras models
pip install "matplotlib>=3.5" "tensorflow>=2.14" "keras>=2.14"

echo "[info] Verifying imports..."
python - <<'PY'
try:
    import matplotlib
    import tensorflow as tf
    import keras
    print("[ok] matplotlib:", matplotlib.__version__)
    print("[ok] tensorflow:", tf.__version__)
    print("[ok] keras:", keras.__version__)
except Exception as e:
    print("[error] Import check failed:", e)
    raise SystemExit(1)
PY

echo "[ok] plot-venv setup finished."
echo ""
echo "To use it now, run:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run your script:"
echo "  python plot_keras_history.py"
