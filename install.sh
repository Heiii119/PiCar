#!/usr/bin/env bash
# install.sh — PiCar (TT02) + Picamera2 setup
# - Prefers APT, falls back to pip where appropriate
# - Works on Bookworm and most recent Raspberry Pi OS images
# - Optional rpicam-apps; skipped if unavailable
# - Installs PCA9685 libs via pip (user site)
# - Leaves TensorFlow install to you (TF 2.4.0 assumed already present)

set -euo pipefail

FAILED_PKGS=()
DISABLED_CORAL_FILES=()
REBOOT_NEEDED=0

have_cmd() { command -v "$1" >/dev/null 2>&1; }

apt_install_one() {
  local pkg="$1"
  echo "==> Installing $pkg ..."
  if ! sudo apt-get install -y "$pkg"; then
    echo "WARNING: Failed to install $pkg"
    FAILED_PKGS+=("$pkg")
    return 1
  fi
}

disable_coral_repos() {
  echo "==> Checking for Coral/EdgeTPU APT sources to disable..."
  local changed=0
  shopt -s nullglob
  for f in /etc/apt/sources.list.d/*coral*.list /etc/apt/sources.list.d/*edgetpu*.list /etc/apt/sources.list.d/*cloud*.list; do
    [[ -f "$f" ]] || continue
    if grep -Ei 'coral|edgetpu|packages\.cloud\.google\.com' "$f" >/dev/null 2>&1; then
      echo "   - Disabling $f (commenting out deb lines)"
      if [[ ! -f "${f}.bak" ]]; then
        sudo cp "$f" "${f}.bak" || true
      fi
      if sudo sed -i -E 's/^[[:space:]]*deb /# deb /' "$f"; then
        DISABLED_CORAL_FILES+=("$f")
        changed=1
      fi
    fi
  done
  shopt -u nullglob
  if (( changed == 0 )); then
    echo "   No Coral/EdgeTPU sources found (nothing to disable)."
  fi
}

echo "=== PiCar (TT02) setup — Picamera2 only ==="
if ! have_cmd sudo; then echo "ERROR: sudo not found."; exit 1; fi
if ! have_cmd python3; then echo "ERROR: python3 not found."; exit 1; fi

echo "System info:"
uname -a || true
cat /etc/os-release || true
echo

# Disable Coral/EdgeTPU sources if present
disable_coral_repos

echo "==> Updating APT indexes..."
if ! sudo apt-get update; then
  echo "APT update failed. Retrying while allowing Release info changes..."
  if ! sudo apt-get update --allow-releaseinfo-change; then
    echo "ERROR: apt-get update failed. Check your network and APT sources."
    exit 1
  fi
fi

echo "==> Installing base dependencies via APT..."
apt_install_one i2c-tools || true
apt_install_one python3-pip || true
apt_install_one python3-numpy || true
apt_install_one python3-pil || true
apt_install_one python3-smbus || true
apt_install_one git || true
# Qt backends for Picamera2 preview (optional but recommended if using on-device preview)
apt_install_one python3-pyqt5 || true
apt_install_one qtwayland5 || true

echo "==> Attempting to install Picamera2 via APT, then pip if needed..."
if ! apt_install_one python3-picamera2; then
  echo "   APT python3-picamera2 not available; trying pip (user site)..."
  python3 -m pip install --user --upgrade pip setuptools wheel || true
  if python3 -m pip install --user --upgrade picamera2; then
    echo "   Installed Picamera2 via pip."
  else
    echo "WARNING: Picamera2 pip install failed. On older OS releases, consider upgrading to Raspberry Pi OS Bookworm."
  fi
fi

echo "==> Installing smbus2 (APT or pip fallback)..."
if ! apt_install_one python3-smbus2; then
  echo "   APT python3-smbus2 not available; trying pip (user site)..."
  python3 -m pip install --user --upgrade smbus2 || echo "WARNING: smbus2 pip install failed."
fi

echo "==> Optional: install camera test apps (rpicam-apps/libcamera-apps)..."
if ! apt_install_one rpicam-apps; then
  echo "   rpicam-apps not available; trying libcamera-apps..."
  apt_install_one libcamera-apps || echo "NOTICE: Skipping camera test binaries (you can still use Picamera2)."
fi

echo "==> Enabling I2C (if raspi-config exists)..."
if have_cmd raspi-config; then
  sudo raspi-config nonint do_i2c 0 || echo "NOTE: raspi-config I2C enable returned non-zero (continuing)."
else
  echo "NOTE: raspi-config not found; skipping automatic I2C enable."
fi

# Add user to i2c group
if id -nG "$USER" | grep -qw "i2c"; then
  echo "User '$USER' already in 'i2c' group."
else
  echo "==> Adding '$USER' to 'i2c' group..."
  if sudo usermod -aG i2c "$USER"; then
    REBOOT_NEEDED=1
  else
    echo "WARNING: Could not add '$USER' to 'i2c' group."
  fi
fi

# Device presence hint
if [[ ! -e /dev/i2c-1 ]]; then
  echo "NOTICE: /dev/i2c-1 not present; a reboot may be required after enabling I2C."
  REBOOT_NEEDED=1
fi

echo "==> Installing PCA9685 Python libraries (user site via pip)..."
python3 -m pip install --user --upgrade adafruit-circuitpython-pca9685 adafruit-blinka || echo "WARNING: CircuitPython PCA9685 install failed."
# Legacy package (optional)
python3 -m pip install --user --upgrade Adafruit-PCA9685 || echo "NOTE: Legacy Adafruit-PCA9685 install failed (not required)."

# PATH hint for user-site scripts
USER_BASE=$(python3 -m site --user-base 2>/dev/null || echo "$HOME/.local")
BIN_DIR="$USER_BASE/bin"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo
  echo "NOTE: Your PATH may not include $BIN_DIR."
  echo "Add this to your ~/.bashrc to use user-site tools:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

# Quick Picamera2 verification
echo
echo "==> Verifying Picamera2 import..."
python3 - <<'PYCHECK' || true
try:
    import picamera2  # noqa: F401
    print("Picamera2 import OK.")
except Exception as e:
    print("Picamera2 NOT available:", e)
    print("TIP: On older Raspberry Pi OS (buster/bullseye), upgrade to Bookworm or build Picamera2 from source.")
PYCHECK

# Report failures
if (( ${#FAILED_PKGS[@]} )); then
  echo
  echo "The following APT packages could not be installed:"
  printf '  - %s\n' "${FAILED_PKGS[@]}"
  echo "Some were replaced by pip or are optional."
fi

if (( ${#DISABLED_CORAL_FILES[@]} )); then
  echo
  echo "Temporarily disabled Coral/EdgeTPU APT entries in:"
  printf '  - %s\n' "${DISABLED_CORAL_FILES[@]}"
  echo "Backups (if created) have .bak suffix."
fi

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  echo
  echo "Reboot recommended so I2C and group changes take effect."
fi

echo
echo "Install done."
echo "Test camera quickly with Picamera2 (if preview display available):"
cat <<'PY'
python3 - <<'EOF'
from picamera2 import Picamera2
from time import sleep
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()
print('Preview started. Ctrl+C to stop.')
try:
    while True: sleep(1)
except KeyboardInterrupt:
    pass
picam2.stop()
EOF
PY

echo
echo "Then run your scripts, e.g.:"
echo "  python3 scripts/drive_train_autopilot_picam2.py"
