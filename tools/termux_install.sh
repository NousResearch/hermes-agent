#!/usr/bin/env bash
# termux_install.sh — reproducible Hermes Agent install for Android/Termux.
# Idempotent: safe to re-run; every step skips if already done.
#
# Usage:  bash tools/termux_install.sh
#
# Tested: Samsung Galaxy S25 (SM-S921E), Android 16, Termux (F-Droid build).

set -euo pipefail

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
HERMES_DIR="${HERMES_DIR:-$HOME/hermes-agent}"
VENV_DIR="$HERMES_DIR/venv"

log()  { printf '\033[1;34m[termux-install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[termux-install]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# --- 0. Sanity -----------------------------------------------------------------
[ -d "$PREFIX" ] || die "Termux prefix not found at $PREFIX — run inside Termux."
command -v pkg >/dev/null || die "pkg not found — run inside Termux."

# --- 1. System packages --------------------------------------------------------
log "Updating package lists"
pkg update -y >/dev/null 2>&1 || true

log "Installing build toolchain (first run takes a while)"
pkg install -y python python-pip clang pkg-config libffi openssl rust \
  binutils libc++ git curl || die "pkg install failed"

# --- 2. Environment (idempotent) ----------------------------------------------
ENV_FILE="$HOME/env.sh"
if [ ! -f "$ENV_FILE" ] || ! grep -q TMPDIR "$ENV_FILE"; then
  log "Writing $ENV_FILE"
  cat > "$ENV_FILE" <<EOF
export PREFIX=$PREFIX
export HOME=$HOME
export LD_LIBRARY_PATH=$PREFIX/lib
export PATH=$PREFIX/bin:/system/bin
export ANDROID_API_LEVEL=\$(/system/bin/getprop ro.build.version.sdk)
export DEBIAN_FRONTEND=noninteractive
export TMPDIR=$PREFIX/tmp
EOF
fi
# shellcheck disable=SC1090
. "$ENV_FILE"

# --- 3. Hermes checkout + venv --------------------------------------------------
if [ ! -d "$HERMES_DIR/.git" ]; then
  log "Cloning hermes-agent into $HERMES_DIR"
  git clone https://github.com/NousResearch/hermes-agent.git "$HERMES_DIR"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "Creating virtualenv"
  python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

log "Installing hermes-agent[termux] (editable, source build)"
pip install --upgrade pip wheel >/dev/null
UVLOOP_USE_SYSTEM_LIBUV=1 pip install -e "$HERMES_DIR[termux]" \
  -c "$HERMES_DIR/constraints-termux.txt"

ln -sf "$VENV_DIR/bin/hermes" "$PREFIX/bin/hermes"
command -v hermes >/dev/null || die "hermes not on PATH after symlink"

# --- 4. First-run smoke test -----------------------------------------------------
log "Smoke test: 'hermes --version'"
hermes --version

log "Done. Configure ~/.hermes/.env with your provider keys, then run: hermes gateway run"
log "Full guide: docs/android-termux.md"
