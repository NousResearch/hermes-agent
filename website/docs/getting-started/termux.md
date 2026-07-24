---
sidebar_position: 3
title: "Android / Termux"
description: "Run Hermes Agent directly on an Android phone with Termux"
---

# Hermes on Android with Termux

:::warning Tier 2 platform
Termux (Android) is a [Tier 2 platform](./platform-support.md#tier-2). The installer script and documentation here are maintained on a best-effort basis only. Commits to `main` may break these packages at any point in time.
:::

Hermes Agent can run directly on an Android phone through [Termux](https://termux.dev/).

It gives you a working local CLI on the phone, plus the core extras that are currently known to install cleanly on Android.

## What is supported in the tested path?

The tested Termux bundle installs:

- the Hermes CLI
- cron support
- PTY/background terminal support
- Telegram gateway support (manual / best-effort background runs)
- MCP support
- Honcho memory support
- ACP support

Concretely, after the Android psutil prebuild shown in the manual path below, it maps to:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

On a fresh Android environment, use the one-line installer or follow the full manual sequence below instead of running that pip command by itself.

## What is not part of the tested path yet?

A few features still need desktop/server-style dependencies that are not published for Android, or have not been validated on phones yet:

- `.[all]` is not supported on Android today
- the `voice` extra is blocked by `faster-whisper -> ctranslate2`, and `ctranslate2` does not publish Android wheels
- automatic browser / Playwright bootstrap is skipped in the Termux installer
- Docker-based terminal isolation is not available inside Termux
- Android may still suspend Termux background jobs, so gateway persistence is best-effort rather than a normal managed service

That does not stop Hermes from working well as a phone-native CLI agent — it just means the recommended mobile install is intentionally narrower than the desktop/server install.

---

## Option 1: One-line installer

Hermes now ships a Termux-aware installer path:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

On Termux, the installer automatically:

- uses `pkg` for system packages
- creates the venv with `python -m venv`
- prebuilds the Android psutil compatibility shim before the Python package install
- attempts the broad `.[termux-all]` extra first and falls back to the smaller `.[termux]` extra (then a base install) — the curl installer matches this order automatically
- links `hermes` into `$PREFIX/bin` so it stays on your Termux PATH
- skips the untested browser / WhatsApp bootstrap

If you want the explicit commands or need to debug a failed install, use the manual path below.

---

## Option 2: Manual install (fully explicit)

### 1. Update Termux and install system packages

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

Why these packages?

- `python` — runtime + venv support
- `git` — clone/update the repo
- `clang`, `rust`, `make`, `pkg-config`, `libffi`, `openssl` — needed to build a few Python dependencies on Android
- `nodejs` — optional Node runtime for experiments beyond the tested core path
- `ripgrep` — fast file search
- `ffmpeg` — media / TTS conversions

### 2. Clone Hermes

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL` is important for Rust / maturin-based packages such as `jiter`.

### 4. Prebuild psutil for Android

```bash
python scripts/install_psutil_android.py
```

Python 3.13 on Termux reports `sys.platform == "android"`, and upstream psutil currently rejects that platform during package metadata generation. Hermes ships this temporary compatibility shim so the manual path matches what the one-line installer does automatically.

If the command fails because the build toolchain is missing, re-run the package install from step 1 and retry this step before continuing.

### 5. Install the tested Termux bundle

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

If you only want the minimal core agent, this also works:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 6. Put `hermes` on your Termux PATH

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` is already on PATH in Termux, so this makes the `hermes` command persist across new shells without re-activating the venv every time.

### 7. Verify the install

```bash
hermes version
hermes doctor
```

### 8. Start Hermes

```bash
hermes
```

---

## Recommended follow-up setup

### Configure a model

```bash
hermes model
```

Or set keys directly in `~/.hermes/.env`.

### Re-run the full interactive setup wizard later

```bash
hermes setup
```

### Install optional Node dependencies manually

The tested Termux path skips Node/browser bootstrap on purpose. If you want to experiment with browser tooling later:

```bash
pkg install nodejs-lts
npm install
```

The browser tool automatically includes Termux directories (`/data/data/com.termux/files/usr/bin`) in its PATH search, so `agent-browser` and `npx` are discovered without any extra PATH configuration.

Treat browser / WhatsApp tooling on Android as experimental until documented otherwise.

---

## Troubleshooting

### `No solution found` when installing `.[all]`

Use the tested Termux bundle instead:

```bash
python scripts/install_psutil_android.py
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

The blocker is currently the `voice` extra:

- `voice` pulls `faster-whisper`
- `faster-whisper` depends on `ctranslate2`
- `ctranslate2` does not publish Android wheels

### `uv pip install` fails on Android

Use the Termux path with the stdlib venv + `pip` instead:

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python scripts/install_psutil_android.py
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` complains about `ANDROID_API_LEVEL`

Set the API level explicitly before installing:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` reports `Unsupported Android architecture: armv8l`

Some 32-bit Termux devices report `armv8l`, which maturin does not accept as a build target. Set an explicit Rust target before installing:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
export CARGO_BUILD_TARGET="armv7-linux-androideabi"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

On a 64-bit Termux runtime, use `aarch64-linux-android` instead:

```bash
export CARGO_BUILD_TARGET="aarch64-linux-android"
```

### `hermes doctor` says ripgrep or Node is missing

Install them with Termux packages:

```bash
pkg install ripgrep nodejs
```

### Build failures while installing Python packages

Make sure the build toolchain is installed:

```bash
pkg install clang rust make pkg-config libffi openssl
```

Then retry:

```bash
python scripts/install_psutil_android.py
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## Known limitations on phones

- Docker backend is unavailable
- local voice transcription via `faster-whisper` is unavailable in the tested path
- browser automation setup is intentionally skipped by the installer
- some optional extras may work, but only `.[termux]` and `.[termux-all]` are currently documented as the tested Android bundles

If you hit a new Android-specific issue, please open a GitHub issue with:

- your Android version
- `termux-info`
- `python --version`
- `hermes doctor`
- the exact install command and full error output
