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

Concretely, it maps to:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## What is not part of the tested path yet?

A few features still need desktop/server-style dependencies that are not published for Android, or have not been validated on phones yet:

- `.[all]` is not supported on Android today
- the `voice` extra is blocked by `faster-whisper -> ctranslate2`, and `ctranslate2` does not publish Android wheels **on bare Termux**
- **works on PRoot Ubuntu** (see [Local STT section](#local-stt-faster-whisper-on-proot-ubuntu))
- automatic browser / Playwright bootstrap is skipped in the Termux installer
- Docker-based terminal isolation is not available inside Termux
- Android may still suspend Termux background jobs, so gateway persistence is best-effort rather than a normal managed service

That does not stop Hermes from working well as a phone-native CLI agent — it just means the recommended mobile install is intentionally narrower than the desktop/server install.

---

## Community-maintained native `pkg` option

:::caution Contributor-operated distribution
This APT repository is **community-maintained by `@adybag14-cyber` and is not an official NousResearch distribution**. NousResearch does not build, sign, host, or audit these packages. Enabling the repository means trusting the contributor-operated repository and its signing key. Termux itself remains a Tier 2 / best-effort platform.
:::

For users who prefer a native package-manager install rather than building Python/Rust dependencies on the phone, a community-maintained APT repository is available. The repository bootstrap and packaging sources are published in [`adybag14-cyber/termux-python`](https://github.com/adybag14-cyber/termux-python), with the Hermes package build in [`adybag14-cyber/termux-hermes`](https://github.com/adybag14-cyber/termux-hermes).

Install the repository key/source and Hermes with:

```bash
curl -fsSL https://raw.githubusercontent.com/adybag14-cyber/termux-python/main/scripts/setup_apt_repo.sh | bash
pkg install hermes-agent
```

The repository signing-key fingerprint currently documented by the community distribution is:

```text
EAD24A2124EFA7393A78B7B14699F966313F7A6B
```

APT-managed Hermes installs are marked with install method `apt`. Hermes therefore does not run its Git self-updater against package-owned files; use the package manager instead:

```bash
pkg update
pkg upgrade hermes-agent
```

Packaging/repository/signing problems for this option should be reported to the community packaging repositories above. Hermes runtime bugs can still be reported here, keeping in mind that Android/Termux support is best-effort.

---

## Option 1: One-line installer

Hermes now ships a Termux-aware installer path:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

On Termux, the installer automatically:

- uses `pkg` for system packages
- creates the venv with `python -m venv`
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

:::warning Supported Python range
Hermes requires **Python >=3.11,&lt;3.14**. Current Termux ships `python`
3.14.x, which is outside that range — the installer detects this, and will
automatically try the [Termux User Repository (TUR)](https://github.com/termux-user-repository/tur)
for a supported interpreter. For a manual install, get one yourself:

```bash
pkg install tur-repo
pkg install python3.13
```

Then use `python3.13` in place of `python` in the commands below
(e.g. `python3.13 -m venv venv`).
:::

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

### 4. Install the tested Termux bundle

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

If you only want the minimal core agent, this also works:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Put `hermes` on your Termux PATH

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` is already on PATH in Termux, so this makes the `hermes` command persist across new shells without re-activating the venv every time.

### 6. Verify the install

```bash
hermes --version
hermes doctor
```

### 7. Start Hermes

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

The tested Termux path skips Node/browser bootstrap on purpose. If you want to experiment with browser tooling later, what you need depends on which backend you use:

- **Cloud browser providers** (Browserbase, Browser Use, Firecrawl) host their own Chromium, so Node.js alone is enough — `agent-browser` resolves lazily via `npx agent-browser` on first use:

  ```bash
  pkg install nodejs-lts
  ```

- **Local browser automation** on Termux needs a real `agent-browser` install — the bare npx fallback is deliberately rejected in local mode as too fragile to advertise as ready:

  ```bash
  pkg install nodejs-lts
  npm install -g agent-browser && agent-browser install
  ```

The browser tool automatically includes Termux directories (`/data/data/com.termux/files/usr/bin`) in its PATH search, so `agent-browser` and `npx` are discovered without any extra PATH configuration.

Treat browser / WhatsApp tooling on Android as experimental until documented otherwise.

---

## Troubleshooting

### `No solution found` when installing `.[all]`

Use the tested Termux bundle instead:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

The blocker is currently the `voice` extra:

- `voice` pulls `faster-whisper`
- `faster-whisper` depends on `ctranslate2`
- `ctranslate2` does not publish Android wheels **on bare Termux**

**Workaround:** If you need local STT and are on **PRoot Ubuntu**, see [Local STT (faster-whisper) on PRoot Ubuntu](#local-stt-faster-whisper-on-proot-ubuntu).

### `uv pip install` fails on Android

Use the Termux path with the stdlib venv + `pip` instead:

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` complains about `ANDROID_API_LEVEL`

Set the API level explicitly before installing:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` says ripgrep or Node is missing

Install them with Termux packages:

```bash
pkg install ripgrep nodejs
```

### Local STT (faster-whisper) on PRoot Ubuntu

If you're running Hermes on **PRoot Ubuntu** (via `proot-distro install ubuntu` inside Termux), you CAN run local voice transcription with `faster-whisper` — the `ctranslate2` blocker only affects bare Termux.

**Tested working setup:**

```bash
# Inside PRoot Ubuntu (not bare Termux)
apt update && apt install -y python3-av ffmpeg
pip install faster-whisper --no-deps
```

Then configure Hermes for local STT in `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: local
  language: en
  local:
    model: base      # or tiny for speed
    compute_type: int8
    device: cpu
```

This works because PRoot Ubuntu provides `glibc` and standard Linux libraries, allowing `ctranslate2` (via `faster-whisper`) to run natively.

**Note:** On bare Termux (no PRoot), STT remains unavailable in the tested path — use cloud STT (OpenAI Whisper API) instead.

### Build failures while installing Python packages

Make sure the build toolchain is installed:

```bash
pkg install clang rust make pkg-config libffi openssl
```

Then retry:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## Known limitations on phones

- Docker backend is unavailable
- local voice transcription via `faster-whisper` is unavailable in the tested path on **bare Termux**
- **works on PRoot Ubuntu** (see [Local STT section](#local-stt-faster-whisper-on-proot-ubuntu))
- browser automation setup is intentionally skipped by the installer
- some optional extras may work, but only `.[termux]` and `.[termux-all]` are currently documented as the tested Android bundles

If you hit a new Android-specific issue, please open a GitHub issue with:

- your Android version
- `termux-info`
- `python --version`
- `hermes doctor`
- the exact install command and full error output
