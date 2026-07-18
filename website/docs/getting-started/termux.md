---
sidebar_position: 3
title: "Android / Termux"
description: "Run Hermes Agent directly on an Android phone with Termux"
---

# Hermes on Android with Termux

:::warning Tier 2 platform
Termux (Android) is a [Tier 2 platform](./platform-support.md#tier-2). The installer and this guide are maintained on a best-effort basis. Changes on `main` may temporarily break Android-specific dependencies.
:::

Hermes Agent can run directly on Android through [Termux](https://termux.dev/). The supported path is the dedicated native Termux installer, which uses Termux packages and `uv`; it does not use proot, Ubuntu, a stdlib `venv` plus `pip`, or the desktop `.[all]` dependency set.

## What the native installer does

On Termux, the normal `install.sh` entrypoint dispatches to `scripts/install-termux.sh`. That installer:

- installs the native Termux runtime packages with `pkg`; compiler and Rust packages are not installed on the normal arm64 path
- automatically selects CPython 3.13 on arm64; CPython 3.11/3.12 remain available only through an explicit `--python` compatibility selection, and Python 3.14 is rejected
- installs a checksum-pinned Python 3.13.14 aarch64 build side-by-side when no CPython 3.13 interpreter exists, without replacing Termux's `python` or `python3` aliases
- creates the Hermes virtual environment with `uv venv`
- derives an Android-safe dependency graph from `pyproject.toml` and `uv.lock`
- verifies that the graph matches the immutable CPython 3.13 Android arm64 wheel release
- downloads ten native wheels from release `wheelhouse-cp313-android24-arm64-20260719.1`, verifying the pinned `SHA256SUMS` file and every wheel before installation
- installs the complete graph with `--only-binary :all:` so a supported phone never silently starts a local C or Rust build
- installs Hermes as an editable package with `--no-deps` and runs dependency/import smoke checks
- writes a launcher to `$PREFIX/bin/hermes` that clears inherited `PYTHONPATH` and `PYTHONHOME`
- preserves the previous virtual environment until the new install validates successfully

The tested mobile dependency set includes the Hermes CLI, cron support, PTY/background terminal support, Telegram gateway support, MCP, Honcho memory, and ACP. The full desktop/server `.[all]` extra is not the Android contract.

## One-line install

Run this inside the current official Termux app:

```bash
pkg update
pkg install -y curl
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

After installation:

```bash
hermes version
hermes doctor
hermes
```

## Installer options

Pass options through `bash -s --` when using the one-line installer:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh   | bash -s -- --skip-setup --skip-browser
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--branch NAME` | Install a specific branch |
| `--commit SHA` | Pin an exact commit |
| `--dir PATH` | Choose the repository checkout directory |
| `--hermes-home PATH` | Choose the Hermes data directory |
| `--python PATH` | Use a specific CPython 3.11-3.13 interpreter |
| `--android-api-level N` | Override the Android wheel build target; default is 24 |
| `--skip-setup` | Skip provider/API setup during installation |
| `--skip-browser` | Skip optional Node/browser command dependencies |
| `--no-skills` | Do not seed bundled skills |
| `--non-interactive` | Never prompt |

Configuration such as model/provider settings belongs in `~/.hermes/config.yaml`; credentials belong in `~/.hermes/.env`. The installer does not require new user-facing `HERMES_*` environment controls for Android build behavior.

## Explicit repository install

For a review branch or a locally inspected checkout, use the same native installer directly rather than recreating the process with manual `pip` commands:

```bash
pkg update
pkg install -y git curl ca-certificates
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
bash scripts/install-termux.sh --skip-setup
```

To test a branch:

```bash
bash scripts/install-termux.sh --branch my-branch --skip-setup
```

This path is intentionally the same installer contract used by the one-line command.

## Updating

Use the normal updater:

```bash
hermes update
```

On a supported CPython 3.13 arm64 installation, the update path regenerates the curated Termux graph, revalidates the cached immutable wheelhouse, and performs the dependency update with `--only-binary :all:`. Before changing the environment it also runs `uv pip install --dry-run --reinstall --only-binary :all:` against PyPI plus the wheelhouse, proving that every package in the complete graph has a compatible binary even when an older source-built copy is already installed. It does not invoke the psutil patch builder, Cargo, Clang, or maturin.

Before accepting newly pulled code, the updater executes the wheel verifier from that new checkout in a separate Python process. If the new lockfile no longer matches its pinned immutable wheel release, if a release asset is unavailable, if any SHA-256 differs, or if the binary-only dry run finds a package that would require compilation, the updater resets Git to the previous commit and leaves the existing environment unchanged. This lets a future change advance its dependency pins and wheel-release constants together without using stale verifier code from the running process.

The verified cache is stored under:

```text
~/.hermes/cache/termux-wheelhouse/wheelhouse-cp313-android24-arm64-20260719.1/
```

Rerunning the installer remains supported when repairing the repository checkout itself:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Local changes are stashed before an update and restored afterward when Git can apply them cleanly.

## Troubleshooting

### Termux currently provides Python 3.14

Hermes supports Python 3.11 through 3.13, but automatic arm64 installs deliberately use CPython 3.13 so they can consume the immutable wheelhouse. When no CPython 3.13 interpreter is available, including when Termux currently provides only Python 3.14?the native installer stages the exact `python_3.13.14_aarch64.deb` asset from release `termux-aarch64-20260719.9.1` side-by-side without replacing Termux's system aliases. The download is locked to SHA-256 `42376a2a47e50048cb7eca2d0f442fc1895fbca2aee2dee3d2fd82728ea1bd80`; installation stops before extraction if the bytes or package metadata differ. This fallback currently supports aarch64 Termux devices only.

A known compatible interpreter can also be selected explicitly. Selecting 3.11 or 3.12 opts into the native-build compatibility path:

```bash
bash scripts/install-termux.sh --python "$PREFIX/bin/python3.13"
```

### An immutable wheel download or verification fails

Do not bypass the checksum check or remove `--only-binary :all:`. Rerun the installer or `hermes update`; a partially downloaded staging directory is discarded, while an already verified cache is preserved.

```bash
bash scripts/install-termux.sh --skip-setup
```

The immutable release is [`wheelhouse-cp313-android24-arm64-20260719.1`](https://github.com/adybag14-cyber/termux-hermes/releases/tag/wheelhouse-cp313-android24-arm64-20260719.1). Its `SHA256SUMS` asset is itself pinned by Hermes to SHA-256 `916ff13af7e5283f75952b810fb6b7eef86ab3422bc5004c1ee1440d5163ade5`.

CPython 3.11/3.12 or non-arm64 Termux environments cannot consume these CPython 3.13 arm64 wheels. Those unsupported targets retain the older one-time native-build compatibility path and therefore still require the compiler/Rust toolchain.

### A wheel is tagged for the wrong Android API

The immutable release targets Android API 24, which is intentionally independent of the phone's newer runtime Android version. `--android-api-level` only affects the unsupported-target source-build compatibility path; normal CPython 3.13 arm64 installs always use the verified API-24 wheels. Only custom Termux toolchains should override it:

```bash
bash scripts/install-termux.sh --android-api-level 28
```

Do not set the build target from `getprop ro.build.version.sdk`; the phone runtime API may be newer than the interpreter target and can produce wheels that `uv` correctly rejects.

### The old environment stopped working after an interrupted install

The installer moves an existing `venv` aside before creating the replacement. If `uv venv` fails, the old environment is restored automatically. After a later-stage failure, inspect directories named `venv.pre-native-termux-*` inside the checkout before deleting anything.

### `hermes doctor` reports missing system commands

Rerun the installer, or install the missing Termux package directly. Common packages include:

```bash
pkg install -y ripgrep nodejs ffmpeg
```

### Browser tooling does not work

Browser and WhatsApp automation remain experimental on Android. The installer can install the optional Node command dependencies, but Android browser availability and background execution differ from desktop Linux. Use `--skip-browser` when only the core CLI is needed.

## Known limitations on phones

- Docker terminal isolation is unavailable inside Termux.
- Local voice transcription through `faster-whisper` is outside the tested dependency graph because `ctranslate2` does not publish Android wheels.
- Android may suspend Termux background processes, so gateway persistence is best-effort.
- Browser/WhatsApp automation is experimental.
- The supported dependency graph is narrower than desktop/server `.[all]`.

When opening an Android-specific issue, include:

- Android version
- `termux-info`
- `python --version`
- `hermes version`
- `hermes doctor`
- the exact installer command and complete error output
