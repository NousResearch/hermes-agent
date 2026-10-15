---
sidebar_position: 3
title: "Android / Termux"
description: "Install Hermes Agent on Android from its signed Termux APT repository"
---

# Hermes on Android with Termux

:::danger Termux is currently broken
The Termux package does not work right now. A fix is in progress and will
ship soon. Until then, the steps below may fail or install a package that
does not run.
:::

:::tip While the package is being fixed: a from-source path that works today
Field-tested on a phone (Galaxy S24 Ultra, Android 16 / arm64, 12 GB RAM, Python 3.14 from the
Termux main repository). Clone the repo, create a venv, install the `termux` extra — and fix these
four things first. Each one is a failure we hit, not a precaution:

```bash
apt update && apt full-upgrade -y -o Dpkg::Options::=--force-confold          # 1
pkg install -y clang rust make pkg-config libffi openssl openssl-tool binutils \
                cmake ninja patchelf libxml2 libxslt nodejs ripgrep ffmpeg libheif   # 2

git clone https://github.com/NousResearch/hermes-agent.git && cd hermes-agent
python -m venv --system-site-packages venv && . venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"                    # 3
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=8
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[termux]'
```

1. **`apt` first, before any `pkg` call.** A fresh Termux can ship a `libcurl` newer than its
   `libssl`, which kills the `curl` binary; `pkg` uses `curl` to pick a mirror, so every `pkg`
   command then fails with *"No mirror or mirror group selected"*. Fix the skew with `apt` (which
   does not need `curl`) and `pkg` works again.
2. **`libheif` is required** by `pillow-heif`, pulled in by the Python 3.14 marker. Without it the
   install stops with *"Package libheif was not found in the pkg-config search path"*.
3. **The Rust build overrides are the difference between minutes and most of an hour.** `pydantic-core`
   ships `[profile.release] lto = "fat", codegen-units = 1`; on a phone that is 20–40 minutes for a
   single package. Cargo honours the environment overrides, and `ANDROID_API_LEVEL` is what
   maturin-based packages such as `jiter` need.
4. **On a phone, prefer the config deny-list over the plugin toggle.** `hermes plugins enable|disable`
   resolves dependencies and compiles Rust **on the device** (measured: >7 minutes with no output,
   ~1.6 GB under `~/.hermes/cache/uv`, free RAM down to ~240 MB). The equivalent config change needs
   no toolchain:

   ```bash
   hermes config set plugins.disabled '["some-plugin"]'
   hermes plugins list --plain
   ```

Full field report with the measurements and the scripts:
[hermes-agent-android-phone-only](https://github.com/Padilha-BR/hermes-agent-android-phone-only).
:::

The Termux package runs Hermes on **aarch64 (arm64-v8a)** Android devices.
Two APT channels are published under
`https://hermes-assets.nousresearch.com/releases/termux/<channel>`:

| Channel | APT suite | Contents |
| --- | --- | --- |
| `stable` | `hermes-stable` | Tagged `vMAJOR.MINOR.PATCH` releases that passed the stable release gate |
| `canary` | `hermes-canary` | Prerelease builds from canary tags; versions carry `~canary.<timestamp>` |

The steps below use `stable`. To follow prereleases, replace `stable` with
`canary` and `hermes-stable` with `hermes-canary` in steps 2 and 4. Both
channels are signed with the same key.

The package includes Python, Node.js, npm, uv, ripgrep, ffmpeg, and their runtime libraries.
CI builds the native Python wheels and the TUI before it creates the package.
The device does not compile core dependencies or assemble its base Python
environment during installation. The package uses Python 3.14 with the bionic
interpreter pin; it does not require the same patch version as desktop CPython.
The wheel closure is core plus `acp`, not all desktop extras.

## Install

Use the standard [Termux](https://termux.dev/) application.
The package requires its standard prefix, `/data/data/com.termux/files/usr`.
Other architectures and renamed Termux application packages are not supported.
The wheels target Android API 24 (`android_24_arm64_v8a`).
Do not use the desktop/server `install.sh` or a glibc Linux archive on this target.

1. Install the tools for repository setup:

   ```bash
   pkg install curl gnupg
   ```

2. Download the public key:

   ```bash
   mkdir -p "$PREFIX/etc/apt/keyrings"
   curl -fsSL \
     https://hermes-assets.nousresearch.com/releases/termux/stable/key.asc \
     -o "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
   ```

3. Verify its primary fingerprint:

   ```bash
   gpg --show-keys --with-fingerprint "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
   ```

   The repository key fingerprint is:

   ```text
   C572 B5FD D1A2 9CCF A9A9 12B6 840B 0848 E139 156D
   ```

   If the fingerprint differs, stop. Do not disable signature verification.

4. Add the repository:

   ```bash
   printf '%s\n' \
     "deb [signed-by=$PREFIX/etc/apt/keyrings/hermes-agent.asc] https://hermes-assets.nousresearch.com/releases/termux/stable hermes-stable main" \
     > "$PREFIX/etc/apt/sources.list.d/hermes-agent.list"
   ```

5. Install Hermes:

   ```bash
   pkg update
   pkg install hermes-agent
   ```

6. Configure a provider, then start the TUI:

   ```bash
   hermes setup
   hermes --tui
   ```

The `hermes`, `hermes-agent`, and `hermes-acp` commands use the packaged runtimes.
They do not require Termux's `python` or `nodejs` packages.

## Files and updates

| Contents | Location |
| --- | --- |
| Package files | `$PREFIX/lib/hermes-agent/` |
| Command symlinks | `$PREFIX/bin/hermes`, `$PREFIX/bin/hermes-agent`, `$PREFIX/bin/hermes-acp` |
| Configuration and user data | `~/.hermes/`, or the selected `HERMES_HOME` |

Update through APT:

```bash
pkg update
pkg upgrade hermes-agent
```

`hermes update` refuses to modify an APT-owned installation.
It prints the package-manager command instead.
Canary versions contain `~canary.<timestamp>` and sort before the corresponding
stable version. Each suite only lists its own channel's packages; to move
between channels, edit the channel path and suite in `hermes-agent.list`, then
`pkg update && pkg upgrade hermes-agent`.

## Gateway

This APT installation does not use systemd, launchd, or Windows Scheduled Tasks.
Run the gateway in a Termux session:

```bash
hermes gateway run
```

For a background process:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/logs"
nohup hermes gateway run >> "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log" 2>&1 &
```

:::warning Android process limits
Android can suspend or terminate background Termux processes.
Battery optimization exemptions and `termux-wake-lock` can help, but do not guarantee persistent operation.
:::

## Limits

The package does not include the `nemo-relay` exporter. Its vendored build
toolchain does not support this target.

The package does not include Electron, local Chromium, or desktop computer-use
tools. A local Docker daemon is not part of the Termux environment. Remote
services have their own requirements and connectivity limits.

Phone-native Termux:API microphone and clipboard adapters are not provided by
this package path. The prebuilt CLI/TUI is not proof of local voice or wake-word
support. Optional integrations and third-party plugins can require dependencies
that do not support Android.

Python 3.14 on this target reports `sys.platform == "android"`. A dependency
or skill gated only to `linux` is not automatically available on Android.

## Uninstall

```bash
pkg uninstall hermes-agent
```

APT removes the package and its command symlinks. It preserves your configuration, sessions, skills, and memories.

## Troubleshooting

- **Package not found:** verify the repository entry, then run `pkg update`.
- **Signature error:** verify the public key fingerprint. Do not use an unsigned repository or bypass the error.
- **Missing command:** verify that `$PREFIX/bin` is on `PATH`, or reinstall the package.
- **Missing library or TUI bundle:** report `hermes --version` and the complete error. The core package must not require a local rebuild.
- **Gateway stops with the screen off:** review Android's battery and background-process limits.

For general diagnostics, run `hermes doctor`.
