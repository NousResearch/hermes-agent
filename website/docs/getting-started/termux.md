---
sidebar_position: 3
title: "Android / Termux"
description: "Install Rabbit Agent on Android from its signed Termux APT repository"
---

# Rabbit on Android with Termux

:::danger Termux is currently broken
The Termux package does not work right now. A fix is in progress and will
ship soon. Until then, the steps below may fail or install a package that
does not run.
:::

The Termux package runs Rabbit on **aarch64 (arm64-v8a)** Android devices.
Two APT channels are published under
`https://rabbit-assets.nousresearch.com/releases/termux/<channel>`:

| Channel | APT suite | Contents |
| --- | --- | --- |
| `stable` | `rabbit-stable` | Tagged `vMAJOR.MINOR.PATCH` releases that passed the stable release gate |
| `canary` | `rabbit-canary` | Prerelease builds from canary tags; versions carry `~canary.<timestamp>` |

The steps below use `stable`. To follow prereleases, replace `stable` with
`canary` and `rabbit-stable` with `rabbit-canary` in steps 2 and 4. Both
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
     https://rabbit-assets.nousresearch.com/releases/termux/stable/key.asc \
     -o "$PREFIX/etc/apt/keyrings/rabbit-agent.asc"
   ```

3. Verify its primary fingerprint:

   ```bash
   gpg --show-keys --with-fingerprint "$PREFIX/etc/apt/keyrings/rabbit-agent.asc"
   ```

   The repository key fingerprint is:

   ```text
   C572 B5FD D1A2 9CCF A9A9 12B6 840B 0848 E139 156D
   ```

   If the fingerprint differs, stop. Do not disable signature verification.

4. Add the repository:

   ```bash
   printf '%s\n' \
     "deb [signed-by=$PREFIX/etc/apt/keyrings/rabbit-agent.asc] https://rabbit-assets.nousresearch.com/releases/termux/stable rabbit-stable main" \
     > "$PREFIX/etc/apt/sources.list.d/rabbit-agent.list"
   ```

5. Install Rabbit:

   ```bash
   pkg update
   pkg install rabbit-agent
   ```

6. Configure a provider, then start the TUI:

   ```bash
   rabbit setup
   rabbit --tui
   ```

The `rabbit`, `rabbit-agent`, and `rabbit-acp` commands use the packaged runtimes.
They do not require Termux's `python` or `nodejs` packages.

## Files and updates

| Contents | Location |
| --- | --- |
| Package files | `$PREFIX/lib/rabbit-agent/` |
| Command symlinks | `$PREFIX/bin/rabbit`, `$PREFIX/bin/rabbit-agent`, `$PREFIX/bin/rabbit-acp` |
| Configuration and user data | `~/.rabbit/`, or the selected `RABBIT_HOME` |

Update through APT:

```bash
pkg update
pkg upgrade rabbit-agent
```

`rabbit update` refuses to modify an APT-owned installation.
It prints the package-manager command instead.
Canary versions contain `~canary.<timestamp>` and sort before the corresponding
stable version. Each suite only lists its own channel's packages; to move
between channels, edit the channel path and suite in `rabbit-agent.list`, then
`pkg update && pkg upgrade rabbit-agent`.

## Gateway

This APT installation does not use systemd, launchd, or Windows Scheduled Tasks.
Run the gateway in a Termux session:

```bash
rabbit gateway run
```

For a background process:

```bash
mkdir -p "${RABBIT_HOME:-$HOME/.rabbit}/logs"
nohup rabbit gateway run >> "${RABBIT_HOME:-$HOME/.rabbit}/logs/gateway.log" 2>&1 &
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
pkg uninstall rabbit-agent
```

APT removes the package and its command symlinks. It preserves your configuration, sessions, skills, and memories.

## Troubleshooting

- **Package not found:** verify the repository entry, then run `pkg update`.
- **Signature error:** verify the public key fingerprint. Do not use an unsigned repository or bypass the error.
- **Missing command:** verify that `$PREFIX/bin` is on `PATH`, or reinstall the package.
- **Missing library or TUI bundle:** report `rabbit --version` and the complete error. The core package must not require a local rebuild.
- **Gateway stops with the screen off:** review Android's battery and background-process limits.

For general diagnostics, run `rabbit doctor`.
