---
sidebar_position: 8
title: "Offline & air-gapped install"
description: "Install Hermes on a machine with no outbound network access using a self-contained payload archive"
---

# Offline and air-gapped install

Hermes can be installed on a machine that has **no outbound network access at
all**. The transfer is a single self-contained archive you build on a connected
machine, plus a JSON manifest you use to verify it arrived intact.

There are two network-free paths, and they suit different situations:

| Situation | Path |
| --- | --- |
| Windows or macOS desktop | The `Hermes-Setup` installer (`.msix` / `.dmg`) already contains a sealed payload. Copy the installer and run it. |
| Linux, servers, or any platform from source | **This page.** Build a payload archive on a connected machine and install from it. |

:::tip Which machines can the archive install onto?
A payload pins the exact standalone Python interpreter, compiled wheels and
native tool binaries for the machine that built it. **It is not portable across
platforms or CPU architectures.** Build one payload per target — for example
one on `linux-arm64` and one on `linux-x64`. The manifest records the target so
a mismatched transfer is detectable.
:::

## How it works

`hermes pm bundle` already produces everything a Hermes install needs: a repo
snapshot, a standalone interpreter, the native tool store, a sealed PM runtime,
a fully built virtual environment with all extras, and a lock-scoped `uv-cache`
whose wheels let later dependency changes resolve without a network.

An offline install is therefore not a special install mode — it is *unpacking a
payload that was built elsewhere*. `scripts/offline/pack.py` turns a payload
into a transferable archive plus a manifest; `scripts/offline/install.sh`
verifies that manifest and unpacks the payload on the air-gapped machine. Both
scripts use only the standard library, `tar`, and a SHA-256 tool.

## On the connected machine

Pick a machine on the **same platform and architecture** as the target, with
Python 3.11+ available, and build the payload:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Build the payload. This downloads the interpreter, tools and wheels, and
# compiles a venv with all extras; expect several minutes and a few GB.
hermes pm bundle --out /tmp/hermes-payload --ref HEAD
```

`pm bundle` verifies the payload it produced before it finishes. Then pack it
for transfer:

```bash
python3 scripts/offline/pack.py \
    --payload /tmp/hermes-payload \
    --out /tmp/hermes-transfer \
    --verify
```

That writes three files:

| File | Purpose |
| --- | --- |
| `hermes-offline-<target>.tar.gz` | The payload archive. |
| `hermes-offline-<target>.manifest.json` | Target, source revision, archive SHA-256, and a SHA-256 per payload file. |
| `hermes-offline-<target>.tar.gz.sha256` | The archive digest, for verifying the copy by hand. |

Copy **all three** to the removable media, then confirm the copy before you
leave the connected machine:

```bash
sha256sum -c /tmp/hermes-transfer/hermes-offline-*.tar.gz.sha256   # Linux
shasum -a 256 -c /tmp/hermes-transfer/hermes-offline-*.tar.gz.sha256   # macOS
```

## On the air-gapped machine

Copy the three files across, then install. Only `bash`, `tar` and a SHA-256
tool are required — there is no `curl | bash`, no clone, and no package
download:

```bash
tar -xzf hermes-offline-<target>.tar.gz -C /tmp      # optional: inspect first
./scripts/offline/install.sh \
    --archive /tmp/hermes-transfer/hermes-offline-<target>.tar.gz \
    --manifest /tmp/hermes-transfer/hermes-offline-<target>.manifest.json
```

The installer:

1. verifies the archive SHA-256 against the manifest **before extracting
   anything** — a corrupt or substituted transfer writes nothing;
2. unpacks the payload to `~/.hermes/payload`, moving any previous payload
   aside rather than deleting it, so a failed upgrade leaves the old install
   runnable;
3. symlinks `hermes`, `hermes-agent` and `hermes-acp` into `~/.local/bin`,
   leaving any launcher it does not own untouched; and
4. runs `hermes --version` from the installed payload to prove it executes.

Then:

```bash
export PATH="$HOME/.local/bin:$PATH"
hermes --version
```

### Options

| Flag | Effect |
| --- | --- |
| `--dest DIR` | Unpack somewhere other than `~/.hermes` (also `HERMES_INSTALL_DIR`). |
| `--bin-dir DIR` | Link launchers somewhere other than `~/.local/bin` (also `HERMES_BIN_DIR`). |
| `--no-verify` | Skip digest verification. **Not recommended** — this is the only check that the transfer is intact. |
| `--no-shell` | Do not modify shell rc files. |
| `--dry-run` | Report what would happen and change nothing. |

## Verifying an existing install offline

Because the payload ships its own interpreter, tools and wheel cache,
`hermes pm doctor` works with the network down and is the fastest way to
confirm a machine is healthy:

```bash
hermes pm doctor
hermes tools list
```

## Updating an air-gapped install

Payloads are not patched incrementally. To update, build a new payload on the
connected machine, pack it, and run the installer again on the target. The
previous payload is moved to `~/.hermes/payload.previous-<timestamp>` rather
than deleted, so you can roll back by swapping the directories back.

## Troubleshooting

**`archive digest mismatch`** — the archive does not match the manifest. The
copy is corrupt, or the two files came from different builds. Re-copy both from
the same `pack.py` run.

**`no manifest found next to the archive`** — the `.manifest.json` was not
copied. Transfer it alongside the archive.

**The launchers are not on `PATH`** — open a new shell, or run
`export PATH="$HOME/.local/bin:$PATH"`. If an existing `hermes` from pipx or
another checkout is present, the installer deliberately leaves it alone; point
your shell at `~/.local/bin` first, or use `--bin-dir` to install elsewhere.

**`hermes --version` fails right after install** — the payload target probably
does not match the machine. Compare the `target` field in the manifest with the
output of `uname -sm` (Linux/macOS) and rebuild the payload on a matching
machine.
