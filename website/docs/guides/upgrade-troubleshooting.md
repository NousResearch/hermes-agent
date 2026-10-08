---
title: "Upgrade Troubleshooting"
description: "Diagnose and fix stuck Hermes upgrades — stale uv index caches, Python version mismatches, leftover update locks, dead desktop builds, and Windows-specific download and spawn failures"
sidebar_label: "Upgrade Troubleshooting"
sidebar_position: 4
---

# Upgrade Troubleshooting

An upgrade that fails repeatedly is almost never one broken thing. `hermes update`
runs a pipeline — fetch, stage a source tree, resolve Python dependencies, build a
virtualenv, publish — and any stage can wedge and leave state behind that makes the
*next* attempt fail differently. This page is organized by symptom: find the one that
matches, clear the state it leaves behind, then re-run.

Two rules cover most cases:

1. **Clear the caches that record package metadata, not the caches that hold packages.**
   A stale index makes a resolvable dependency look unsatisfiable.
2. **A version you read from a file is not the version the app shows.** The CLI, the
   desktop window, and the packaged bundle each read a different stamp.

---

## Quick diagnostic

Run this first. It prints the three values that disagree in almost every broken upgrade.

```bash
# What the code says (git)
cd "$(hermes --install-dir 2>/dev/null || echo ~/.hermes/hermes-agent)"
git rev-parse --short HEAD
git log --oneline HEAD..origin/main | wc -l     # commits behind

# What the CLI reports
hermes --version

# What the desktop window shows: the stamp inside the built bundle
cat apps/desktop/release/win-unpacked/resources/install-stamp.json 2>/dev/null
```

If `git log HEAD..origin/main` is `0` but the desktop window shows an old version, the
code is current and only the **build** is stale — see
[Desktop app shows an old version](#desktop-app-shows-an-old-version).

---

## `has no publish time` / `requirements are unsatisfiable`

```
× No solution found when resolving dependencies for split (markers:
│ python_full_version == '3.14.*' and sys_platform != 'android'):
╰─▶ Because tomli-w{python_full_version >= '3.14'}==1.2.0 has no publish
    time and hermes-agent depends on tomli-w{...}==1.2.0, we can conclude
    that hermes-agent's requirements are unsatisfiable.
```

**This reads like an upstream packaging bug. It is not.** `pyproject.toml` sets
`exclude-newer = "14 days"`, which can only filter a release when the package index
reports its upload time. uv reads the [PEP 691][] simple index, whose `upload-time`
field is `null` for some releases — while the JSON API's `upload_time_iso_8601` for the
same release is populated. So a package that *does* have a publish date is filtered out
as undated.

The giveaway: the blamed package changes between runs (`resvg-py`, then `tomli-w`, then
`browser-harness`). A real upstream omission does not rotate.

**Fix — delete the index caches.** There are two, and PM uses its own:

```bash
rm -rf "$HERMES_HOME/cache/uv"/simple-v*      # PM's cache (pm/packages.py::uv_cache_dir)
rm -rf "${LOCALAPPDATA:-$HOME/.local/share}/uv/cache"/simple-v*   # the global default
```

Then confirm before re-running the upgrade:

```bash
uv lock --python 3.14      # should print "Resolved N packages" in seconds
```

Leave `wheels-v6/`, `archive-v0/`, and `git-v0/` alone — they hold the multi-gigabyte
package payloads and re-downloading them is the slow part. Only `simple-v*` caches
*metadata*.

:::tip Confirm it yourself before changing `pyproject.toml`
Do not add an `exclude-newer-package` exemption to make the error go away. Adding one
pins today's workaround into your checkout, diverges from upstream, and the next
ungpinned dependency fails the same way. Verify with the index API, which reports what
uv actually reads:

```bash
curl -s -H "Accept: application/vnd.pypi.simple.v1+json" \
  https://pypi.org/simple/tomli-w/ | python -c "import json,sys; print(sum(1 for f in json.load(sys.stdin)['files'] if not f.get('upload-time')))"
```

If that prints `0`, the index is fine and the cache is the problem. If it prints a
non-zero count, the exemption is genuinely warranted.
:::

[PEP 691]: https://peps.python.org/pep-0691/

---

## `Python 3.11` in the venv, or 3.14-only wheels missing

```
error: Package `resvg-py` requires-python ">=3.14"
```

Hermes supports **one** Python: 3.14. `.python-version`, `pm/lock.json`
(`python = 3.14.7+20260901`) and a comment in `pyproject.toml` all say so. A venv on
3.11 cannot install `resvg-py`, `browser-harness`, or any `winrt-*` package, so the
dependency stage fails on packages that look fine.

```bash
venv/bin/python --version        # Linux/macOS
venv\Scripts\python.exe --version   # Windows
```

If it is not 3.14.x, rebuild against the pinned interpreter:

```bash
uv sync --frozen --python 3.14
```

:::caution Do not "fix" this by downgrading `.python-version`
The file is correct as shipped. Editing it to match a stale venv is the wrong direction
and costs an afternoon to undo.
:::

---

## `an update is still running` / `source-update completion failed`

```
hermes: source-update completion failed: an update is still running;
wait for it to exit, then relaunch Hermes
```

PM guards concurrent upgrades with marker files. If the process holding them was killed
— a closed terminal, a crashed CLI, a force-quit — the markers survive and every later
upgrade is refused until they expire (about 20 minutes).

```bash
rm -f "$HERMES_HOME/.hermes-update-in-progress"
rm -f "$HERMES_HOME/installs/"*/source-completion-pending
rm -f "$HERMES_HOME/installs/"*/source-completion-attempts
```

Confirm nothing is actually running before deleting them:

```bash
ps aux | grep -E "hermes_cli|uv " | grep -v grep
```

---

## Desktop app shows an old version

The desktop window's version comes from `app.getVersion()`, which reads
`resources/install-stamp.json` **inside the built bundle** — not from `git`, and not
from the checkout's root stamp. Refreshing the root `install-stamp.json` updates the
CLI and leaves the window unchanged. This is why "I updated and it still says 0.17"
repeats after a correct-looking upgrade.

Check what the bundle actually carries:

```bash
cat apps/desktop/release/win-unpacked/resources/install-stamp.json
```

Two failure shapes:

| Symptom | Cause | Fix |
|---|---|---|
| `commit` is old | The bundle was never rebuilt | Rebuild (below) |
| `baseVersion` / `displayVersion` are `null` | The build ran without a resolvable base version, so the UI falls back to the bundled `package.json` | Regenerate the stamp with an explicit version (below) |

Regenerate the stamp, then copy it into the built bundle:

```bash
python scripts/write_install_stamp.py \
  --output apps/desktop/build/install-stamp.json \
  --base-version <version> --distance 0 --update-mechanism self

cp apps/desktop/build/install-stamp.json \
   apps/desktop/release/win-unpacked/resources/install-stamp.json
```

:::caution A dirty checkout poisons the version string
The stamp records `dirty` from `git status`, and a dirty tree renders as
`0.21.6+27252.g34f8ec3.dirty`. Commit or stash local changes first — including the
`package-lock.json` that `npm install` rewrites — or the UI will show a version nobody
can match.
:::

Then rebuild the desktop app, and prefer your **own** terminal over an agent sandbox (see
the Windows section below):

```bash
cd apps/desktop
npm run build
npm run builder -- --dir --win --publish never
```

`win-unpacked.bak` keeps the previous build for rollback.

---

## Windows specifics

### Downloads fail while `github.com` looks reachable

`github.com` answering is not the same as its release assets being reachable. The
binaries live on `release-assets.githubusercontent.com` and `objects.githubusercontent.com`,
which are separately blocked on some networks.

```bash
for h in github.com objects.githubusercontent.com \
         release-assets.githubusercontent.com nodejs.org \
         cdn.npmmirror.com pypi.org; do
  printf '%-38s direct=%s\n' "$h" \
    "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 --noproxy '*' "https://$h" 2>&1)"
done
```

If only a proxy reaches them, point every tool at the same proxy and prefer a mirror for
the two binary hosts:

```bash
export https_proxy=http://127.0.0.1:7890   # your proxy's real port
export npm_config_disturl=https://cdn.npmmirror.com/binaries/node
export NODEJS_ORG_MIRROR=https://cdn.npmmirror.com/binaries/node
export ELECTRON_MIRROR=https://cdn.npmmirror.com/binaries/electron/
```

Find the real port rather than assuming: `7890`, `7897` and `10809` are conventions, not
facts.

### `spawnSync ... EBUSY` on every child process

```
Error: spawnSync C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe EBUSY
  errno: -4082, code: 'EBUSY'
```

The desktop build compiles a C# helper (`hud-modifier-monitor.exe`) during
`stage-native-deps`. Some sandboxes and endpoint-security products refuse to let a
process spawn children — this reproduces with `cmd.exe` and `powershell.exe` too, which
rules out the compiler itself.

Distinguish the two causes:

| Test | Result | Meaning |
|---|---|---|
| Launch the same exe from PowerShell | works | The target is fine; the **parent** is restricted |
| Launch it from the build process | `EBUSY` | Sandbox restriction, not a file problem |

If the parent is restricted, run the build from a normal terminal outside the sandbox. To
keep the source tree portable, `build-hud-modifier-monitor.mjs` honours an opt-in that
routes that single call through a generated batch file:

```bash
export HERMES_HUD_COMPILE_VIA_SHELL=1
```

It is off by default, so upstream builds are unaffected.

### `error: Failed to bytecode-compile ... (os error 231)`

`os error 231` is "all pipe instances are in use" — a captured pipe, not a package
problem. Hand the child the terminal instead of a pipe:

```python
# Wrong: the child inherits a pipe it can deadlock on
subprocess.run([...], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Right
subprocess.run([...])
```

### `schtasks.EXE` blocked

A scheduled-task registration at the end of an upgrade is blocked by policy. The upgrade
itself has already finished by then; the marker files from the "still running" section
above are what need clearing.

---

## Verify a healthy install

```bash
git log --oneline HEAD..origin/main | wc -l   # 0
hermes --version                             # matches the target version
```

From Python, the authoritative check is PM's own predicate:

```bash
venv/bin/python -c "import pm, pathlib; print(pm.venv_is_current())"   # True
```

`venv_is_current()` returning `True` means PM will not re-run dependency resolution on
the next launch. If it stays `False` while the environment looks correct, a fact was
never committed — see `facts.json` under `installs/<id>/`, written by
`pm/install.py::_commit_selection`.

---

## Pre-upgrade checklist

```bash
# 1. Back up anything you cannot recreate
cp -r "$HERMES_HOME" ~/hermes-backup-$(date +%Y%m%d)

# 2. Clear metadata caches (packages are kept)
rm -rf "$HERMES_HOME/cache/uv"/simple-v*
rm -rf "${LOCALAPPDATA:-$HOME/.local/share}/uv/cache"/simple-v*

# 3. Clear stale update markers
rm -f "$HERMES_HOME/.hermes-update-in-progress"
rm -f "$HERMES_HOME/installs/"*/source-completion-{pending,attempts}

# 4. Confirm the interpreter is the supported one
venv/bin/python --version    # 3.14.x

# 5. Confirm nothing is mid-upgrade
ps aux | grep -E "hermes_cli|uv " | grep -v grep

# 6. Keep the tree clean so version stamps stay readable
git status --short
```

## Recovering the old build

Every desktop build rotates `win-unpacked` into `win-unpacked.bak` before writing. To
roll back, copy it back over `win-unpacked`. Configuration, credentials and conversation
history live under `$HERMES_HOME`, not in the build directory, so a rollback never
touches them.