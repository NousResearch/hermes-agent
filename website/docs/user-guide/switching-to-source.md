---
title: "Switching to a Source Install"
description: "Run a separate source checkout without overwriting packaged files or losing track of user data"
---

# Switching to a source install

A source checkout and a packaged app are separate installations. A bundled
app continues to use its own payload; it does not adopt a nearby checkout.
Use a source-built desktop when you want the GUI to run modified code.

User data normally lives outside the application:

| Host | Default data location |
|---|---|
| Linux, macOS, WSL, Termux | `~/.hermes/` |
| Native Windows | `%LOCALAPPDATA%\hermes\` |
| Official Docker container | `/opt/data/`, mapped to host storage |

`HERMES_HOME` and the selected profile can override these defaults. Record the
actual source and destination homes before changing installations.

## 1. Back up and stop the old runtime

From the existing installation, run:

```bash
hermes backup
hermes gateway status
```

Keep the backup outside any directory you plan to remove. Backups can contain
credentials, so protect them accordingly.

Quit the desktop and stop gateways/services that you own before the handoff.
Desktop quit and `hermes gateway stop` are different operations: quitting the
app does not necessarily stop an independently managed messaging gateway.

The per-profile gateway lock prevents duplicate gateways. Session locks and
SQLite concurrency are separate concerns; starting a second process does not
itself switch SQLite journal mode. For the handoff, avoid mixed code versions
writing the same home while either version performs migrations.

## 2. Clone an independent checkout

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

For development, clone your fork instead and add the canonical repository as
`upstream`. Select the branch or commit before preparing dependencies.
Do not clone into a signed app package or overwrite the packaged runtime.

## 3. Prepare the source runtime

Read the [developer workflow](/reference/package-management#developer-workflow)
for native build prerequisites and current bootstrap limitations. Select your
intended `HERMES_HOME` before preparation, then use the checkout's PM bootstrap:

```bash
bash setup-hermes.sh
source ./activate
python hermes --version
```

On native Windows, use PowerShell:

```powershell
.\setup-hermes.ps1
. .\activate.ps1
python hermes --version
```

The bootstrap reads tool pins from `pm/lock.json` and delegates installation
to PM. Current first-party code requires Python 3.14 (`>=3.14,<3.15`).
The source default is the `all` extra, not the desktop bundle's `--all-extras`.

Activation composes the installed tool environment. `python hermes` explicitly
runs this checkout and avoids an older `hermes` command or MSIX alias on PATH.
`deactivate` restores the shell environment when you finish.

For test dependencies and manual environments, use the
[development setup](/developer-guide/contributing).
See [Package management](/reference/package-management) for selected Python
generations and writable tool storage.

## 4. Select data deliberately

For normal use on the same host, select the same `HERMES_HOME` and profile as
the previous installation. For development, a separate home is safer because
new code can migrate stored data.

POSIX example:

```bash
export HERMES_HOME="$HOME/hermes-source-data"
python hermes setup
python hermes
```

PowerShell example:

```powershell
$env:HERMES_HOME = Join-Path $HOME 'hermes-source-data'
python hermes setup
python hermes
```

If you change the home after preparing PM state, run the bootstrap for that
home before relying on its selected dependencies. Do not assume that changing
the environment variable moves data or copies runtime state.

To build a source desktop, run `python hermes desktop` from the prepared
checkout. Opening the old packaged app still starts its packaged backend.

## Docker users

`/opt/data` is a container path, not necessarily a usable host path. For a bind
mount, use the host-side directory as the source process's `HERMES_HOME`.
For a named volume or Docker Desktop VM storage, stop the old gateway first.
Then export/import a backup or copy data through a controlled mount.
Check ownership and permissions on the destination.

A local `docker build -t hermes-agent .` produces another image-managed install.
It does not turn the running container into a self-updating source checkout.
Recreate the container to use that image. See [Docker](/user-guide/docker).

## Nix and Termux users

A local `nix run .` still runs a Nix-owned derivation. Its package files remain
immutable and updates stay with Nix. Use `nix develop` for a development shell,
or the source procedure above where the host supports it.

The Termux distribution is a bionic APT package. The desktop/server source
bootstrap is not its supported development or repair route. Use the
[Termux guide](/getting-started/termux) for its package and build boundaries.

## Switch back without assuming a downgrade is safe

Stop the source runtime, leave its activation, and open the packaged app.
Inspect which CLI command resolves before using `hermes` again:

```bash
command -v hermes
```

On Windows, use `Get-Command hermes -All`. Do not replace an unrelated command
or execution alias without checking its owner.

A newer source revision can change data formats. Returning to an older package
is not the reverse of a schema migration. Preserve current data and restore a
compatible pre-switch backup if the older package requires it.

Deleting the source checkout does not remove the packaged app. It also does
not automatically collect every PM tool entry or Python generation. Use PM's
diagnostics and garbage collection rather than deleting the shared data root.

## Troubleshooting

- **Wrong version:** inspect command resolution, then use `python hermes --version`
  from the activated checkout.
- **Missing dependencies:** run `python -m pm.cli install` from the intended
  source environment, then restart the affected Hermes process.
- **Gateway already running:** inspect `python hermes gateway status` for the
  selected profile. Stop the identified owner; do not kill unrelated processes.
- **Different skills after first run:** newer code can sync bundled skills into
  the data home. A source checkout is not a read-only view of that home.
