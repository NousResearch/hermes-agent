---
sidebar_position: 6
---

# Isolated Desktop instances

Give a remote Hermes agent its **own Desktop application** — its own window,
settings, Chromium profile, and single-instance lock — while still sharing
the one local Hermes installation you already update.

This is **not** [Settings → Connections](./multi-connection-desktop.md).
Connections keep one shared Electron shell and add more agent sources
inside it. Isolated instances are for the case where you tried that and
need a second, independent Desktop.

Create, launch, shortcut, and repair work on **Windows, macOS, and Linux**.

## When to use which

| You want… | Use |
|---|---|
| Several remotes in **one** window, one Settings app, one dock icon | **Settings → Connections** |
| Independent windows, caches, and shell identity (`Hermes Grace`, `Hermes Athena`, ordinary Hermes) | **Isolated Desktop instances** |
| Separate agent state without a second Desktop | [`hermes profile`](./profiles.md) |

An explicit rejection of Connections after you have tried it is a workflow
preference. Keep using isolated instances; do not migrate those remotes back
into the shared-shell registry unless you ask to.

## Open from Settings

On an SSH row in **Settings → Connections**, click **Open as isolated
Desktop**. Hermes keeps the selected registry `connectionId` plus the
full non-secret SSH dial contract (host, user, port, key path, absolute
remote Hermes path, remote profile). If that instance already exists and
the selected row was retargeted, the action **fails closed** instead of
launching a stale route. Remote-gateway and Cloud rows stay in the
shared window.

The SSH connection must already have an **absolute** Remote Hermes path.

## CLI

The remote machine must already have Hermes and a working SSH alias:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 <alias> \
  "bash -lc 'command -v hermes; hermes --version; hermes profile list'"
```

Then, on the machine that runs Desktop:

```bash
hermes desktop instance create grace \
  --ssh-host grace \
  --remote-hermes-path /home/you/.local/bin/hermes \
  --remote-profile default \
  --display-name "Hermes Grace"

hermes desktop instance create athena \
  --ssh-host bear-agent \
  --remote-hermes-path /home/you/.local/bin/hermes \
  --remote-profile default \
  --display-name "Hermes Athena"
```

That writes a non-secret manifest, seeds isolated `connection.json` (SSH
fields only — no token bytes), and installs a platform launcher plus
shortcut.

```bash
hermes desktop instance list
hermes desktop instance launch grace
hermes desktop instance launch grace --deep-link hermes://blueprint/morning
hermes desktop instance shortcut grace
hermes desktop instance repair --all       # after a local Desktop update
hermes desktop instance remove grace       # launcher + shortcut only
hermes desktop instance remove grace --purge-local
```

`remove` never deletes anything on the remote machine. Without
`--purge-local` it also keeps the isolated local home and Electron
userData so you can recreate the launcher later.

Ordinary `hermes desktop` is unchanged and still opens the canonical
local shell.

## What is isolated vs shared

Each named instance gets:

1. Its own Electron `userData` (`HERMES_DESKTOP_USER_DATA_DIR` plus early `--user-data-dir`)
2. Its own local `HERMES_HOME`
3. Its own app / process name and single-instance namespace
4. Its own Windows AppUserModelID (`com.nousresearch.hermes.instance.<name>`)
5. A persisted SSH target, absolute remote Hermes path, and remote profile
6. Its own clickable Desktop shortcut
7. No global quick-entry / HUD-snap hotkey (those stay on the ordinary Desktop)
8. No `hermes://` protocol capture (the ordinary Desktop remains the OS handler)

What stays shared:

- The canonical local Hermes executable / runtime / update path
- On Windows, adjacent Electron resources (via a differently named **hardlink** next to `Hermes.exe`)

What is **never** cloned locally:

- Remote sessions, memory, skills, configuration, and credentials

A copied, symlinked, or hardlinked executable is **not** isolation by itself.
The two state-root overrides plus the distinct app name are the boundary.

## Platform launchers

| Platform | Launch artifact | Shortcut |
|---|---|---|
| Windows | Native `/target:winexe` launcher + named hardlink beside `Hermes.exe` | `.lnk` on the Desktop |
| macOS | `bash` `.command` wrapper (env + `--user-data-dir`) | same `.command` on the Desktop |
| Linux | `bash` wrapper plus a `.desktop` entry | `.desktop` on the Desktop |

On Windows the named hardlink avoids a path-specific AppCompat
`RUNASADMIN` layer on the canonical `Hermes.exe` while still sharing
updates. macOS and Linux launch the shared binary directly with the
isolation environment.

## Deep links

The ordinary Desktop remains the OS `hermes://` handler. Instance-scoped
URLs look like:

```text
hermes://instance/<name>/blueprint/morning
```

The ordinary shell forwards that to the named instance by launching it with
the remainder both in `HERMES_DESKTOP_PENDING_DEEP_LINK` and as a `hermes://`
argv token. If that isolated window is already open, Electron's
`second-instance` handler delivers the remainder the same way ordinary
deep links work. Isolated shells do not re-register the protocol.

## Layout

Defaults (no hardcoded usernames or hosts):

```text
<HERMES_HOME>/desktop-instances/<name>/
  instance.json          # non-secret manifest
  home/                  # isolated HERMES_HOME
  user-data/             # isolated Electron userData
  launcher/              # platform launcher
```

On Windows the named hardlink lives **beside** the shared executable, for
example `…/release/win-unpacked/Hermes Grace.exe` → same bytes as
`Hermes.exe`.

## Updates

Close isolated windows before rebuilding the shared Desktop artifact
(`hermes desktop --force-build` or the in-app updater). A successful
packaged rebuild then refreshes every instance launcher it can; a running
Windows instance keeps its in-use hardlink image and is repaired on the
next clean launch.

If a shortcut opens the ordinary Desktop after an update, run:

```bash
hermes desktop instance repair --all
hermes desktop instance shortcut <name>
```

Update each **remote** Hermes install separately. Isolated instances do
not copy or pin the remote runtime.

## Migrating from a hand-rolled launcher

If you already have a working native launcher (the documented workaround):

1. Leave those launchers and isolated roots in place.
2. Create first-party instances with the same SSH alias, absolute remote
   path, and remote profile — or use **Open as isolated Desktop** on the
   matching SSH Connection. If a `connection.json` already exists in the
   new instance `user-data` directory, Hermes will not overwrite it.
3. Launch from the new shortcut and confirm a visible renderer plus
   `SSH: <host>` in the status badge. A ready backend without a window is
   a failure.
4. Only then retire the old `.lnk` / compiled launcher. Do not delete
   remote state.

To go the other way — from isolated instances back to **one** shared
shell — use Settings → Connections, then `hermes desktop instance remove`
once the shared-shell route is verified. Isolated local roots can stay
until you pass `--purge-local`.

## Reverting / uninstalling the feature

- `hermes desktop instance remove NAME` — drop launcher, named hardlink
  (Windows), and shortcut.
- `--purge-local` — also delete that instance's local home and userData.
- Ordinary Desktop, the canonical Hermes home, and every remote host are
  left alone.

## Pitfalls

- **VBS / Python / `ShellExecute` wrappers** focus the ordinary Desktop
  or lose process-local `HERMES_HOME`. Use the compiled launcher or
  `hermes desktop instance launch`.
- **Win32 error 740** — canonical `Hermes.exe` has a path-specific
  AppCompat run-as-admin layer. The named hardlink is the supported
  workaround; do not change the ordinary app's compatibility tab.
- **`__COMPAT_LAYER=RunAsInvoker`** can start the backend while the
  renderer dies (`render-process-gone`, `exitCode=18`). Do not add
  `--no-sandbox`.
- **Backend-only false positive** — `Remote Hermes backend is ready` is
  not success. You need a visible window and the SSH status badge.
- Isolated shells do not take the global quick-entry hotkey. That does
  not invalidate SSH connectivity.
