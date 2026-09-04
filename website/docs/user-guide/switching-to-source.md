# Switching to a Source Install

You started with the desktop app, the installer script, Docker, or Nix —
and now you want to run from a source checkout (to develop on Hermes, use
a branch, or escape a packaged-update issue). This page is that move,
without losing your sessions, memory, skills, or configuration.

**The one invariant: your data lives in `HERMES_HOME` (default
`~/.hermes`), never inside the application.** A source install is just a
new code checkout pointing at the same home. Nothing is copied; nothing
is lost. Switching back later is the same procedure in reverse.

---

## Step 1 — back up (one command)

```bash
hermes backup
```

This snapshots your home (config, sessions, memory, skills, cron jobs).
Not strictly required — the switch doesn't delete anything — but it's
the cheap insurance before any install change.

## Step 2 — clone the checkout

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

For your own development: fork first, clone your fork, and add the
upstream remote:

```bash
git remote add upstream https://github.com/NousResearch/hermes-agent.git
```

## Step 3 — create the venv and install

```bash
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv sync --extra all
```

This uses the committed `uv.lock` — the same dependency set the
packaged builds carry, with hashes.

## Step 4 — run from source

```bash
hermes        # or: hermes gateway / hermes --tui / python -m hermes_cli.main
```

The CLI resolves from your checkout (the venv's `hermes` entry point
points at the repo). Your `HERMES_HOME` is untouched — the source
install reads the same config, sessions, and memory the packaged install
did.

:::warning Windows App Installer / MSIX users
The desktop app's bundled payload and a source checkout are separate
installs that can coexist. If the desktop app is running, its backend
keeps its own payload — stop it (`hermes gateway stop` or quit the app)
before using the source CLI against the same home, so two writers never
share one `state.db` (the gateway uses WAL mode; a second writer flips
journal modes).
:::

## Step 5 — switching back

Just run the packaged command again (open the desktop app, or use the
installer script). Both installs read the same `HERMES_HOME`; the last
one to run owns the session locks. If you stop developing, delete the
checkout — your home survives it.

---

## Docker users

A source checkout replaces the image: run the checkout's `hermes`
directly, or build the image from the checkout
(`docker build -t hermes-agent .`). Your data volume (`/opt/data` by
default — the `HERMES_HOME` inside the container) is mounted, not
copied: point the source install's `HERMES_HOME` at the same volume and
it sees everything the container did.

## Nix users

Use the flake from the checkout (`nix run .` / `nix develop`). The Nix
store paths change per checkout; your `HERMES_HOME` does not.

## What moves where (reference)

| Thing | Location | Moves? |
|---|---|---|
| Sessions, memory, skills, config | `HERMES_HOME` (`~/.hermes`) | **No** — both installs read it |
| Cron jobs | `HERMES_HOME/cron` | No |
| Logs | `HERMES_HOME/logs` | No (both install kinds write the same logs) |
| Tool binaries (pm store) | machine-scoped tools dir | No — shared between installs |
| Python venv | inside the checkout | New — the source venv is the source install |
| Desktop payload | inside the app package | Untouched |

## Troubleshooting

**"ModuleNotFoundError" running `hermes`** — you're outside the venv, or
the venv was built from an old lock. Re-run `uv sync --extra all` inside
the activated venv.

**Two gateways started** — the packaged install's gateway is still
running. `hermes gateway status` shows it; `hermes gateway stop` stops
it. The gateway lock (`gateway.lock`) prevents silent double-runs on
the same profile, but stop one anyway.

**Different versions of the same skill** — skills live in `HERMES_HOME`,
not the checkout; both installs share them. Bundled skills re-sync on
first run of the newer code (the skills-sync step in boot bootstrap).
