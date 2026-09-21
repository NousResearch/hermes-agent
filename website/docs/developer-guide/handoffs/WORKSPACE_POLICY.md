# KENSEI Workspace — Source Checkout Policy

Effective 2026-05-31. Non-negotiable for all coding sessions.

## Two Hermes-Agent Checkouts Exist

| Role | Path | Purpose | Edits Allowed? |
|---|---|---|---|
| **Dev / PR source** | `~/repos/KenseiAgent/` | Sahil's personal fork. All patches, features, experiments, PR branches. | **Yes — this is the ONLY place we edit** |
| **Upstream reference** | `~/.hermes/hermes-agent/` | Clean upstream tracking. Used for reading source, diffing against upstream, emergency fallback. | **No — read-only unless explicitly asked** |

## Active Symlinks

```
~/.local/bin/hermes -> ~/repos/KenseiAgent/.venv/bin/hermes
```

The CLI binary runs from **KenseiAgent**. The systemd gateway services hardcode `~/.hermes/hermes-agent/venv/` (see memory note on venv duality).

## Rule

**Before editing ANY hermes-agent source file, verify the absolute path contains `KenseiAgent`, not `hermes-agent`.**

If the file path is under `~/.hermes/hermes-agent/`, **STOP** and ask Sahil whether this is intentional. The default assumption is: edit KenseiAgent, leave hermes-agent read-only.

## Why Two Copies?

- `KenseiAgent` is the branded personal fork where Sahil commits and opens PRs
- `hermes-agent` is the original-name clone that predates the rebrand, still referenced by some systemd service definitions and used as a clean upstream baseline
- Consolidation to one checkout would require repointing systemd services + venv — deferred until a dedicated maintenance window

## Re-apply Check (after `git pull`)

```bash
cd ~/repos/KenseiAgent
git log --oneline -5        # confirm recent commits are yours
git status --short          # confirm no uncommitted drift in hermes-agent files
```
