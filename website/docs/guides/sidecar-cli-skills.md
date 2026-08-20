---
sidebar_position: 13
title: "Sidecar CLI skills"
description: "Install host CLIs under $HERMES_HOME/bin so skills can shell out from the gateway and docker sandboxes"
---

# Sidecar CLI skills

Some skills work by shelling out to a **local CLI** (mail clients, identity
helpers, custom ops tools) rather than calling an HTTP API from Python. Hermes
supports that pattern without baking the CLI into the agent image.

## Convention

| Host path | Role |
|-----------|------|
| `$HERMES_HOME/skills/…/SKILL.md` | Instructions: when to use the tool, which commands to run |
| `$HERMES_HOME/bin/<command>` | Executable shim or binary the skill invokes |
| `$HERMES_HOME/secrets/…` (optional) | Credentials the CLI reads — **not** auto-mounted; add via `terminal.docker_volumes` if the sandbox needs them |

`$HERMES_HOME` is `~/.hermes` by default, or `/opt/data` in the official
Docker/Podman gateway image.

## What Hermes does automatically

1. **Local terminal** — `$HERMES_HOME/bin` is already on the agent subshell PATH.
2. **Docker / Singularity sandboxes** — if `$HERMES_HOME/bin` exists, Hermes
   bind-mounts it at `/root/.hermes/bin` (read-only) and prepends that path to
   `PATH` inside the sandbox. Symlinks inside `bin/` are sanitized (same as
   skills mounts) so a malicious link cannot expose arbitrary host paths.
3. **Modal / Daytona / SSH** — files under `$HERMES_HOME/bin` are synced into
   the remote home the same way skills are.
4. **Gateway container image** — `/opt/data/bin` is on the image `PATH`, so
   shims installed into the persistent volume work after `docker exec` /
   gateway restart without hand-editing `PATH`.

You should **not** need to hand-edit `terminal.docker_volumes` just to expose
`$HERMES_HOME/bin`. Extra mounts (tool trees outside Hermes home, secrets,
datasets) still go in `docker_volumes` as today.

## Install shape

```bash
# 1. Skill instructions
mkdir -p "$HERMES_HOME/skills/identity/my-tool"
cp SKILL.md "$HERMES_HOME/skills/identity/my-tool/"

# 2. CLI on PATH for gateway + sandboxes
mkdir -p "$HERMES_HOME/bin"
install -m 755 ./my-tool "$HERMES_HOME/bin/my-tool"

# 3. Optional secrets (host only unless you mount them)
mkdir -p "$HERMES_HOME/secrets/my-tool"
chmod 700 "$HERMES_HOME/secrets" "$HERMES_HOME/secrets/my-tool"
```

In the skill, tell the agent to call `my-tool` (not an absolute path), and
never to print JWTs or private keys into chat.

## Optional secrets in the docker sandbox

Secrets stay off the default mount list on purpose. If a sidecar CLI must read
them inside the docker terminal backend, opt in:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "${HERMES_HOME}/secrets/my-tool:/root/.hermes/secrets/my-tool:ro"
```

Prefer host-absolute paths when the gateway itself runs inside Docker/Podman
and spawns nested sandboxes — nested `docker run -v` resolves paths on the
**host** engine, not at `/opt/data` inside the gateway container. Dual-mount
the app dir at its host path (or set `HERMES_HOME` to that host path) so volume
sources resolve.

## Example consumers

Anything that is “skill + binary under `bin/`” benefits: email CLIs, cloud
CLIs you do not want in the image, and third-party identity helpers. Keep
product-specific enroll/purchase flows in the external package; Hermes only
needs the PATH + mount convention above.
