---
sidebar_position: 2
title: "Configuration"
description: "Configure Hermes Agent — config.yaml, providers, models, API keys, and more"
---

# Configuration

All settings are stored in the `~/.hermes/` directory for easy access.

:::tip Easiest path to a working `config.yaml`
Run `hermes setup --portal` — one OAuth gets you a model provider and all four Tool Gateway tools without hand-editing YAML. Portal subscribers also get 10% off token-billed providers. See [Nous Portal](/integrations/nous-portal).
:::

## Directory Structure

```text
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Primary agent identity (slot #1 in system prompt)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs (errors.log, gateway.log — secrets auto-redacted)
```

## Managing Configuration

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config set KEY VAL  # Set a specific value
hermes config check        # Check for missing options (after updates)
hermes config migrate      # Interactively add missing options

# Examples:
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

:::tip
The `hermes config set` command automatically routes values to the right file — API keys are saved to `.env`, everything else to `config.yaml`.
:::

## Configuration Precedence

Settings are resolved in this order (highest priority first):

1. **CLI arguments** — e.g., `hermes chat --model anthropic/claude-sonnet-4` (per-invocation override)
2. **`~/.hermes/config.yaml`** — the primary config file for all non-secret settings
3. **`~/.hermes/.env`** — fallback for env vars; **required** for secrets (API keys, tokens, passwords)
4. **Built-in defaults** — hardcoded safe defaults when nothing else is set

:::info Rule of Thumb
Secrets (API keys, bot tokens, passwords) go in `.env`. Everything else (model, terminal backend, compression settings, memory limits, toolsets) goes in `config.yaml`. When both are set, `config.yaml` wins for non-secret settings.
:::

:::tip Org deployments
An administrator can pin specific config and secret values that a standard user
cannot override, via a system-level managed directory. See
[Managed Scope](/user-guide/managed-scope).
:::

## Environment Variable Substitution

You can reference environment variables in `config.yaml` using `${VAR_NAME}` syntax:

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

Multiple references in a single value work: `url: "${HOST}:${PORT}"`. If a referenced variable is not set, the placeholder is kept verbatim (`${UNDEFINED_VAR}` stays as-is). Only the `${VAR}` syntax is supported — bare `$VAR` is not expanded.

For AI provider setup (OpenRouter, Anthropic, Copilot, custom endpoints, self-hosted LLMs, fallback models, etc.), see [AI Providers](/integrations/providers).

### Provider Timeouts

You can set `providers.<id>.request_timeout_seconds` for a provider-wide request timeout, plus `providers.<id>.models.<model>.timeout_seconds` for a model-specific override. Applies to the primary turn client on every transport (OpenAI-wire, native Anthropic, Anthropic-compatible), the fallback chain, rebuilds after credential rotation, and (for OpenAI-wire) the per-request timeout kwarg — so the configured value wins over the legacy `HERMES_API_TIMEOUT` env var.

You can also set `providers.<id>.stale_timeout_seconds` for the non-streaming stale-call detector, plus `providers.<id>.models.<model>.stale_timeout_seconds` for a model-specific override. This wins over the legacy `HERMES_API_CALL_STALE_TIMEOUT` env var.

Leaving these unset keeps the legacy defaults (`HERMES_API_TIMEOUT=1800`s, `HERMES_API_CALL_STALE_TIMEOUT=90`s, native Anthropic 900s). The non-streaming stale detector is auto-disabled for local endpoints when left implicit and can scale upward for very large contexts. Not currently wired for AWS Bedrock (both `bedrock_converse` and AnthropicBedrock SDK paths use boto3 with its own timeout configuration). See the commented example in [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example).

## Update Behavior

`hermes update` settings live under `updates` in `config.yaml`:

```yaml
updates:
  pre_update_backup: false       # Create a full HERMES_HOME zip before every update
  backup_keep: 5                 # Keep this many pre-update backup zips
  non_interactive_local_changes: stash  # stash | discard
```

For git installs, Hermes auto-stashes dirty tracked files and untracked files before checking out the update branch or pulling. Interactive terminal updates prompt before restoring that stash. Non-interactive updates (desktop/chat app, gateway, or `--yes`) use `updates.non_interactive_local_changes`: `stash` restores local source edits after a successful pull, while `discard` drops the update-created stash after a successful pull. Use `discard` only on managed installs where local source edits are never meant to persist.

Before that stash step, Hermes also restores tracked `package-lock.json` diffs left by npm install/build churn. Commit or manually stash intentional lockfile edits before updating.

## Terminal Backend Configuration

Hermes supports six terminal backends. Each determines where the agent's shell commands actually execute — your local machine, a Docker container, a remote server via SSH, a Modal cloud sandbox (direct or via the Nous-managed gateway), a Daytona workspace, or a Singularity/Apptainer container.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | singularity
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  timeout: 180      # Per-command timeout in seconds
  home_mode: auto   # auto | real | profile — subprocess HOME policy
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Container image for Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Container image for Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Container image for Daytona backend
```

For cloud sandboxes such as Modal and Daytona, `container_persistent: true` means Hermes will try to preserve filesystem state across sandbox recreation. It does not promise that the same live sandbox, PID space, or background processes will still be running later.

### Backend Overview

| Backend | Where commands run | Isolation | Best for |
|---------|-------------------|-----------|----------|
| **local** | Your machine directly | None | Development, personal use |
| **docker** | Single persistent Docker container (shared across session, `/new`, subagents) | Full (namespaces, cap-drop) | Safe sandboxing, CI/CD |
| **ssh** | Remote server via SSH | Network boundary | Remote dev, powerful hardware |
| **modal** | Modal cloud sandbox | Full (cloud VM) | Ephemeral cloud compute, evals |
| **daytona** | Daytona workspace | Full (cloud container) | Managed cloud dev environments |
| **singularity** | Singularity/Apptainer container | Namespaces (--containall) | HPC clusters, shared machines |

### Local Backend

The default. Commands run directly on your machine with no isolation. No special setup required.

```yaml
terminal:
  backend: local
```

By default, local tool subprocesses keep your real OS-user `HOME`. This lets
external CLIs such as `git`, `ssh`, `gh`, `az`, `npm`, Claude Code, and Codex
find the credentials and config they already use in your normal shell. Hermes
state is still profile-scoped through `HERMES_HOME`; `HOME` is not how profiles
select config, memory, sessions, or skills.

Hermes does **not** change your system-wide `HOME`, your shell startup files, or
the operating system account home. This setting only controls the environment
passed to subprocesses that Hermes launches through tools such as `terminal`,
background terminal processes, `execute_code`, and ACP helper processes.

#### `terminal.home_mode`

| Mode | Host installs | Containers | Tradeoff |
|---|---|---|---|
| `auto` | Keep the real OS-user `HOME` | Use `{HERMES_HOME}/home` | Recommended default. Host CLIs keep working; container state persists. |
| `real` | Force the real OS-user `HOME` | Force the real OS-user `HOME` if visible | Useful if a parent process accidentally started with `HOME` pointed at a profile home. |
| `profile` | Use `{HERMES_HOME}/home` when it exists | Use `{HERMES_HOME}/home` when it exists | Strict per-profile CLI config isolation, but normal `~/.ssh`, `~/.gitconfig`, `~/.azure`, `~/.config/gh`, Claude/Codex auth, npm state, etc. will not be visible unless you initialize or link them inside the profile home. |

The downside of the default is that host profiles share the same normal
user-level CLI credentials/config under `~`. If you need a profile with a
separate git identity, SSH keys, GitHub CLI login, npm config, or cloud CLI
login, use `home_mode: profile` and initialize those tools inside that profile
home deliberately.

If you intentionally want strict per-profile tool-config isolation, set:

```yaml
terminal:
  home_mode: profile
```

In that mode tool subprocesses use `{HERMES_HOME}/home` as `HOME`. Hermes also
sets `HERMES_REAL_HOME` so scripts can still locate the actual user home when
they need it. Container backends keep using `{HERMES_HOME}/home` in `auto` mode
because that directory lives on the persistent Hermes data volume.

Scripts that need to distinguish profile state from the real user home should
prefer `HERMES_HOME` for Hermes data and `HERMES_REAL_HOME` for the account home:

```python
from pathlib import Path
import os

hermes_home = Path(os.environ["HERMES_HOME"])
real_home = Path(os.environ.get("HERMES_REAL_HOME", os.environ["HOME"]))
```

:::warning
The agent has the same filesystem access as your user account. Use `hermes tools` to disable tools you don't want, or switch to Docker for sandboxing.
:::

### Docker Backend

Runs commands inside a Docker container with security hardening (all capabilities dropped, no privilege escalation, PID limits).

**Single persistent container, shared across Hermes processes.** Hermes starts ONE long-lived container on first use and routes every terminal, file, and `execute_code` call through `docker exec` into that same container — across sessions, `/new`, `/reset`, and `delegate_task` subagents. Working-directory changes, installed packages, files in `/workspace`, and **background processes** all carry over from one tool call to the next, and from one Hermes process to the next. When you close a TUI session, run `/quit`, or start a new `hermes` invocation, the container keeps running and the next Hermes process reuses it via a labeled lookup. See **Container lifecycle** below for the exact teardown rules.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace
  docker_run_as_host_user: false   # See "Running container as host user" below
  docker_forward_env:              # Host env vars to forward into container
    - "GITHUB_TOKEN"
  docker_env:                      # Literal env vars to inject (KEY=value)
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:                  # Host directory mounts
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro for read-only
  docker_extra_args:               # Extra flags appended verbatim to `docker run`
    - "--gpus=all"
    - "--network=host"
  docker_network: true             # false = air-gap the container (--network=none)

  # Resource limits
  container_cpu: 1                 # CPU cores (0 = unlimited)
  container_memory: 5120           # MB (0 = unlimited)
  container_disk: 51200            # MB (requires overlay2 on XFS+pquota)
  container_persistent: true       # Persist /workspace and /root bind-mount dirs

  # Cross-process container reuse (defaults match the "one long-lived
  # container shared across sessions" contract — see Container lifecycle).
  docker_persist_across_processes: true   # Reuse container across Hermes restarts
  docker_orphan_reaper: true              # Sweep abandoned Exited containers at startup

  # Cross-backend lifecycle settings (apply to docker as well)
  timeout: 180                     # Per-command timeout in seconds
  lifetime_seconds: 300            # Idle-reaper window; also feeds 2× orphan-reaper threshold
```

**`docker_env`** vs **`docker_forward_env`**: the former injects literal `KEY=value` pairs you specify in the config (the values live in your `config.yaml` or are passed as a JSON dict via `TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'`). The latter forwards values from your shell or `~/.hermes/.env`, so the actual secret never appears in the config file. Use `docker_forward_env` for tokens and `docker_env` for static knobs the container needs.

**`terminal.docker_extra_args`** (also overridable via `TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'`) lets you pass arbitrary `docker run` flags that Hermes doesn't surface as first-class keys — `--gpus`, `--network`, `--add-host`, alternative `--security-opt` overrides, etc. Each entry must be a string; the list is appended last to the assembled `docker run` invocation so it can override Hermes' defaults if needed. Use sparingly — flags that conflict with the sandbox hardening (capability drops, `--user`, the workspace bind mount) will silently weaken isolation.

**`terminal.docker_network`** (default `true`; env: `TERMINAL_DOCKER_NETWORK`) — set to `false` to run the sandbox container with `--network=none`, cutting off all network egress from agent commands. This applies to the execution container used by `terminal`, `execute_code`, and the file tools. Because containers persist across Hermes processes, flipping this to `false` while an older networked container exists will remove that container and start a fresh air-gapped one (a warning is logged); background processes running inside it are lost. Prefer this key over passing `--network=none` through `docker_extra_args`.

**Requirements:** Docker Desktop or Docker Engine installed and running. Hermes probes `$PATH` plus common macOS install locations (`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, Docker Desktop app bundle). Podman is supported out of the box: set `HERMES_DOCKER_BINARY=podman` (or the full path) to force it when both are installed.

#### Container lifecycle

Every Hermes-managed container is tagged with three labels so subsequent processes (and the orphan reaper) can identify it:

- `hermes-agent=1` — marks it as Hermes-managed
- `hermes-task-id=<sanitized task_id>` — keys the per-task reuse probe
- `hermes-profile=<sanitized profile name>` — scopes reuse and reaping to the active Hermes profile

On startup, Hermes runs `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<profile>` and **attaches to the existing container** when it finds one. If the container is `exited` (e.g. after a Docker daemon restart), it's `docker start`'d and reused — filesystem state and any installed packages survive, but in-container background processes do not.

When a Hermes process exits — `/quit`, closing a TUI session, gateway shutdown, even SIGKILL — the cleanup path is a **no-op for the container in default mode**. The container keeps running. The next Hermes process attaches to it in milliseconds via the label probe. This is the behavior the "one long-lived container shared across sessions" contract requires: it's the only way background processes (npm watchers, dev servers, long-running pytest) survive across sessions.

**The container is only torn down (stopped and `docker rm -f`'d) in these cases:**

| Trigger | When it fires |
|---|---|
| `docker_persist_across_processes: false` | Explicit per-process isolation. Every `cleanup()` does `stop` + `rm -f`. Matches pre-issue-#20561 behavior. |
| Idle reaper (`lifetime_seconds`, default 300s) | Only when the env is `persist_across_processes=false`. Persist-mode envs are no-op'd; container survives the idle sweep. |
| Orphan reaper at next startup | Sweeps **Exited** hermes-labeled containers older than `2 × lifetime_seconds` (default 600s = 10 min), scoped to the current profile. **Running containers are never touched** — sibling-process safety. Set `docker_orphan_reaper: false` to disable. |
| Direct user action | `docker rm -f`, `docker system prune`, Docker Desktop restart. We don't set `--restart=always`, so a host reboot leaves the container `Exited` (its CoW layer survives and gets reused on next startup, but bg processes are gone). |

Edge cases worth knowing:

- **OOM kill of in-container PID 1** transitions the container to `Exited`. Next reuse will `docker start` it; filesystem state survives, bg processes do not.
- **Switching profiles** isolates containers from each other — a container labeled `hermes-profile=work` is invisible to a Hermes process running under `hermes-profile=research`. The orphan reaper is profile-scoped too, so cross-profile containers don't get reaped accidentally, but they also won't get cleaned up automatically until you start Hermes again under their original profile.

Parallel subagents spawned via `delegate_task(tasks=[...])` share this one container — concurrent `cd`, env mutations, and writes to the same path will collide. If a subagent needs an isolated sandbox, it must register a per-task image override via `register_task_env_overrides()`, which RL and benchmark environments (TerminalBench2, HermesSweEnv, etc.) do automatically for their per-task Docker images.

**Security hardening:**
- `--cap-drop ALL` with only `DAC_OVERRIDE`, `CHOWN`, `FOWNER` added back
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- Size-limited tmpfs for `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)

**Credential forwarding:** Env vars listed in `docker_forward_env` are resolved from your shell environment first, then `~/.hermes/.env`. Skills can also declare `required_environment_variables` which are merged automatically.

#### Environment variable overrides

Every key under `terminal:` has an env-var override of the form `TERMINAL_<KEY_UPPERCASE>`. The most useful ones for the Docker backend:

| Env var | Maps to | Notes |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | Base image |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | JSON array: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | JSON dict: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | JSON array of `"host:container[:ro]"` strings |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | JSON array |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_NETWORK` | `docker_network` | `true` / `false` — default `true`; `false` = `--network=none` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — default `true` |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — default `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | CPU cores |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `true` / `false` — controls the bind-mount workspace dirs, distinct from `docker_persist_across_processes` |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | Idle reaper window |
| `TERMINAL_TIMEOUT` | `timeout` | Per-command timeout |
| `HERMES_DOCKER_BINARY` | _none_ | Force a specific docker/podman binary path |

### SSH Backend

Runs commands on a remote server over SSH. Uses ControlMaster for connection reuse (5-minute idle keepalive). Persistent shell is enabled by default — state (cwd, env vars) survives across commands.

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # Keep a long-lived bash session (default: true)
```

**Required environment variables:**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | SSH port |
| `TERMINAL_SSH_KEY` | (system default) | Path to SSH private key |
| `TERMINAL_SSH_PERSISTENT` | `true` | Enable persistent shell |

**How it works:** Connects at init time with `BatchMode=yes` and `StrictHostKeyChecking=accept-new`. Persistent shell keeps a single `bash -l` process alive on the remote host, communicating via temporary files. Commands that need `stdin_data` or `sudo` automatically fall back to one-shot mode.

### Modal Backend

Runs commands in a [Modal](https://modal.com) cloud sandbox. Each task gets an isolated VM with configurable CPU, memory, and disk. Filesystem can be snapshot/restored across sessions.

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # Snapshot/restore filesystem
```

**Required:** Either `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` environment variables, or a `~/.modal.toml` config file.

**Persistence:** When enabled, the sandbox filesystem is snapshotted on cleanup and restored on next session. Snapshots are tracked in `~/.hermes/modal_snapshots.json`. This preserves filesystem state, not live processes, PID space, or background jobs.

**Credential files:** Automatically mounted from `~/.hermes/` (OAuth tokens, etc.) and synced before each command.

### Daytona Backend

Runs commands in a [Daytona](https://daytona.io) managed workspace. Supports stop/resume for persistence.

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB → converted to GiB
  container_disk: 10240            # MB → converted to GiB (max 10 GiB)
  container_persistent: true       # Stop/resume instead of delete
```

**Required:** `DAYTONA_API_KEY` environment variable.

**Persistence:** When enabled, sandboxes are stopped (not deleted) on cleanup and resumed on next session. Sandbox names follow the pattern `hermes-{task_id}`.

**Disk limit:** Daytona enforces a 10 GiB maximum. Requests above this are capped with a warning.

### Singularity/Apptainer Backend

Runs commands in a [Singularity/Apptainer](https://apptainer.org) container. Designed for HPC clusters and shared machines where Docker isn't available.

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB
  container_persistent: true       # Writable overlay persists across sessions
```

**Requirements:** `apptainer` or `singularity` binary in `$PATH`.

**Image handling:** Docker URLs (`docker://...`) are automatically converted to SIF files and cached. Existing `.sif` files are used directly.

**Scratch directory:** Resolved in order: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent` (HPC convention) → `~/.hermes/sandboxes/singularity`.

**Isolation:** Uses `--containall --no-home` for full namespace isolation without mounting the host home directory.

### Common Terminal Backend Issues

If terminal commands fail immediately or the terminal tool is reported as disabled:

- **Local** — No special requirements. The safest default when getting started.
- **Docker** — Run `docker version` to verify Docker is working. If it fails, fix Docker or `hermes config set terminal.backend local`.
- **SSH** — Both `TERMINAL_SSH_HOST` and `TERMINAL_SSH_USER` must be set. Hermes logs a clear error if either is missing.
- **Modal** — Needs `MODAL_TOKEN_ID` env var or `~/.modal.toml`. Run `hermes doctor` to check.
- **Daytona** — Needs `DAYTONA_API_KEY`. The Daytona SDK handles server URL configuration.
- **Singularity** — Needs `apptainer` or `singularity` in `$PATH`. Common on HPC clusters.

When in doubt, set `terminal.backend` back to `local` and verify that commands run there first.

### Remote-to-Host File Sync on Teardown

For the **SSH**, **Modal**, and **Daytona** backends (anywhere the agent's working tree lives on a different machine than the host running Hermes), Hermes tracks files the agent touched inside the remote sandbox and, on session teardown / sandbox cleanup, **syncs the modified files back to the host** under `~/.hermes/cache/remote-syncs/<session-id>/`.

- Triggers on: session close, `/new`, `/reset`, gateway message timeout, `delegate_task` subagent completion when the child used a remote backend.
- Covers the whole tree the agent modified, not just files it explicitly opened. Additions, edits, and deletions are all captured.
- The remote sandbox may have been torn down by the time you go looking; the local `~/.hermes/cache/remote-syncs/…` copy is the authoritative record of what the agent changed.
- Large binary outputs (model checkpoints, raw datasets) are capped by size — the sync skips files over `file_sync_max_mb` (default `100`). Bump that if you expect bigger artifacts to come back.

```yaml
terminal:
  file_sync_max_mb: 100     # default — sync files up to 100 MB each
  file_sync_enabled: true   # default — set false to skip the sync entirely
```

This is how you recover results from ephemeral cloud sandboxes that get destroyed after the session ends, without having to tell the agent to explicitly `scp` or `modal volume put` every artifact.

### Docker Volume Mounts

When using the Docker backend, `docker_volumes` lets you share host directories with the container. Each entry uses standard Docker `-v` syntax: `host_path:container_path[:options]`.

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # Read-write (default)
    - "/home/user/datasets:/data:ro"              # Read-only
    - "/home/user/.hermes/cache/documents:/output" # Gateway-visible exports
```

This is useful for:
- **Providing files** to the agent (datasets, configs, reference code)
- **Receiving files** from the agent (generated code, reports, exports)
- **Shared workspaces** where both you and the agent access the same files

If you use a messaging gateway and want the agent to send generated files via
`MEDIA:/...`, prefer a dedicated host-visible export mount such as
`/home/user/.hermes/cache/documents:/output`.

- Write files inside Docker to `/output/...`
- Emit the **host path** in `MEDIA:`, for example:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- Do **not** emit `/workspace/...` or `/output/...` unless that exact path also
  exists for the gateway process on the host

:::warning
YAML duplicate keys silently override earlier ones. If you already have a
`docker_volumes:` block, merge new mounts into the same list instead of adding
another `docker_volumes:` key later in the file.
:::

Can also be set via environment variable: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (JSON array).

### Docker Credential Forwarding

By default, Docker terminal sessions do not inherit arbitrary host credentials. If you need a specific token inside the container, add it to `terminal.docker_forward_env`.

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes resolves each listed variable from your current shell first, then falls back to `~/.hermes/.env` if it was saved with `hermes config set`.

:::warning
Anything listed in `docker_forward_env` becomes visible to commands run inside the container. Only forward credentials you are comfortable exposing to the terminal session.
:::

### Running the Container as Your Host User

By default Docker containers run as `root` (UID 0). Files created inside `/workspace` or other bind-mounts end up owned by root on the host, so after a session you have to `sudo chown` them before you can edit them from your host editor. The `terminal.docker_run_as_host_user` flag fixes this:

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # default: false
```

When enabled, Hermes appends `--user $(id -u):$(id -g)` to the `docker run` command so files written into bind-mounted directories (`/workspace`, `/root`, anything in `docker_volumes`) are owned by your host user, not root. The trade-off: the container can no longer `apt install` or write to root-owned paths like `/root/.npm` — use a base image whose `HOME` is owned by a non-root user (or add your required tooling at image build time) if you need both.

Leave this `false` (the default) for backwards-compatible behavior. Turn it on when your workflow is mostly "edit mounted host files" and you're tired of `sudo chown -R`.

### Optional: Mount the Launch Directory into `/workspace`

Docker sandboxes stay isolated by default. Hermes does **not** pass your current host working directory into the container unless you explicitly opt in.

Enable it in `config.yaml`:

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

When enabled:
- if you launch Hermes from `~/projects/my-app`, that host directory is bind-mounted to `/workspace`
- the Docker backend starts in `/workspace`
- file tools and terminal commands both see the same mounted project

When disabled, `/workspace` stays sandbox-owned unless you explicitly mount something via `docker_volumes`.

Security tradeoff:
- `false` preserves the sandbox boundary
- `true` gives the sandbox direct access to the directory you launched Hermes from

Use the opt-in only when you intentionally want the container to work on live host files.

### Persistent Shell

By default, each terminal command runs in its own subprocess — working directory, environment variables, and shell variables reset between commands. When **persistent shell** is enabled, a single long-lived bash process is kept alive across `execute()` calls so that state survives between commands.

This is most useful for the **SSH backend**, where it also eliminates per-command connection overhead. Persistent shell is **enabled by default for SSH** and disabled for the local backend.

```yaml
terminal:
  persistent_shell: true   # default — enables persistent shell for SSH
```

To disable:

```bash
hermes config set terminal.persistent_shell false
```

**What persists across commands:**
- Working directory (`cd /tmp` sticks for the next command)
- Exported environment variables (`export FOO=bar`)
- Shell variables (`MY_VAR=hello`)

**Precedence:**

| Level | Variable | Default |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| SSH override | `TERMINAL_SSH_PERSISTENT` | follows config |
| Local override | `TERMINAL_LOCAL_PERSISTENT` | `false` |

Per-backend environment variables take highest precedence. If you want persistent shell on the local backend too:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
Commands that require `stdin_data` or sudo automatically fall back to one-shot mode, since the persistent shell's stdin is already occupied by the IPC protocol.
:::

See [Code Execution](features/code-execution.md) and the [Terminal section of the README](features/tools.md) for details on each backend.

## Skill Settings

Skills can declare their own configuration settings via their SKILL.md frontmatter. These are non-secret values (paths, preferences, domain settings) stored under the `skills.config` namespace in `config.yaml`.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # Example — each skill defines its own keys
```

**How skill settings work:**

- `hermes config migrate` scans all enabled skills, finds unconfigured settings, and offers to prompt you
- `hermes config show` displays all skill settings under "Skill Settings" with the skill they belong to
- When a skill loads, its resolved config values are injected into the skill context automatically

**Setting values manually:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

For details on declaring config settings in your own skills, see [Creating Skills — Config Settings](/developer-guide/creating-skills#config-settings-configyaml).

### Guard on agent-created skill writes

When the agent uses `skill_manage` to create, edit, patch, or delete a skill, Hermes can optionally scan the new/updated content for dangerous keyword patterns (credential harvesting, obvious prompt injection, exfil instructions). The scanner is **off by default** — real agent workflows that legitimately touch `~/.ssh/` or mention `$OPENAI_API_KEY` were tripping the heuristic too often. Turn it back on if you want the scanner to prompt you before the agent's skill writes land:

```yaml
skills:
  guard_agent_created: true   # default: false
```

When on, any flagged `skill_manage` write surfaces as an approval prompt with the scanner's rationale. Accepted writes land; denied writes return an explanatory error to the agent.

### Write approval for skill writes

Independent of the content scanner above, `skills.write_approval` gates **every** agent skill write (create / edit / patch / delete / supporting files) behind your explicit approval — the same approve/deny mechanism as dangerous commands:

```yaml
skills:
  write_approval: false   # false = write freely (default) | true = stage every write for review
```

When on, skill writes are staged under `~/.hermes/pending/skills/` and reviewed with `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>` — from the CLI or any messaging platform. Toggle at runtime with `/skills approval on|off`. Memory has the same gate (`memory.write_approval`, below). Full walkthrough: [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).

## Memory Configuration

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # true = require approval before any memory write
```

With `memory.write_approval: true`, memory writes need your approval before they land: interactive CLI turns prompt inline; messaging sessions and the background self-improvement review stage the write for `/memory pending` → `/memory approve <id>` / `/memory reject <id>` review. Toggle at runtime with `/memory approval on|off`. See [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval).

## Context File Truncation

Controls how much content Hermes loads from each automatic context file before applying head/tail truncation. This applies to files injected into the system prompt such as `SOUL.md`, `.hermes.md`, `AGENTS.md`, `CLAUDE.md`, and `.cursorrules`. It does **not** affect the `read_file` tool.

```yaml
context_file_max_chars: 20000  # default
```

Raise it when you intentionally keep larger identity or project-context files and run models with enough context window to carry them:

```yaml
context_file_max_chars: 25000
```

## File Read Safety

Controls how much content a single `read_file` call can return. Reads that exceed the limit are rejected with an error telling the agent to use `offset` and `limit` for a smaller range. This prevents a single read of a minified JS bundle or large data file from flooding the context window.

```yaml
file_read_max_chars: 100000  # default — ~25-35K tokens
```

Raise it if you're on a model with a large context window and frequently read big files. Lower it for small-context models to keep reads efficient:

```yaml
# Large context model (200K+)
file_read_max_chars: 200000

# Small local model (16K context)
file_read_max_chars: 30000
```

The agent also deduplicates file reads automatically — if the same file region is read twice and the file hasn't changed, a lightweight stub is returned instead of re-sending the content. This resets on context compression so the agent can re-read files after their content is summarized away.

## Tool Output Truncation Limits

Three related caps control how much raw output a tool can return before Hermes truncates it:

```yaml
tool_output:
  max_bytes: 50000        # terminal output cap (chars)
  max_lines: 2000         # read_file pagination cap
  max_line_length: 2000   # per-line cap in read_file's line-numbered view
```

- **`max_bytes`** — When a `terminal` command produces more than this many characters of combined stdout/stderr, Hermes keeps the first 40% and last 60% and inserts a `[OUTPUT TRUNCATED]` notice between them. Default `50000` (≈12-15K tokens across typical tokenisers).
- **`max_lines`** — Upper bound on the `limit` parameter of a single `read_file` call. Requests above this are clamped so a single read can't flood the context window. Default `2000`.
- **`max_line_length`** — Per-line cap applied when `read_file` emits the line-numbered view. Lines longer than this are truncated to this many chars followed by `... [truncated]`. Default `2000`.

Raise the limits on models with large context windows that can afford more raw output per call. Lower them for small-context models to keep tool results compact:

```yaml
# Large context model (200K+)
tool_output:
  max_bytes: 150000
  max_lines: 5000

# Small local model (16K context)
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## Global Toolset Disable

To suppress specific toolsets across the CLI and every gateway platform in one
place, list their names under `agent.disabled_toolsets`:

```yaml
agent:
  disabled_toolsets:
    - memory       # hide memory tools + MEMORY_GUIDANCE injection
    - web          # no web_search / web_extract anywhere
```

This applies **after** per-platform tool config (`platform_toolsets` written by
`hermes tools`), so a toolset listed here is always removed — even if a
platform's saved config still lists it. Use this when you want a single
switch for "turn X off everywhere" rather than editing 15+ platform rows in
the `hermes tools` UI.

Leaving the list empty, or omitting the key, is a no-op.

## Git Worktree Isolation

Enable isolated git worktrees for running multiple agents in parallel on the same repo:

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

When enabled, each CLI session creates a fresh worktree under `.worktrees/` with its own branch. Agents can edit files, commit, push, and create PRs without interfering with each other. Clean worktrees are removed on exit; dirty ones are kept for manual recovery.

By default the new worktree branches from the **freshly-fetched remote tip** (the current branch's upstream, otherwise the remote's default branch) so it starts current with the project rather than from the local clone's possibly-stale `HEAD`. This keeps a PR's diff scoped to the actual change instead of inheriting whatever the local clone was behind by. Set `worktree_sync: false` to branch from local `HEAD` instead — useful offline, or when you deliberately want the clone's exact current state as the base. If the remote can't be reached, it falls back to local `HEAD` automatically.

```yaml
worktree_sync: true    # Default — branch from the fetched remote tip
# worktree_sync: false # Branch from local HEAD (offline / pinned base)
```

You can also list gitignored files to copy into worktrees via `.worktreeinclude` in your repo root:

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## Context Compression

Hermes automatically compresses long conversations to stay within your model's context window. The compression summarizer is a separate LLM call — you can point it at any provider or endpoint.

All compression settings live in `config.yaml` (no environment variables).

### Full reference

```yaml
compression:
  enabled: true                                     # Toggle compression on/off
  threshold: 0.50                                   # Compress at this % of context limit
  target_ratio: 0.20                                # Fraction of threshold to preserve as recent tail
  protect_last_n: 20                                # Min recent messages to keep uncompressed
  protect_first_n: 3                                # Non-system head messages pinned across compactions (0 = pin nothing)
  hygiene_hard_message_limit: 5000                  # Gateway safety valve — see below

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: ""                                       # Empty = use main chat model. Override with e.g. "google/gemini-3-flash-preview" for cheaper/faster compression.
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info Legacy config migration
Older configs with `compression.summary_model`, `compression.summary_provider`, and `compression.summary_base_url` are automatically migrated to `auxiliary.compression.*` on first load (config version 17). No manual action needed.
:::

`hygiene_hard_message_limit` is a gateway-only **pre-compression safety valve**. It exists to break a death spiral: when API calls keep disconnecting on an oversized session, the gateway never receives token-usage data, so the token-based threshold can't fire, so the transcript keeps growing and disconnects get worse. This count-based floor fires on message count alone (always known, regardless of API failures) to force compression and recover the session. Default `5000` — far above any normal session, including large-context (1M+) models doing thousands of short turns, which compress on the token threshold long before this. Raise it further for unusual platforms, lower it to force more aggressive compression. Editing this value on a running gateway takes effect on the next message (see below).

`protect_first_n` controls how many **non-system** head messages are pinned across every compaction. Default `3` — the opening user/assistant exchange survives every summarizer pass so the original goal stays visible. On long-running rolling-compaction sessions where the opening turn is no longer relevant, set `protect_first_n: 0` to pin nothing but the system prompt + summary + tail. The system prompt itself is always preserved regardless of this setting.

:::tip Gateway hot-reload of compression and context length
As of recent releases, editing `model.context_length` or any `compression.*` key in `config.yaml` on a running gateway takes effect on the next message — no gateway restart, no `/reset`, no session rotation required. The cached-agent signature includes these keys, so the gateway transparently rebuilds the agent when it sees a change. API keys and tool/skill config still require the usual reload paths.
:::

### Common setups

**Default (auto-detect) — no configuration needed:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
Uses your main provider and main model. Override per-task (e.g. `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`) if you want compression on a cheaper model than your main chat model.

**Force a specific provider** (OAuth or API-key based):
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
Works with any provider: `nous`, `openrouter`, `codex`, `anthropic`, `main`, etc.

**Custom endpoint** (self-hosted, Ollama, zai, DeepSeek, etc.):
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
Points at a custom OpenAI-compatible endpoint. Uses `OPENAI_API_KEY` for auth.

### How the three knobs interact

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | Result |
|---------------------|---------------------|--------|
| `auto` (default) | not set | Auto-detect best available provider |
| `nous` / `openrouter` / etc. | not set | Force that provider, use its auth |
| any | set | Use the custom endpoint directly (provider ignored) |

:::warning Summary model context length requirement
The summary model **must** have a context window at least as large as your main agent model's. The compressor sends the full middle section of the conversation to the summary model — if that model's context window is smaller than the main model's, the summarization call will fail with a context length error. When this happens, the middle turns are **dropped without a summary**, losing conversation context silently. If you override the model, verify its context length meets or exceeds your main model's.
:::

## Context Engine

The context engine controls how conversations are managed when approaching the model's token limit. The built-in `compressor` engine uses lossy summarization (see [Context Compression](/developer-guide/context-compression-and-caching)). Plugin engines can replace it with alternative strategies.

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
```

To use a plugin engine (e.g., LCM for lossless context management):

```yaml
context:
  engine: "lcm"          # must match the plugin's name
```

Plugin engines are **never auto-activated** — you must explicitly set `context.engine` to the plugin name. Available engines can be browsed and selected via `hermes plugins` → Provider Plugins → Context Engine.

See [Memory Providers](/user-guide/features/memory-providers) for the analogous single-select system for memory plugins.

## Iteration Budget

When the agent is working on a complex task with many tool calls, it can burn through its iteration budget (default: 90 turns). Hermes does **not** inject mid-task pressure warnings — earlier builds warned the model at 70%/90% budget, which caused models to abandon complex tasks prematurely and was removed in April 2026.

Instead, when the budget is actually exhausted (90/90), Hermes injects one message asking the model to wrap up and allows a single **grace call** so it can deliver a final response. If that grace call still doesn't produce text, the agent is asked to summarise what it accomplished.

```yaml
agent:
  max_turns: 90                # Max iterations per conversation turn (default: 90)
  api_max_retries: 3           # Retries per provider before fallback engages (default: 3)
```

When the iteration budget is fully exhausted, the CLI shows a notification to the user: `⚠ Iteration budget reached (90/90) — response may be incomplete`.

`agent.api_max_retries` controls how many times Hermes retries a provider API call on transient errors (rate limits, connection drops, 5xx) **before** fallback-provider switching engages. The default is `3` — four attempts total. If you have [fallback providers](/user-guide/features/fallback-providers) configured and want to fail over faster, drop this to `0` so the first transient error on your primary immediately hands off to the fallback instead of churning retries against the flaky endpoint.

## Standing Goals (`/goal`)

When a standing goal is active, Hermes judges whether each assistant response satisfies it. If not, it feeds a continuation prompt back into the same session and keeps working until the goal is done, the turn budget is exhausted, or the user pauses/clears it. The turn budget is the real backstop — judge failures fail **open** (continue) so a flaky judge never wedges progress.

```yaml
goals:
  max_turns: 20   # Max continuation turns before Hermes auto-pauses the goal (default: 20)
```

`max_turns` caps how many continuation turns a goal can drive before Hermes auto-pauses it and asks the user to `/goal resume`. It protects against judge false negatives (goal actually done but judge says continue) and unbounded model spend on fuzzy or unachievable goals. See [Goals](/user-guide/features/goals) for the full feature.

### API Timeouts

Hermes has separate timeout layers for streaming, plus a stale detector for non-streaming calls. The stale detectors auto-adjust for local providers only when you leave them at their implicit defaults.

| Timeout | Default | Local providers | Config / env |
|---------|---------|----------------|--------------|
| Socket read timeout | 120s | Auto-raised to 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Stale stream detection | 180s | Auto-disabled | `HERMES_STREAM_STALE_TIMEOUT` |
| Stale non-stream detection | 300s | Auto-disabled when left implicit | `providers.<id>.stale_timeout_seconds` or `HERMES_API_CALL_STALE_TIMEOUT` |
| API call (non-streaming) | 1800s | Unchanged | `providers.<id>.request_timeout_seconds` / `timeout_seconds` or `HERMES_API_TIMEOUT` |

The **socket read timeout** controls how long httpx waits for the next chunk of data from the provider. Local LLMs can take minutes for prefill on large contexts before producing the first token, so Hermes raises this to 30 minutes when it detects a local endpoint. If you explicitly set `HERMES_STREAM_READ_TIMEOUT`, that value is always used regardless of endpoint detection.

The **stale stream detection** kills connections that receive SSE keep-alive pings but no actual content. This is disabled entirely for local providers since they don't send keep-alive pings during prefill.

The **stale non-stream detection** kills non-streaming calls that produce no response for too long. By default Hermes disables this on local endpoints to avoid false positives during long prefills. If you explicitly set `providers.<id>.stale_timeout_seconds`, `providers.<id>.models.<model>.stale_timeout_seconds`, or `HERMES_API_CALL_STALE_TIMEOUT`, that explicit value is honored even on local endpoints.

## Context Pressure Warnings

Separate from iteration budget pressure, context pressure tracks how close the conversation is to the **compaction threshold** — the point where context compression fires to summarize older messages. This helps both you and the agent understand when the conversation is getting long.

| Progress | Level | What happens |
|----------|-------|-------------|
| **≥ 60%** to threshold | Info | CLI shows a cyan progress bar; gateway sends an informational notice |
| **≥ 85%** to threshold | Warning | CLI shows a bold yellow bar; gateway warns compaction is imminent |

In the CLI, context pressure appears as a progress bar in the tool output feed:

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

On messaging platforms, a plain-text notification is sent:

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

If auto-compression is disabled, the warning tells you context may be truncated instead.

Context pressure is automatic — no configuration needed. It fires purely as a user-facing notification and does not modify the message stream or inject anything into the model's context.

## Credential Pool Strategies

When you have multiple API keys or OAuth tokens for the same provider, configure the rotation strategy:

```yaml
credential_pool_strategies:
  openrouter: round_robin    # cycle through keys evenly
  anthropic: least_used      # always pick the least-used key
```

Options: `fill_first` (default), `round_robin`, `least_used`, `random`. See [Credential Pools](/user-guide/features/credential-pools) for full documentation.

## Prompt caching

Hermes turns on cross-session prompt caching automatically when the active provider supports it — no user config needed.

For Claude on **native Anthropic**, **OpenRouter**, and **Nous Portal**, Hermes attaches `cache_control` breakpoints with the 1-hour TTL (`ttl: "1h"`) on the system prompt and skill blocks. The first send within a fresh hour pays full input rates; subsequent sends across any session within the same hour pull from the cache at the discounted cached-read rate. This means the system prompt, loaded skill content, and the early portion of any long-context include get reused across `hermes` sessions and across forked subagents for the first hour.

The Qwen Cloud (Alibaba DashScope) upstream caps cache TTL at 5 minutes, so Hermes uses the 5-minute breakpoint TTL there instead. Other Claude-via-third-party paths (AWS Bedrock, Azure Foundry) fall back to the provider's own caching defaults. xAI Grok uses a separate session-pinned conversation-id mechanism — see [xAI prompt caching](/integrations/providers#xai-grok--responses-api--prompt-caching).

No knob exists to disable this — caching is always-on and saves money even on single-turn conversations because the system prompt alone is a meaningful fraction of the input token count.

The one explicit knob is the cache TTL tier Hermes requests on Anthropic-style breakpoints:

```yaml
prompt_caching:
  cache_ttl: "5m"   # "5m" or "1h" (Anthropic-supported tiers); other values are ignored
```

`cache_ttl` selects the breakpoint TTL Hermes attaches for Claude via the native Anthropic API, OpenRouter, and Nous Portal. Only the two Anthropic-supported tiers (`"5m"`, `"1h"`) are honored — any other value is ignored. Providers with their own caps (e.g. Qwen Cloud, which maxes at 5 minutes) still clamp to what the upstream allows.

## Auxiliary Models

Hermes uses "auxiliary" models for side tasks like image analysis, web page summarization, browser screenshot analysis, session-title generation, and context compression. By default (`auxiliary.*.provider: "auto"`), Hermes routes every auxiliary task to your **main chat model** — the same provider/model you picked in `hermes model`. You don't need to configure anything to get started, but be aware that on expensive reasoning models (Opus, MiniMax M2.7, etc.) auxiliary tasks add meaningful cost. If you want cheap-and-fast side tasks regardless of your main model, set `auxiliary.<task>.provider` and `auxiliary.<task>.model` explicitly (for example, Gemini Flash on OpenRouter for vision and web extraction).

:::note Why "auto" uses your main model
Earlier builds split aggregator users (OpenRouter, Nous Portal) onto a cheap provider-side default. That was surprising — users who paid for an aggregator subscription would see a different model handling their auxiliary traffic. `auto` now uses the main model for everyone, and per-task overrides in `config.yaml` still win (see [Full auxiliary config reference](#full-auxiliary-config-reference) below).
:::

### Configuring auxiliary models interactively

Instead of hand-editing YAML, run `hermes model` and pick **"Configure auxiliary models"** from the menu. You'll get an interactive per-task picker:

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] tts_audio_tags       currently: auto / main model
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
```

Select a task, pick a provider (OAuth flows open a browser; API-key providers prompt), pick a model. The change persists to `auxiliary.<task>.*` in `config.yaml`. Same machinery as the main-model picker — no extra syntax to learn.

### Video Tutorial

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### The universal config pattern

Every model slot in Hermes — auxiliary tasks, compression, fallback — uses the same three knobs:

| Key | What it does | Default |
|-----|-------------|---------|
| `provider` | Which provider to use for auth and routing | `"auto"` |
| `model` | Which model to request | provider's default |
| `base_url` | Custom OpenAI-compatible endpoint (overrides provider) | not set |

When `base_url` is set, Hermes ignores the provider and calls that endpoint directly (using `api_key` or `OPENAI_API_KEY` for auth). When only `provider` is set, Hermes uses that provider's built-in auth and base URL.

Available providers for auxiliary tasks: `auto`, `main`, plus any provider in the [provider registry](/reference/environment-variables) — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `azure-foundry` — or any named custom provider from your `custom_providers` list (e.g. `provider: "beans"`).

:::tip MiniMax OAuth
`minimax-oauth` logs in via browser OAuth (no API key needed). Run `hermes model` and select **MiniMax (OAuth)** to authenticate. Auxiliary tasks use `MiniMax-M2.7-highspeed` automatically. See the [MiniMax OAuth guide](../guides/minimax-oauth.md).
:::

:::tip xAI Grok OAuth
`xai-oauth` logs in via browser OAuth for SuperGrok and X Premium+ subscribers (no API key needed). Run `hermes model` and select **xAI Grok OAuth (SuperGrok / Premium+)** to authenticate. The same OAuth token is reused for every direct-to-xAI surface (chat, auxiliary tasks, TTS, image gen, video gen, transcription). See the [xAI Grok OAuth guide](../guides/xai-grok-oauth.md), and if Hermes is on a remote host see [OAuth over SSH / Remote Hosts](../guides/oauth-over-ssh.md).
:::

:::warning `"main"` is for auxiliary tasks only
The `"main"` provider option means "use whatever provider my main agent uses" — it's only valid inside `auxiliary:`, `compression:`, and primary fallback entries (`fallback_providers:` or legacy `fallback_model:`). It is **not** a valid value for your top-level `model.provider` setting. If you use a custom OpenAI-compatible endpoint, set `provider: custom` in your `model:` section. See [AI Providers](/integrations/providers) for all main model provider options.
:::

### Full auxiliary config reference

```yaml
auxiliary:
  # Image analysis (vision_analyze tool + browser screenshots)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "codex", "main", etc.
    model: ""                  # e.g. "openai/gpt-4o", "google/gemini-2.5-flash"
    base_url: ""               # Custom OpenAI-compatible endpoint (overrides provider)
    api_key: ""                # API key for base_url (falls back to OPENAI_API_KEY)
    timeout: 120               # seconds — LLM API call timeout; vision payloads need generous timeout
    download_timeout: 30       # seconds — image HTTP download; increase for slow connections
    max_concurrency: 8         # max concurrent image encode/resize bursts across the process
                               # (default: host CPU core count, no ceiling) — bounds only the
                               # CPU-bound encode step so a video-frame fan-out can't saturate
                               # every core and starve the event loop; LLM calls stay fully
                               # concurrent. Minimum 1; values < 1 are ignored.

  # Web page summarization + browser page text extraction
  web_extract:
    provider: "auto"
    model: ""                  # e.g. "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # seconds (6min) — per-attempt LLM summarization

  # Dangerous command approval classifier
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # seconds

  # Gemini 3.1 TTS hidden audio-tag insertion
  tts_audio_tags:
    provider: "auto"
    model: ""                  # empty = main chat model
    base_url: ""
    api_key: ""
    timeout: 30

  # Context compression timeout (separate from compression.* config)
  compression:
    timeout: 120               # seconds — compression summarizes long conversations, needs more time
    # fallback_chain:           # Optional — providers to try on rate-limit / connectivity failure
    #   - provider: nous
    #     model: deepseek/deepseek-chat
    #   - provider: openrouter
    #     model: google/gemini-2.5-flash
    #     base_url: ""
    #     api_key: ""

  # Auto-generated session titles. Empty language follows the conversation;
  # set e.g. "English" or "Japanese" to pin titles to one language.
  title_generation:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    language: ""

  # Skills hub — skill matching and search
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP tool dispatch
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # Kanban triage specifier — `hermes kanban specify <id>` (or the
  # dashboard's ✨ Specify button on Triage-column cards) uses this
  # slot to expand a one-liner into a concrete spec and promote the
  # task to `todo`. Cheap fast models work well here; spec expansion
  # is short and doesn't need reasoning depth.
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
Each auxiliary task has a configurable `timeout` (in seconds). Defaults: vision 120s, web_extract 360s, approval 30s, compression 120s. Increase these if you use slow local models for auxiliary tasks. Vision also has a separate `download_timeout` (default 30s) for the HTTP image download — increase this for slow connections or self-hosted image servers.
:::

:::info
Context compression has its own `compression:` block for thresholds and an `auxiliary.compression:` block for model/provider settings — see [Context Compression](#context-compression) above. The primary fallback chain uses a top-level `fallback_providers:` list — see [Fallback Providers](/integrations/providers#fallback-providers). All three follow the same provider/model/base_url pattern.
:::

### Per-task fallback chain for auxiliary tasks

Each auxiliary task can optionally define a `fallback_chain` — a list of provider/model entries that Hermes tries when the primary auxiliary provider fails due to rate limits, connectivity issues, or payment restrictions:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openai/gpt-4o-mini
    fallback_chain:
      - provider: nous
        model: deepseek/deepseek-chat
      - provider: openrouter
        model: google/gemini-2.5-flash
```

When the primary auxiliary provider (`openrouter` / `openai/gpt-4o-mini`) returns a rate-limit, connection timeout, or payment-required error, Hermes walks the `fallback_chain` in order. It skips entries whose provider matches the already-failed provider, and tries each remaining entry until one succeeds or the chain is exhausted. If all fallbacks fail, Hermes falls back to the main agent model as a final safety net.

Each entry supports the same three knobs as any auxiliary task config:

| Key | Description |
|-----|-------------|
| `provider` | Provider name (`nous`, `openrouter`, `anthropic`, `gemini`, `main`, etc.) |
| `model` | Model name for that provider |
| `base_url` | (Optional) Custom OpenAI-compatible endpoint |

`fallback_chain` is available on any auxiliary task — `compression`, `vision`, `web_extract`, `approval`, `skills_hub`, `mcp`, etc.

### OpenRouter routing & Pareto Code for auxiliary tasks

When an auxiliary task resolves to OpenRouter (either explicitly or via `provider: "main"` while your main agent is on OpenRouter), the main agent's `provider_routing` and `openrouter.min_coding_score` settings **do not propagate** — by design, each auxiliary task is independent. To set OpenRouter provider preferences or use the [Pareto Code router](/integrations/providers#openrouter-pareto-code-router) for a specific aux task, set them per-task via `extra_body`:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # use the Pareto Code router for this task
    extra_body:
      provider:                            # OpenRouter provider routing prefs
        order: [anthropic, google]         # try these providers in order
        sort: throughput                   # or "price" | "latency"
        # only: [anthropic]                # restrict to a specific provider
        # ignore: [deepinfra]              # exclude specific providers
      plugins:                             # OpenRouter Pareto Code router knob
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0; higher = stronger coders
```

The shape mirrors what OpenRouter accepts in the chat completions request body. Hermes forwards the entire `extra_body` verbatim, so any other OpenRouter request-body field documented at [openrouter.ai/docs](https://openrouter.ai/docs) works the same way.

### Changing the Vision Model

To use GPT-4o instead of Gemini Flash for image analysis:

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

Or via environment variable (in `~/.hermes/.env`):

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### Provider Options

These options apply to **auxiliary task configs** (`auxiliary:`, `compression:`) and primary fallback entries (`fallback_providers:` or legacy `fallback_model:`), not to your main `model.provider` setting.

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | Best available (default). Vision tries OpenRouter → Nous → Codex. | — |
| `"openrouter"` | Force OpenRouter — routes to any model (Gemini, GPT-4o, Claude, etc.) | `OPENROUTER_API_KEY` |
| `"nous"` | Force Nous Portal | `hermes auth` |
| `"codex"` | Force Codex OAuth (ChatGPT account). Supports vision (gpt-5.3-codex). | `hermes model` → Codex |
| `"minimax-oauth"` | Force MiniMax OAuth (browser login, no API key). Uses MiniMax-M2.7-highspeed for auxiliary tasks. | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | Force xAI Grok OAuth (browser login for SuperGrok or X Premium+ subscribers, no API key). Same OAuth token covers chat, TTS, image, video, and transcription. | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | Use your active custom/main endpoint. This can come from `OPENAI_BASE_URL` + `OPENAI_API_KEY` or from a custom endpoint saved via `hermes model` / `config.yaml`. Works with OpenAI, local models, or any OpenAI-compatible API. **Auxiliary tasks only — not valid for `model.provider`.** | Custom endpoint credentials + base URL |

Direct API-key providers from the main provider catalog also work here when you want side tasks to bypass your default router. `gmi` is valid once `GMI_API_KEY` is configured:

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

For GMI auxiliary routing, use the exact model ID returned by GMI's `/v1/models` endpoint.

### Common Setups

**Using a direct custom endpoint** (clearer than `provider: "main"` for local/self-hosted APIs):
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` takes precedence over `provider`, so this is the most explicit way to route an auxiliary task to a specific endpoint. For direct endpoint overrides, Hermes uses the configured `api_key` or falls back to `OPENAI_API_KEY`; it does not reuse `OPENROUTER_API_KEY` for that custom endpoint.

**Using OpenAI API key for vision:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**Using OpenRouter for vision** (route to any model):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Using Codex OAuth** (ChatGPT Pro/Plus account — no API key needed):
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    # model defaults to gpt-5.3-codex (supports vision)
```

**Using MiniMax OAuth** (browser login, no API key needed):
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
Run `hermes model` and select **MiniMax (OAuth)** to log in and set this automatically. For the China region, the base URL will be `https://api.minimaxi.com/anthropic`. See the [MiniMax OAuth guide](../guides/minimax-oauth.md) for the full walkthrough.

**Using a local/self-hosted model:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your active custom endpoint
    model: "my-local-model"
```

`provider: "main"` uses whatever provider Hermes uses for normal chat — whether that's a named custom provider (e.g. `beans`), a built-in provider like `openrouter`, or a legacy `OPENAI_BASE_URL` endpoint.

:::tip
If you use Codex OAuth as your main model provider, vision works automatically — no extra configuration needed. Codex is included in the auto-detection chain for vision.
:::

:::warning
**Vision requires a multimodal model.** If you set `provider: "main"`, make sure your endpoint supports multimodal/vision — otherwise image analysis will fail.
:::

### Environment Variables (legacy)

Auxiliary models can also be configured via environment variables. However, `config.yaml` is the preferred method — it's easier to manage and supports all options including `base_url` and `api_key`.

| Setting | Environment Variable |
|---------|---------------------|
| Vision provider | `AUXILIARY_VISION_PROVIDER` |
| Vision model | `AUXILIARY_VISION_MODEL` |
| Vision endpoint | `AUXILIARY_VISION_BASE_URL` |
| Vision API key | `AUXILIARY_VISION_API_KEY` |
| Web extract provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web extract model | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web extract endpoint | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web extract API key | `AUXILIARY_WEB_EXTRACT_API_KEY` |

Compression and fallback model settings are config.yaml-only.

:::tip
Run `hermes config` to see your current auxiliary model settings. Overrides only show up when they differ from the defaults.
:::

## Reasoning Effort

Control how much "thinking" the model does before responding:

```yaml
agent:
  reasoning_effort: ""   # empty = medium (default). Options: none, minimal, low, medium, high, xhigh (max)
```

When unset (default), reasoning effort defaults to "medium" — a balanced level that works well for most tasks. Setting a value overrides it — higher reasoning effort gives better results on complex tasks at the cost of more tokens and latency.

:::note Adaptive-thinking models (Claude 4.6+, Fable/Mythos-class) over OpenRouter
These models use *adaptive* thinking and don't accept the usual `reasoning.effort`
field — OpenRouter ignores it for them. Hermes transparently routes your
`reasoning_effort` to OpenRouter's `verbosity` parameter instead (which maps to
Anthropic's `output_config.effort`), so the same `low`/`medium`/`high`/`xhigh`
knob keeps working — no extra configuration needed. `none` (or unset) leaves the
model on its own adaptive default. (`max` is accepted on the wire but is not a
selectable `reasoning_effort` value; `xhigh` is the configurable ceiling.) The
native Anthropic provider already controls effort directly and is unaffected.
:::

You can also change the reasoning effort at runtime with the `/reasoning` command:

```
/reasoning           # Show current effort level and display state
/reasoning high      # Set reasoning effort to high
/reasoning none      # Disable reasoning
/reasoning show      # Show model thinking above each response
/reasoning hide      # Hide model thinking
```

## Tool-Use Enforcement

Some models occasionally describe intended actions as text instead of making tool calls ("I would run the tests..." instead of actually calling the terminal). Tool-use enforcement injects system prompt guidance that steers the model back to actually calling tools.

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Enabled for models matching: `gpt`, `codex`, `gemini`, `gemma`, `grok`. Disabled for all others (Claude, DeepSeek, Qwen, etc.). |
| `true` | Always enabled, regardless of model. Useful if you notice your current model describing actions instead of performing them. |
| `false` | Always disabled, regardless of model. |
| `["gpt", "codex", "qwen", "llama"]` | Enabled only when the model name contains one of the listed substrings (case-insensitive). |

### What it injects

When enabled, three layers of guidance may be added to the system prompt:

1. **General tool-use enforcement** (all matched models) — instructs the model to make tool calls immediately instead of describing intentions, keep working until the task is complete, and never end a turn with a promise of future action.

2. **OpenAI execution discipline** (GPT and Codex models only) — additional guidance addressing GPT-specific failure modes: abandoning work on partial results, skipping prerequisite lookups, hallucinating instead of using tools, and declaring "done" without verification.

3. **Google operational guidance** (Gemini and Gemma models only) — conciseness, absolute paths, parallel tool calls, and verify-before-edit patterns.

These are transparent to the user and only affect the system prompt. Models that already use tools reliably (like Claude) don't need this guidance, which is why `"auto"` excludes them.

### When to turn it on

If you're using a model not in the default auto list and notice it frequently describes what it *would* do instead of doing it, set `tool_use_enforcement: true` or add the model substring to the list:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## Tool-Loop Guardrails

Hermes detects when the agent is stuck in an unproductive tool-calling loop — the same tool call failing repeatedly, the same tool failing over and over, or an idempotent call returning the same result with no progress. By default it injects a **warning** into the tool result so the model self-corrects; it does not hard-stop, since a person watching the CLI/TUI can intervene.

For unattended gateway / server deployments, enable hard stops so a stuck agent is circuit-broken instead of burning the iteration budget:

```yaml
tool_loop_guardrails:
  warnings_enabled: true       # inject warnings into tool results (default: true)
  hard_stop_enabled: false     # also BLOCK the call past the hard-stop threshold (default: false)
  warn_after:
    exact_failure: 2           # identical failing call repeated N times
    same_tool_failure: 3       # same tool failing N times (different args)
    idempotent_no_progress: 2  # same result, no progress, N times
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
```

`hard_stop_enabled` defaults to `false` because interactive sessions have a human in the loop. In unattended deployments (gateway, cron, kanban workers) set it to `true` so repeated failures are blocked rather than only warned. See also [Docker / unattended deployments](docker.md).

## TTS Configuration

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # Global speed multiplier (fallback for all providers)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Speed multiplier (converted to rate percentage, e.g. 1.5 → +50%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # Speed multiplier (clamped to 0.25–4.0 by the API)
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
  minimax:
    speed: 1.0                  # Speech speed multiplier
    # base_url: ""              # Optional: override for OpenAI-compatible TTS endpoints
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, etc.
    audio_tags: false           # Hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # xAI TTS voice
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 bitrate
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

This controls both the `text_to_speech` tool and spoken replies in voice mode (`/voice tts` in the CLI or messaging gateway).

**Speed fallback hierarchy:** provider-specific speed (e.g. `tts.edge.speed`) → global `tts.speed` → `1.0` default. Set the global `tts.speed` to apply a uniform speed across all providers, or override per-provider for fine-grained control.

## Display Settings

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # Enable /verbose slash command in messaging gateway
  platforms: {}           # Per-platform display overrides (see below)
  tool_progress_overrides: {}  # DEPRECATED — use display.platforms instead
  interim_assistant_messages: true  # Gateway: send natural mid-turn assistant updates as separate messages
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: "kawaii"  # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  show_reasoning: false   # Show model reasoning/thinking above each response (toggle with /reasoning show|hide)
  streaming: false        # Stream tokens to terminal as they arrive (real-time output)
  show_cost: false        # Show estimated $ cost in the CLI status bar
  timestamps: false       # When true, prefixes user and assistant labels with [HH:MM] timestamps in the CLI / TUI transcript
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  runtime_footer:         # Gateway: append a runtime-context footer to final replies
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # Append an advisory footer when write_file/patch calls failed this turn
  credits_notices: true   # Nous credits status-bar notices (usage bands, grant-spent, depleted). false = silence them; /usage still works
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### File-mutation verifier

When `display.file_mutation_verifier` is `true` (default), Hermes appends a one-line advisory to the assistant's final response whenever a `write_file` or `patch` call failed during the turn and was never superseded by a successful write to the same path. This catches the "batch of parallel patches, half silently fail, model summarises success" class of over-claim without requiring you to manually run `git status` after every edit.

Example footer:

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

Set `file_mutation_verifier: false` (or `HERMES_FILE_MUTATION_VERIFIER=0`) to suppress the footer. The verifier only fires when real failures are outstanding at turn end — a model that retries a failed patch and succeeds within the same turn will not trigger it for that file.

### UI language for static messages

The `display.language` setting translates a small set of static user-facing messages — the CLI approval prompt, a handful of gateway slash-command replies (e.g. restart-drain notices, "approval expired", "goal cleared"). It does **not** translate agent responses, log lines, tool output, error tracebacks, or slash-command descriptions — those stay in English. If you want the agent itself to reply in another language, just tell it in your prompt or system message.

Supported values: `en` (default), `zh` (Simplified Chinese), `zh-hant` (Traditional Chinese), `ja` (Japanese), `de` (German), `es` (Spanish), `fr` (French), `tr` (Turkish), `uk` (Ukrainian), `af` (Afrikaans), `ko` (Korean), `it` (Italian), `ga` (Irish), `pt` (Portuguese), `ru` (Russian), `hu` (Hungarian). Unknown values fall back to English.

You can also set this per-session with the `HERMES_LANGUAGE` env var, which overrides the config value.

```yaml
display:
  language: zh   # CLI approval prompts appear in Chinese
```

| Mode | What you see |
|------|-------------|
| `off` | Silent — just the final response |
| `new` | Tool indicator only when the tool changes |
| `all` | Every tool call with a short preview (default) |
| `verbose` | Full args, results, and debug logs |

In the CLI, cycle through these modes with `/verbose`. To use `/verbose` in messaging platforms (Telegram, Discord, Slack, etc.), set `tool_progress_command: true` in the `display` section above. The command will then cycle the mode and save to config.

Tool progress requires a gateway adapter that can display progress updates safely. Platforms without message editing support, including Signal, suppress tool-progress bubbles even if `/verbose` saves a non-`off` mode.

### Runtime-metadata footer (gateway only)

When `display.runtime_footer.enabled: true`, Hermes appends a small runtime-context footer to the **final** message of each gateway turn. The current footer can show the model, context-window percentage, and current working directory. Off by default; opt in per-gateway if your team wants every reply to include this provenance.

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # supported fields: model, context_pct, cwd
```

The `/footer` slash command toggles this at runtime in any session.

Example footer appended to a Telegram/Discord/Slack reply:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

Only the **final** message of a turn gets the footer; interim updates stay clean.

### Per-platform progress overrides

Different platforms have different verbosity needs. Use `display.platforms` to set per-platform modes:

```yaml
display:
  tool_progress: all          # global default
  platforms:
    signal:
      tool_progress: 'off'    # Signal cannot currently display tool-progress bubbles
    telegram:
      tool_progress: verbose  # detailed progress on Telegram
    slack:
      tool_progress: 'off'    # quiet in shared Slack workspace
```

Platforms without an override fall back to the global `tool_progress` value. Valid platform keys: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. The legacy `display.tool_progress_overrides` key still loads for backward compatibility but is deprecated and migrated into `display.platforms` on first load.

Signal is listed as a valid platform key because the setting can be saved per platform, but the current Signal adapter cannot edit sent messages and does not render tool-progress bubbles. Keep Signal `tool_progress` set to `off`; use the CLI or an editing-capable messaging platform if you need to watch each tool call live.

`interim_assistant_messages` is gateway-only. When enabled, Hermes sends completed mid-turn assistant updates as separate chat messages. This is independent from `tool_progress` and does not require gateway streaming.

## Privacy

```yaml
privacy:
  redact_pii: false  # Strip PII from LLM context (gateway only)
```

When `redact_pii` is `true`, the gateway redacts personally identifiable information from the system prompt before sending it to the LLM on supported platforms:

| Field | Treatment |
|-------|-----------|
| Phone numbers (user ID on WhatsApp/Signal) | Hashed to `user_<12-char-sha256>` |
| User IDs | Hashed to `user_<12-char-sha256>` |
| Chat IDs | Numeric portion hashed, platform prefix preserved (`telegram:<hash>`) |
| Home channel IDs | Numeric portion hashed |
| User names / usernames | **Not affected** (user-chosen, publicly visible) |

**Platform support:** Redaction applies to WhatsApp, Signal, and Telegram. Discord and Slack are excluded because their mention systems (`<@user_id>`) require the real ID in the LLM context.

Hashes are deterministic — the same user always maps to the same hash, so the model can still distinguish between users in group chats. Routing and delivery use the original values internally.

## Speech-to-Text (STT)

```yaml
stt:
  enabled: true                # Auto-transcribe inbound voice messages (default: true)
  echo_transcripts: true       # Post raw transcripts back to the chat as 🎙️ "..." (default: true)
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # Legacy fallback key still respected
```

Set `stt.echo_transcripts: false` when the gateway should transcribe voice notes for the agent but must not post the raw transcript back to the chat (for example, customer-facing WhatsApp bots).

Provider behavior:

- `local` uses `faster-whisper` running on your machine. Install it separately with `pip install faster-whisper`.
- `groq` uses Groq's Whisper-compatible endpoint and reads `GROQ_API_KEY`.
- `openai` uses the OpenAI speech API and reads `VOICE_TOOLS_OPENAI_KEY`.

If the requested provider is unavailable, Hermes falls back automatically in this order: `local` → `groq` → `openai`.

Groq and OpenAI model overrides are environment-driven:

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## Voice Mode (CLI)

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  beep_enabled: true            # Play record start/stop beeps in CLI voice mode
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

Use `/voice on` in the CLI to enable microphone mode, `record_key` to start/stop recording, and `/voice tts` to toggle spoken replies. See [Voice Mode](/user-guide/features/voice-mode) for end-to-end setup and platform-specific behavior.

## Streaming

Stream tokens to the terminal or messaging platforms as they arrive, instead of waiting for the full response.

### CLI Streaming

```yaml
display:
  streaming: true         # Stream tokens to terminal in real-time
  show_reasoning: true    # Also stream reasoning/thinking tokens (optional)
```

When enabled, responses appear token-by-token inside a streaming box. Tool calls are still captured silently. If the provider doesn't support streaming, it falls back to the normal display automatically.

### Gateway Streaming (Telegram, Discord, Slack)

```yaml
streaming:
  enabled: true           # Enable progressive message editing
  transport: edit         # "edit" (progressive message editing) or "off"
  edit_interval: 0.3      # Seconds between message edits
  buffer_threshold: 40    # Characters before forcing an edit flush
  cursor: " ▉"            # Cursor shown during streaming
  fresh_final_after_seconds: 0    # Opt in to fresh final (Telegram) when preview is this old
```

When enabled, the bot sends a message on the first token, then progressively edits it as more tokens arrive. Platforms that don't support message editing (Signal, Email, Home Assistant) are auto-detected on the first attempt — streaming is gracefully disabled for that session with no flood of messages.

For separate natural mid-turn assistant updates without progressive token editing, set `display.interim_assistant_messages: true`.

**Overflow handling:** If the streamed text exceeds the platform's message length limit (~4096 chars), the current message is finalized and a new one starts automatically.

**Fresh final (Telegram):** Telegram's `editMessageText` preserves the original message timestamp, so a long-running streamed reply would keep the first-token timestamp even after completion. Set `fresh_final_after_seconds > 0` to opt in to delivering old previews as brand-new final messages with best-effort preview deletion. The default is `0`, which always finalizes streamed replies in place and avoids the brief duplicate-message/delete sequence on clients that show both operations.

:::note Per-platform streaming defaults
The master `streaming.enabled` switch is `false` by default — nothing streams until you flip it. Once enabled, streaming is decided **per platform**: Telegram ships with `display.platforms.telegram.streaming: true` (streams) and Discord with `display.platforms.discord.streaming: false` (does not). So after enabling streaming, Telegram streams out of the box and Discord stays on whole-message replies until you change its toggle. You can adjust these per-platform switches from the dashboard's **Channels** toggles or directly in `~/.hermes/config.yaml`.
:::

## Group Chat Session Isolation

Limit how many chat sessions can actively be open across CLI, TUI/dashboard,
and messaging gateway:

```yaml
max_concurrent_sessions: null  # null/0 = unlimited; positive integer = active session cap
```

When the cap is reached, Hermes returns a direct limit message for new sessions.
Existing active sessions keep their normal behavior.

The canonical key is top-level `max_concurrent_sessions`. Hermes also accepts
`gateway.max_concurrent_sessions` as a fallback, but the top-level key wins when
both are set.

The cap is enforced with a local runtime lease file and is best-effort: Hermes
fails open if the registry cannot be read or locked so users are not stranded.
It is intended for a single host/profile runtime, not a shared `$HERMES_HOME`
mounted across multiple machines.

Control whether shared chats keep one conversation per room or one conversation per participant:

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true` is the default and recommended setting. In Discord channels, Telegram groups, Slack channels, and similar shared contexts, each sender gets their own session when the platform provides a user ID.
- `false` reverts to the old shared-room behavior. That can be useful if you explicitly want Hermes to treat a channel like one collaborative conversation, but it also means users share context, token costs, and interrupt state.
- Direct messages are unaffected. Hermes still keys DMs by chat/DM ID as usual.
- Threads stay isolated from their parent channel either way; with `true`, each participant also gets their own session inside the thread.

For the behavior details and examples, see [Sessions](/user-guide/sessions) and the [Discord guide](/user-guide/messaging/discord).

## Unauthorized DM Behavior

Control what Hermes does when an unknown user sends a direct message:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` is the default for chat-style DM platforms. Hermes denies access, but replies with a one-time pairing code in DMs.
- `ignore` silently drops unauthorized DMs.
- Email defaults to `ignore` unless `platforms.email.unauthorized_dm_behavior: pair` is set, because inboxes can contain unrelated unread mail.
- Platform sections override the global default, so you can keep pairing enabled broadly while making one platform quieter.

## Quick Commands

Define custom commands that either run shell commands without invoking the LLM, or alias one slash command to another. Exec quick commands are zero-token and useful from messaging platforms (Telegram, Discord, etc.) for quick server checks or utility scripts.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

Usage: type `/status`, `/disk`, `/update`, `/gpu`, or `/restart` in the CLI or any messaging platform. `exec` commands run locally on the host and return the output directly — no LLM call, no tokens consumed. `alias` commands rewrite to the configured slash command target.

- **30-second timeout** — long-running commands are killed with an error message
- **Priority** — quick commands are checked before skill commands, so you can override skill names
- **Autocomplete** — quick commands are resolved at dispatch time and are not shown in the built-in slash-command autocomplete tables
- **Type** — supported types are `exec` and `alias`; other types show an error
- **Works everywhere** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant

String-only prompt shortcuts are not valid quick commands. For reusable prompt workflows, create a skill or alias to an existing slash command.

## Human Delay

Simulate human-like response pacing in messaging platforms:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # Minimum delay (custom mode)
  max_ms: 2500                 # Maximum delay (custom mode)
```

## Code Execution

Configure the `execute_code` tool:

```yaml
code_execution:
  mode: project                # project (default) | strict
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

**`mode`** controls the working directory and Python interpreter for scripts:

- **`project`** (default) — scripts run in the session's working directory with the active virtualenv/conda env's python. Project deps (`pandas`, `torch`, project packages) and relative paths (`.env`, `./data.csv`) resolve naturally, matching what `terminal()` sees.
- **`strict`** — scripts run in a temp staging directory with `sys.executable` (Hermes's own python). Maximum reproducibility, but project deps and relative paths won't resolve.

Environment scrubbing (strips `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL`, `*_PASSWD`, `*_AUTH`) and the tool whitelist apply identically in both modes — switching mode does not change the security posture.

## Web Search Backends

The `web_search` and `web_extract` tools support five backend providers. Configure the backend in `config.yaml` or via `hermes tools`:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # Or use per-capability keys to mix providers (e.g. free search + paid extract):
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| Backend | Env Var | Search | Extract |
|---------|---------|--------|---------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ |

**Backend selection:** If `web.backend` is not set, the backend is auto-detected from available API keys. If only `SEARXNG_URL` is set, SearXNG is used. If only `EXA_API_KEY` is set, Exa is used. If only `TAVILY_API_KEY` is set, Tavily is used. If only `PARALLEL_API_KEY` is set, Parallel is used. Otherwise Firecrawl is the default.

**SearXNG** is a free, self-hosted, privacy-respecting metasearch engine that queries 70+ search engines. No API key needed — just set `SEARXNG_URL` to your instance (e.g., `http://localhost:8080`). SearXNG is search-only; `web_extract` requires a separate extract provider (set `web.extract_backend`). See the [Web Search setup guide](/user-guide/features/web-search) for Docker setup instructions.

**Self-hosted Firecrawl:** Set `FIRECRAWL_API_URL` to point at your own instance. When a custom URL is set, the API key becomes optional (set `USE_DB_AUTHENTICATION=*** on the server to disable auth).

**Parallel search modes:** Set `PARALLEL_SEARCH_MODE` to control search behavior — `fast`, `one-shot`, or `agentic` (default: `agentic`).

**Exa:** Set `EXA_API_KEY` in `~/.hermes/.env`. Supports `category` filtering (`company`, `research paper`, `news`, `people`, `personal site`, `pdf`) and domain/date filters.

## Browser

Configure browser automation behavior:

```yaml
browser:
  inactivity_timeout: 120        # Seconds before auto-closing idle sessions
  command_timeout: 30             # Timeout in seconds for browser commands (screenshot, navigate, etc.)
  record_sessions: false         # Auto-record browser sessions as WebM videos to ~/.hermes/browser_recordings/
  # Optional CDP override — when set, Hermes attaches directly to your own
  # Chromium-family browser (via /browser connect) rather than starting a headless browser.
  cdp_url: ""
  # Dialog supervisor — controls how native JS dialogs (alert / confirm / prompt)
  # are handled when a CDP backend is attached (Browserbase, local Chromium-family
  # browser via /browser connect). Ignored on Camofox and default local agent-browser mode.
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # Safety auto-dismiss under must_respond (seconds)
  camofox:
    managed_persistence: false   # When true, Camofox sessions persist cookies/logins across restarts
    user_id: ""                  # Optional externally managed Camofox userId
    session_key: ""              # Optional session key sent when Hermes creates a tab
    adopt_existing_tab: false    # Reuse an existing tab for this identity before creating one
```

**Dialog policies:**

- `must_respond` (default) — capture the dialog, surface it in `browser_snapshot.pending_dialogs`, and wait for the agent to call `browser_dialog(action=...)`. After `dialog_timeout_s` seconds with no response, the dialog is auto-dismissed to prevent the page's JS thread from stalling forever.
- `auto_dismiss` — capture, dismiss immediately. The agent still sees the dialog record in `browser_snapshot.recent_dialogs` with `closed_by="auto_policy"` after the fact.
- `auto_accept` — capture, accept immediately. Useful for pages with aggressive `beforeunload` prompts.

See the [browser feature page](./features/browser.md#browser_dialog) for the full dialog workflow.

The browser toolset supports multiple providers. See the [Browser feature page](/user-guide/features/browser) for details on Browserbase, Browser Use, and local Chromium-family CDP setup.

## Timezone

Override the server-local timezone with an IANA timezone string. Affects timestamps in logs, cron scheduling, and system prompt time injection.

```yaml
timezone: "America/New_York"   # IANA timezone (default: "" = server-local time)
```

Supported values: any IANA timezone identifier (e.g. `America/New_York`, `Europe/London`, `Asia/Kolkata`, `UTC`). Leave empty or omit for server-local time.

## Discord

Configure Discord-specific behavior for the messaging gateway:

```yaml
discord:
  require_mention: true          # Require @mention to respond in server channels
  free_response_channels: ""     # Comma-separated channel IDs where bot responds without @mention
  auto_thread: true              # Auto-create threads on @mention in channels
```

- `require_mention` — when `true` (default), the bot only responds in server channels when mentioned with `@BotName`. DMs always work without mention.
- `free_response_channels` — comma-separated list of channel IDs where the bot responds to every message without requiring a mention.
- `auto_thread` — when `true` (default), mentions in channels automatically create a thread for the conversation, keeping channels clean (similar to Slack threading).

## Security

Pre-execution security scanning and secret redaction:

```yaml
security:
  redact_secrets: true           # Redact API key patterns in tool output and logs (on by default)
  tirith_enabled: true           # Enable Tirith security scanning for terminal commands
  tirith_path: "tirith"          # Path to tirith binary (default: "tirith" in $PATH)
  tirith_timeout: 5              # Seconds to wait for tirith scan before timing out
  tirith_fail_open: true         # Allow command execution if tirith is unavailable
  website_blocklist:             # See Website Blocklist section below
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — when `true`, automatically detects and redacts patterns that look like API keys, tokens, and passwords in tool output before it enters the conversation context and logs. **On by default**. Set to `false` explicitly only when you need raw credential-like strings for debugging or redactor development.
- `tirith_enabled` — when `true`, terminal commands are scanned by [Tirith](https://github.com/sheeki03/tirith) before execution to detect potentially dangerous operations.
- `tirith_path` — path to the tirith binary. Set this if tirith is installed in a non-standard location.
- `tirith_timeout` — maximum seconds to wait for a tirith scan. Commands proceed if the scan times out.
- `tirith_fail_open` — when `true` (default), commands are allowed to execute if tirith is unavailable or fails. Set to `false` to block commands when tirith cannot verify them.

## Website Blocklist

Block specific domains from being accessed by the agent's web and browser tools:

```yaml
security:
  website_blocklist:
    enabled: false               # Enable URL blocking (default: false)
    domains:                     # List of blocked domain patterns
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # Load additional rules from external files
      - "/etc/hermes/blocked-sites.txt"
```

When enabled, any URL matching a blocked domain pattern is rejected before the web or browser tool executes. This applies to `web_search`, `web_extract`, `browser_navigate`, and any tool that accesses URLs.

Domain rules support:
- Exact domains: `admin.example.com`
- Wildcard subdomains: `*.internal.company.com` (blocks all subdomains)
- TLD wildcards: `*.local`

Shared files contain one domain rule per line (blank lines and `#` comments are ignored). Missing or unreadable files log a warning but don't disable other web tools.

The policy is cached for 30 seconds, so config changes take effect quickly without restart.

## Smart Approvals

Control how Hermes handles potentially dangerous commands:

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| Mode | Behavior |
|------|----------|
| `manual` (default) | Prompt the user before executing any flagged command. In the CLI, shows an interactive approval dialog. In messaging, queues a pending approval request. |
| `smart` | Use an auxiliary LLM to assess whether a flagged command is actually dangerous. Low-risk commands are auto-approved with session-level persistence. Genuinely risky commands are escalated to the user. |
| `off` | Skip all approval checks. Equivalent to `HERMES_YOLO_MODE=true`. **Use with caution.** |

Smart mode is particularly useful for reducing approval fatigue — it lets the agent work more autonomously on safe operations while still catching genuinely destructive commands.

:::warning
Setting `approvals.mode: off` disables all safety checks for terminal commands. Only use this in trusted, sandboxed environments.
:::

### Deny rules

`approvals.deny` is a list of glob patterns that block matching terminal commands unconditionally — even under `--yolo`, `/yolo`, or `mode: off`. It's the user-editable counterpart to the built-in hardline blocklist:

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

Patterns are case-insensitive fnmatch globs and must be quoted in YAML (a bare leading `*` is a parse error). See [Security — User-Defined Deny Rules](/user-guide/security#user-defined-deny-rules-approvalsdeny) for details.

## Checkpoints

Automatic filesystem snapshots before destructive file operations. See the [Checkpoints & Rollback](/user-guide/checkpoints-and-rollback) for details.

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes chat --checkpoints). Default: false (opt-in).
  max_snapshots: 20              # Max checkpoints to keep per directory (default: 20)
```


## Delegation

Configure subagent behavior for the delegate tool:

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  # api_mode: ""                            # Wire protocol for base_url: "chat_completions", "codex_responses", or "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix → anthropic_messages). Set explicitly for non-standard endpoints the heuristic can't detect.
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
```

**Subagent provider:model override:** By default, subagents inherit the parent agent's provider and model. Set `delegation.provider` and `delegation.model` to route all subagents to a different provider:model pair — e.g., use a cheap/fast model for narrowly-scoped subtasks while your primary agent runs an expensive reasoning model. A `delegate_task` call can also pass `model` and `provider` directly, either at the top level or on individual batch task objects, to override this default for that child only.

**Direct endpoint override:** If you want the obvious custom-endpoint path, set `delegation.base_url`, `delegation.api_key`, and `delegation.model`. That sends subagents directly to that OpenAI-compatible endpoint and takes precedence over `delegation.provider`. If `delegation.api_key` is omitted, Hermes falls back to `OPENAI_API_KEY` only.

**Wire protocol (`api_mode`):** Hermes auto-detects the wire protocol from `delegation.base_url` (e.g. paths ending in `/anthropic` → `anthropic_messages`; Codex / native Anthropic / Kimi-coding hostnames keep their existing detection). For endpoints the heuristic can't classify — for example Azure AI Foundry, MiniMax, Zhipu GLM, or LiteLLM proxies fronting an Anthropic-shaped backend — set `delegation.api_mode` explicitly to one of `chat_completions`, `codex_responses`, or `anthropic_messages`. Leave it empty (the default) to keep auto-detection.

The delegation provider uses the same credential resolution as CLI/gateway startup. All configured providers are supported: `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. When a provider is set, the system automatically resolves the correct base URL, API key, and API mode — no manual credential wiring needed.

**Precedence:** per-call `provider` on `delegate_task` → `delegation.base_url` in config → `delegation.provider` in config → parent provider (inherited). Per-call `model` on `delegate_task` → `delegation.model` in config → parent model (inherited). Passing `"self"`, `"inherit"`, or `"parent"` as a per-call value means inherit. Setting just `model` without `provider` changes only the model name while keeping the inherited credentials (useful for switching models within the same provider like OpenRouter).

**Width and depth:** `max_concurrent_children` caps how many subagents run in parallel per batch (default `3`, floor of 1, no ceiling). Can also be set via the `DELEGATION_MAX_CONCURRENT_CHILDREN` env var. When the model submits a `tasks` array longer than the cap, `delegate_task` returns a tool error explaining the limit rather than silently truncating. `max_spawn_depth` controls the delegation tree depth (clamped to 1-3). At the default `1`, delegation is flat: children cannot spawn grandchildren, and passing `role="orchestrator"` silently degrades to `leaf`. Raise to `2` so orchestrator children can spawn leaf grandchildren; `3` for three-level trees. The agent opts into orchestration per call via `role="orchestrator"`; `orchestrator_enabled: false` forces every child back to leaf regardless. Cost scales multiplicatively — at `max_spawn_depth: 3` with `max_concurrent_children: 3`, the tree can reach 3×3×3 = 27 concurrent leaf agents. See [Subagent Delegation → Depth Limit and Nested Orchestration](features/delegation.md#depth-limit-and-nested-orchestration) for usage patterns.

## Clarify

Configure the clarification prompt behavior:

```yaml
clarify:
  timeout: 120                 # Seconds to wait for user clarification response
```

## Context Files (SOUL.md, AGENTS.md)

Hermes uses two different context scopes:

| File | Purpose | Scope |
|------|---------|-------|
| `SOUL.md` | **Primary agent identity** — defines who the agent is (slot #1 in the system prompt) | `~/.hermes/SOUL.md` or `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | Project-specific instructions (highest priority) | Walks to git root |
| `AGENTS.md` | Project-specific instructions, coding conventions | Recursive directory walk |
| `CLAUDE.md` | Claude Code context files (also detected) | Working directory only |
| `.cursorrules` | Cursor IDE rules (also detected) | Working directory only |
| `.cursor/rules/*.mdc` | Cursor rule files (also detected) | Working directory only |

- **SOUL.md** is the agent's primary identity. It occupies slot #1 in the system prompt, completely replacing the built-in default identity. Edit it to fully customize who the agent is.
- If SOUL.md is missing, empty, or cannot be loaded, Hermes falls back to a built-in default identity.
- **Project context files use a priority system** — only ONE type is loaded (first match wins): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. SOUL.md is always loaded independently.
- **AGENTS.md** is hierarchical: if subdirectories also have AGENTS.md, all are combined.
- Hermes automatically seeds a default `SOUL.md` if one does not already exist.
- All loaded context files are capped at `context_file_max_chars` characters (default 20,000) with smart truncation.

See also:
- [Personality & SOUL.md](/user-guide/features/personality)
- [Context Files](/user-guide/features/context-files)

## Working Directory

| Context | Default |
|---------|---------|
| **CLI (`hermes`)** | Current directory where you run the command |
| **Messaging gateway** | `terminal.cwd` from `~/.hermes/config.yaml`; if unset, home directory `~` |
| **Docker / Singularity / Modal / SSH** | User's home directory inside the container or remote machine |

Override the working directory:
```yaml
# In ~/.hermes/config.yaml:
terminal:
  cwd: /home/myuser/projects
```

`MESSAGING_CWD` and direct `TERMINAL_CWD` entries in `~/.hermes/.env` are legacy compatibility fallbacks. New configurations should use `terminal.cwd`.

## Network

Connectivity workarounds for outbound HTTP:

```yaml
network:
  force_ipv4: false   # Force IPv4 for outbound connections (default: false)
```

`force_ipv4` — on servers with broken or unreachable IPv6, Python resolves AAAA records first and can hang for the full TCP timeout before falling back to IPv4. Set this to `true` to skip IPv6 entirely and connect over IPv4 directly.

## Onboarding

First-touch onboarding hints and the structured profile-build offer:

```yaml
onboarding:
  profile_build: "ask"   # "ask" (default) | "off"
  seen: {}               # internal latch — leave empty
```

- `profile_build` — controls the profile-build path offered on the very first gateway message ever. `"ask"` (default) offers to build a user profile; the offer is **opt-in and consent-gated** — the agent asks before any lookup and never reads connected accounts silently. `"off"` shows a plain intro only. The offer fires at most once.
- `seen` — internal state. Hermes latches each shown hint here so it never fires again; the profile-build offer is also recorded here once shown. Don't hand-edit it — wipe the whole `onboarding` section if you want to re-see all hints.

## Dashboard

Configuration for the [web dashboard](/user-guide/features/web-dashboard) — visual theme, public URL, and authentication providers. The auth providers (OAuth, basic password, drain) are documented in detail on the web-dashboard page; this is the `config.yaml` shape.

```yaml
dashboard:
  theme: "default"            # "default" | "midnight" | "ember" | "mono" | "cyberpunk" | "rose"
  show_token_analytics: false # Re-enable the (local-estimate-only) token/cost analytics surfaces
  public_url: ""              # Full public authority for OAuth redirect_uri (env: HERMES_DASHBOARD_PUBLIC_URL)
  oauth:                      # Portal OAuth gate (engaged with --host and not --insecure)
    client_id: ""             # agent:{instance_id} — Portal provisions this
    portal_url: ""            # blank → plugin default (production Portal)
  basic_auth:                 # Self-hosted username/password gate (dashboard_auth/basic plugin)
    username: ""              # blank → plugin no-op
    password_hash: ""         # scrypt$... (preferred — no plaintext at rest)
    password: ""              # plaintext fallback (hashed in-memory at load)
    secret: ""                # token-signing key; blank → random per-process
    session_ttl_seconds: 0    # 0 → plugin default (12h)
  drain_auth:                 # Drain-control service-credential gate (dashboard_auth/drain plugin)
    scope: "drain"            # capability label on the verified principal
    min_secret_chars: 43      # entropy bar (url-safe-b64 chars; 43 ≈ 256 bits)
```

- `theme` — dashboard visual theme.
- `show_token_analytics` — off by default. The Analytics page and token/cost figures are a **local lower-bound estimate** (they exclude auxiliary calls, retries, fallbacks, and cache writes), so they can read far below the provider bill. Set `true` only if you understand they're not billing.
- `public_url` — when set, this is the complete authority (scheme + host + optional path prefix) the OAuth `redirect_uri` is built from. Set it for deploys behind reverse proxies that don't reliably forward `X-Forwarded-*` headers. Leave empty to use proxy-header reconstruction.
- `oauth` / `basic_auth` / `drain_auth` — auth provider config read by the bundled dashboard-auth plugins. The drain secret itself is **not** set here; it's provisioned via the `HERMES_DASHBOARD_DRAIN_SECRET` env var. See [Web Dashboard](/user-guide/features/web-dashboard) for full auth setup.
