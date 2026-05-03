---
sidebar_position: 2
title: "Configuration"
description: "Configure Hermes Agent — config.yaml, providers, models, API keys, and more"
---

# Configuration

All settings are stored in the `~/.hermes/` directory for easy access.

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

For AI provider setup (OpenRouter, Anthropic, Copilot, custom endpoints, self-hosted LLMs, fallback models, etc.), see [AI Providers](/docs/integrations/providers).

### Provider Timeouts

You can set `providers.<id>.request_timeout_seconds` for a provider-wide request timeout, plus `providers.<id>.models.<model>.timeout_seconds` for a model-specific override. Applies to the primary turn client on every transport (OpenAI-wire, native Anthropic, Anthropic-compatible), the fallback chain, rebuilds after credential rotation, and (for OpenAI-wire) the per-request timeout kwarg — so the configured value wins over the legacy `HERMES_API_TIMEOUT` env var.

You can also set `providers.<id>.stale_timeout_seconds` for the non-streaming stale-call detector, plus `providers.<id>.models.<model>.stale_timeout_seconds` for a model-specific override. This wins over the legacy `HERMES_API_CALL_STALE_TIMEOUT` env var.

Leaving these unset keeps the legacy defaults (`HERMES_API_TIMEOUT=1800`s, `HERMES_API_CALL_STALE_TIMEOUT=300`s, native Anthropic 900s). Not currently wired for AWS Bedrock (both `bedrock_converse` and AnthropicBedrock SDK paths use boto3 with its own timeout configuration). See the commented example in [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example).

## Terminal Backend Configuration

Hermes supports seven terminal backends. Each determines where the agent's shell commands actually execute — your local machine, a Docker container, a remote server via SSH, a Modal cloud sandbox (direct or via the Nous-managed gateway), a Daytona workspace, a Vercel Sandbox, or a Singularity/Apptainer container.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  timeout: 180      # Per-command timeout in seconds
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Container image for Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Container image for Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Container image for Daytona backend
```

For cloud sandboxes such as Modal, Daytona, and Vercel Sandbox, `container_persistent: true` means Hermes will try to preserve filesystem state across sandbox recreation. It does not promise that the same live sandbox, PID space, or background processes will still be running later.

### Backend Overview

| Backend | Where commands run | Isolation | Best for |
|---------|-------------------|-----------|----------|
| **local** | Your machine directly | None | Development, personal use |
| **docker** | Single persistent Docker container (shared across session, `/new`, subagents) | Full (namespaces, cap-drop) | Safe sandboxing, CI/CD |
| **ssh** | Remote server via SSH | Network boundary | Remote dev, powerful hardware |
| **modal** | Modal cloud sandbox | Full (cloud VM) | Ephemeral cloud compute, evals |
| **daytona** | Daytona workspace | Full (cloud container) | Managed cloud dev environments |
| **vercel_sandbox** | Vercel Sandbox | Full (cloud microVM) | Cloud execution with snapshot-backed filesystem persistence |
| **singularity** | Singularity/Apptainer container | Namespaces (--containall) | HPC clusters, shared machines |

### Local Backend

The default. Commands run directly on your machine with no isolation. No special setup required.

```yaml
terminal:
  backend: local
```

:::warning
The agent has the same filesystem access as your user account. Use `hermes tools` to disable tools you don't want, or switch to Docker for sandboxing.
:::

### Docker Backend

Runs commands inside a Docker container with security hardening (all capabilities dropped, no privilege escalation, PID limits).

**Single persistent container, not per-command.** Hermes starts ONE long-lived container on first use and routes every terminal, file, and `execute_code` call through `docker exec` into that same container — across sessions, `/new`, `/reset`, and `delegate_task` subagents — for the lifetime of the Hermes process. Working-directory changes, installed packages, and files in `/workspace` carry over from one tool call to the next, just like a local shell. The container is stopped and removed on shutdown. See **Container lifecycle** below for details.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace
  docker_run_as_host_user: false   # See "Running container as host user" below
  docker_forward_env:              # Env vars to forward into container
    - "GITHUB_TOKEN"
  docker_volumes:                  # Host directory mounts
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro for read-only

  # Resource limits
  container_cpu: 1                 # CPU cores (0 = unlimited)
  container_memory: 5120           # MB (0 = unlimited)
  container_disk: 51200            # MB (requires overlay2 on XFS+pquota)
  container_persistent: true       # Persist /workspace and /root across sessions
```

**Requirements:** Docker Desktop or Docker Engine installed and running. Hermes probes `$PATH` plus common macOS install locations (`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, Docker Desktop app bundle). Podman is supported out of the box: set `HERMES_DOCKER_BINARY=podman` (or the full path) to force it when both are installed.

**Container lifecycle:** Hermes reuses a single long-lived container (`docker run -d ... sleep 2h`) for every terminal and file-tool call, across sessions, `/new`, `/reset`, and `delegate_task` subagents, for the lifetime of the Hermes process. Commands run via `docker exec` with a login shell, so working-directory changes, installed packages, and files in `/workspace` all persist from one tool call to the next. The container is stopped and removed on Hermes shutdown (or when the idle-sweep reclaims it).

Parallel subagents spawned via `delegate_task(tasks=[...])` share this one container — concurrent `cd`, env mutations, and writes to the same path will collide. If a subagent needs an isolated sandbox, it must register a per-task image override via `register_task_env_overrides()`, which RL and benchmark environments (TerminalBench2, HermesSweEnv, etc.) do automatically for their per-task Docker images.

**Security hardening:**
- `--cap-drop ALL` with only `DAC_OVERRIDE`, `CHOWN`, `FOWNER` added back
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- Size-limited tmpfs for `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)

**Credential forwarding:** Env vars listed in `docker_forward_env` are resolved from your shell environment first, then `~/.hermes/.env`. Skills can also declare `required_environment_variables` which are merged automatically.

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

### Vercel Sandbox Backend

Runs commands in a [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) cloud microVM. Hermes uses the normal terminal and file tool surfaces; there are no Vercel-specific model-facing tools.

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # default workspace root
  container_persistent: true      # Snapshot/restore filesystem
  container_disk: 51200           # Shared default only; custom disk is unsupported
```

**Required install:** Install the optional SDK extra:

```bash
pip install 'hermes-agent[vercel]'
```

**Required authentication:** Configure access-token auth with all three of `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, and `VERCEL_TEAM_ID`. This is the supported setup for deployments and normal long-running Hermes processes on Render, Railway, Docker, and similar hosts.

For one-off local development, Hermes also accepts short-lived Vercel OIDC tokens:

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

From a linked Vercel project directory, you can omit the project name:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

OIDC tokens are short-lived and should not be used as the documented deployment path.

**Runtime:** `terminal.vercel_runtime` supports `node24`, `node22`, and `python3.13`. If unset, Hermes defaults to `node24`.

**Persistence:** When `container_persistent: true`, Hermes snapshots the sandbox filesystem during cleanup and restores a later sandbox for the same task from that snapshot. Snapshot contents can include Hermes-synced credentials, skills, and cache files that were copied into the sandbox. This preserves filesystem state only; it does not preserve live sandbox identity, PID space, shell state, or running background processes.

**Background commands:** `terminal(background=true)` uses Hermes' generic non-local background process flow. You can spawn, poll, wait, view logs, and kill processes through the normal process tool while the sandbox is alive. Hermes does not provide native Vercel detached-process recovery after cleanup or restart.

**Disk sizing:** Vercel Sandbox does not currently support Hermes' `container_disk` resource knob. Leave `container_disk` unset or at the shared default `51200`; non-default values fail diagnostics and backend creation instead of being silently ignored.

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

For details on declaring config settings in your own skills, see [Creating Skills — Config Settings](/docs/developer-guide/creating-skills#config-settings-configyaml).

### Guard on agent-created skill writes

When the agent uses `skill_manage` to create, edit, patch, or delete a skill, Hermes can optionally scan the new/updated content for dangerous keyword patterns (credential harvesting, obvious prompt injection, exfil instructions). The scanner is **off by default** — real agent workflows that legitimately touch `~/.ssh/` or mention `$OPENAI_API_KEY` were tripping the heuristic too often. Turn it back on if you want the scanner to prompt you before the agent's skill writes land:

```yaml
skills:
  guard_agent_created: true   # default: false
```

When on, any flagged `skill_manage` write surfaces as an approval prompt with the scanner's rationale. Accepted writes land; denied writes return an explanatory error to the agent.

## Memory Configuration

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
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
  hygiene_hard_message_limit: 400                   # Gateway safety valve — see below

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: "google/gemini-3-flash-preview"          # Model for summarization
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info Legacy config migration
Older configs with `compression.summary_model`, `compression.summary_provider`, and `compression.summary_base_url` are automatically migrated to `auxiliary.compression.*` on first load (config version 17). No manual action needed.
:::

`hygiene_hard_message_limit` is a gateway-only **pre-compression safety valve**. Runaway sessions with thousands of messages can hit model context limits before the normal percent-of-context threshold fires; when message count crosses this ceiling, Hermes forces compression regardless of token usage. Default `400` — raise it for platforms where very long sessions are normal, lower it to force more aggressive compression. Editing this value on a running gateway takes effect on the next message (see below).

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

The context engine controls how conversations are managed when approaching the model's token limit. The built-in `compressor` engine uses lossy summarization (see [Context Compression](/docs/developer-guide/context-compression-and-caching)). Plugin engines can replace it with alternative strategies.

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

See [Memory Providers](/docs/user-guide/features/memory-providers) for the analogous single-select system for memory plugins.

## Iteration Budget Pressure

When the agent is working on a complex task with many tool calls, it can burn through its iteration budget (default: 90 turns) without realizing it's running low. Budget pressure automatically warns the model as it approaches the limit:

| Threshold | Level | What the model sees |
|-----------|-------|---------------------|
| **70%** | Caution | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | Warning | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

Warnings are injected into the last tool result's JSON (as a `_budget_warning` field) rather than as separate messages — this preserves prompt caching and doesn't disrupt the conversation structure.

```yaml
agent:
  max_turns: 90                # Max iterations per conversation turn (default: 90)
  api_max_retries: 2           # Retries per provider before fallback engages (default: 2)
```

Budget pressure is enabled by default. The agent sees warnings naturally as part of tool results, encouraging it to consolidate its work and deliver a response before running out of iterations.

When the iteration budget is fully exhausted, the CLI shows a notification to the user: `⚠ Iteration budget reached (90/90) — response may be incomplete`. If the budget runs out during active work, the agent generates a summary of what was accomplished before stopping.

`agent.api_max_retries` controls how many times Hermes retries a provider API call on transient errors (rate limits, connection drops, 5xx) **before** fallback-provider switching engages. The default is `2` — three attempts total, matching the OpenAI SDK default. If you have [fallback providers](/docs/user-guide/features/fallback-providers) configured and want to fail over faster, drop this to `0` so the first transient error on your primary immediately hands off to the fallback instead of churning retries against the flaky endpoint.

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

Options: `fill_first` (default), `round_robin`, `least_used`, `random`. See [Credential Pools](/docs/user-guide/features/credential-pools) for full documentation.

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
[ ] session_search       currently: openrouter / google/gemini-2.5-flash
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
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

Available providers for auxiliary tasks: `auto`, `main`, plus any provider in the [provider registry](/docs/reference/environment-variables) — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `google-gemini-cli`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `ai-gateway`, `azure-foundry` — or any named custom provider from your `custom_providers` list (e.g. `provider: "beans"`).

:::tip MiniMax OAuth
`minimax-oauth` logs in via browser OAuth (no API key needed). Run `hermes model` and select **MiniMax (OAuth)** to authenticate. Auxiliary tasks use `MiniMax-M2.7-highspeed` automatically. See the [MiniMax OAuth guide](../guides/minimax-oauth.md).
:::

:::warning `"main"` is for auxiliary tasks only
The `"main"` provider option means "use whatever provider my main agent uses" — it's only valid inside `auxiliary:`, `compression:`, and `fallback_model:` configs. It is **not** a valid value for your top-level `model.provider` setting. If you use a custom OpenAI-compatible endpoint, set `provider: custom` in your `model:` section. See [AI Providers](/docs/integrations/providers) for all main model provider options.
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

  # Context compression timeout (separate from compression.* config)
  compression:
    timeout: 120               # seconds — compression summarizes long conversations, needs more time

  # Session search — summarizes past session matches
  session_search:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    max_concurrency: 3       # Limit parallel summaries to reduce request-burst 429s
    extra_body: {}           # Provider-specific OpenAI-compatible request fields

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
```

:::tip
Each auxiliary task has a configurable `timeout` (in seconds). Defaults: vision 120s, web_extract 360s, approval 30s, compression 120s. Increase these if you use slow local models for auxiliary tasks. Vision also has a separate `download_timeout` (default 30s) for the HTTP image download — increase this for slow connections or self-hosted image servers.
:::

:::info
Context compression has its own `compression:` block for thresholds and an `auxiliary.compression:` block for model/provider settings — see [Context Compression](#context-compression) above. The fallback model uses a `fallback_model:` block — see [Fallback Model](/docs/integrations/providers#fallback-model). All three follow the same provider/model/base_url pattern.
:::

### Session Search Tuning

If you use a reasoning-heavy model for `auxiliary.session_search`, Hermes now gives you two built-in controls:

- `auxiliary.session_search.max_concurrency`: limits how many matched sessions Hermes summarizes at once
- `auxiliary.session_search.extra_body`: forwards provider-specific OpenAI-compatible request fields on the summarization calls

Example:

```yaml
auxiliary:
  session_search:
    provider: "main"
    model: "glm-4.5-air"
    timeout: 60
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

Use `max_concurrency` when your provider rate-limits request bursts and you want `session_search` to trade some parallelism for stability.

Use `extra_body` only when your provider documents OpenAI-compatible request-body fields you want Hermes to pass through for that task. Hermes forwards the object as-is.

:::warning
`extra_body` is only effective when your provider actually supports the field you send. If the provider does not expose a native OpenAI-compatible reasoning-off flag, Hermes cannot synthesize one on its behalf.
:::

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

These options apply to **auxiliary task configs** (`auxiliary:`, `compression:`, `fallback_model:`), not to your main `model.provider` setting.

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | Best available (default). Vision tries OpenRouter → Nous → Codex. | — |
| `"openrouter"` | Force OpenRouter — routes to any model (Gemini, GPT-4o, Claude, etc.) | `OPENROUTER_API_KEY` |
| `"nous"` | Force Nous Portal | `hermes auth` |
| `"codex"` | Force Codex OAuth (ChatGPT account). Supports vision (gpt-5.3-codex). | `hermes model` → Codex |
| `"minimax-oauth"` | Force MiniMax OAuth (browser login, no API key). Uses MiniMax-M2.7-highspeed for auxiliary tasks. | `hermes model` → MiniMax (OAuth) |
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
    model: "gemini-2.5-flash-preview-tts"   # or gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, etc.
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
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  runtime_metadata_footer: false  # Gateway: append a runtime-context footer to final replies
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | ja | de | es
```

### UI language for static messages

The `display.language` setting translates a small set of static user-facing messages — the CLI approval prompt, a handful of gateway slash-command replies (e.g. restart-drain notices, "approval expired", "goal cleared"). It does **not** translate agent responses, log lines, tool output, error tracebacks, or slash-command descriptions — those stay in English. If you want the agent itself to reply in another language, just tell it in your prompt or system message.

Supported values: `en` (default), `zh` (Simplified Chinese), `ja` (Japanese), `de` (German), `es` (Spanish). Unknown values fall back to English.

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

### Runtime-metadata footer (gateway only)

When `display.runtime_metadata_footer: true`, Hermes appends a small runtime-context footer to the **final** message of each gateway turn — same info the CLI shows in its status bar (model, session duration, tokens, cost). Off by default; opt in per-gateway if your team wants every reply to include the provenance.

```yaml
display:
  runtime_metadata_footer: true
```

Example footer appended to a Telegram/Discord/Slack reply:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

Only the **final** message of a turn gets the footer; interim updates stay clean.

### Per-platform progress overrides

Different platforms have different verbosity needs. For example, Signal can't edit messages, so each progress update becomes a separate message — noisy. Use `display.platforms` to set per-platform modes:

```yaml
display:
  tool_progress: all          # global default
  platforms:
    signal:
      tool_progress: 'off'    # silence progress on Signal
    telegram:
      tool_progress: verbose  # detailed progress on Telegram
    slack:
      tool_progress: 'off'    # quiet in shared Slack workspace
```

Platforms without an override fall back to the global `tool_progress` value. Valid platform keys: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. The legacy `display.tool_progress_overrides` key still loads for backward compatibility but is deprecated and migrated into `display.platforms` on first load.

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
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # Legacy fallback key still respected
```

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

Use `/voice on` in the CLI to enable microphone mode, `record_key` to start/stop recording, and `/voice tts` to toggle spoken replies. See [Voice Mode](/docs/user-guide/features/voice-mode) for end-to-end setup and platform-specific behavior.

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
  fresh_final_after_seconds: 60   # Send fresh final (Telegram) when preview is this old; 0 = always edit in place
```

When enabled, the bot sends a message on the first token, then progressively edits it as more tokens arrive. Platforms that don't support message editing (Signal, Email, Home Assistant) are auto-detected on the first attempt — streaming is gracefully disabled for that session with no flood of messages.

For separate natural mid-turn assistant updates without progressive token editing, set `display.interim_assistant_messages: true`.

**Overflow handling:** If the streamed text exceeds the platform's message length limit (~4096 chars), the current message is finalized and a new one starts automatically.

**Fresh final (Telegram):** Telegram's `editMessageText` preserves the original message timestamp, so a long-running streamed reply would keep the first-token timestamp even after completion. When `fresh_final_after_seconds > 0` (default `60`), the completed reply is delivered as a brand-new message (with the stale preview best-effort deleted) so Telegram's visible timestamp reflects completion time. Short previews still finalize in place. Set to `0` to always edit in place.

:::note
Streaming is disabled by default. Enable it in `~/.hermes/config.yaml` to try the streaming UX.
:::

## Group Chat Session Isolation

Control whether shared chats keep one conversation per room or one conversation per participant:

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true` is the default and recommended setting. In Discord channels, Telegram groups, Slack channels, and similar shared contexts, each sender gets their own session when the platform provides a user ID.
- `false` reverts to the old shared-room behavior. That can be useful if you explicitly want Hermes to treat a channel like one collaborative conversation, but it also means users share context, token costs, and interrupt state.
- Direct messages are unaffected. Hermes still keys DMs by chat/DM ID as usual.
- Threads stay isolated from their parent channel either way; with `true`, each participant also gets their own session inside the thread.

For the behavior details and examples, see [Sessions](/docs/user-guide/sessions) and the [Discord guide](/docs/user-guide/messaging/discord).

## Unauthorized DM Behavior

Control what Hermes does when an unknown user sends a direct message:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` is the default. Hermes denies access, but replies with a one-time pairing code in DMs.
- `ignore` silently drops unauthorized DMs.
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

The `web_search`, `web_extract`, and `web_crawl` tools support four backend providers. Configure the backend in `config.yaml` or via `hermes tools`:

```yaml
web:
  backend: firecrawl    # firecrawl | parallel | tavily | exa
```

| Backend | Env Var | Search | Extract | Crawl |
|---------|---------|--------|---------|-------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

**Backend selection:** If `web.backend` is not set, the backend is auto-detected from available API keys. If only `EXA_API_KEY` is set, Exa is used. If only `TAVILY_API_KEY` is set, Tavily is used. If only `PARALLEL_API_KEY` is set, Parallel is used. Otherwise Firecrawl is the default.

**Self-hosted Firecrawl:** Set `FIRECRAWL_API_URL` to point at your own instance. When a custom URL is set, the API key becomes optional (set `USE_DB_AUTHENTICATION=false` on the server to disable auth).

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
  # Chrome (via /browser connect) rather than starting a headless browser.
  cdp_url: ""
  # Dialog supervisor — controls how native JS dialogs (alert / confirm / prompt)
  # are handled when a CDP backend is attached (Browserbase, local Chrome via
  # /browser connect). Ignored on Camofox and default local agent-browser mode.
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # Safety auto-dismiss under must_respond (seconds)
  camofox:
    managed_persistence: false   # When true, Camofox sessions persist cookies/logins across restarts
```

**Dialog policies:**

- `must_respond` (default) — capture the dialog, surface it in `browser_snapshot.pending_dialogs`, and wait for the agent to call `browser_dialog(action=...)`. After `dialog_timeout_s` seconds with no response, the dialog is auto-dismissed to prevent the page's JS thread from stalling forever.
- `auto_dismiss` — capture, dismiss immediately. The agent still sees the dialog record in `browser_snapshot.recent_dialogs` with `closed_by="auto_policy"` after the fact.
- `auto_accept` — capture, accept immediately. Useful for pages with aggressive `beforeunload` prompts.

See the [browser feature page](./features/browser.md#browser_dialog) for the full dialog workflow.

The browser toolset supports multiple providers. See the [Browser feature page](/docs/user-guide/features/browser) for details on Browserbase, Browser Use, and local Chrome CDP setup.

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
  redact_secrets: false          # Redact API key patterns in tool output and logs (off by default)
  tirith_enabled: true           # Enable Tirith security scanning for terminal commands
  tirith_path: "tirith"          # Path to tirith binary (default: "tirith" in $PATH)
  tirith_timeout: 5              # Seconds to wait for tirith scan before timing out
  tirith_fail_open: true         # Allow command execution if tirith is unavailable
  website_blocklist:             # See Website Blocklist section below
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — when `true`, automatically detects and redacts patterns that look like API keys, tokens, and passwords in tool output before it enters the conversation context and logs. **Off by default** — enable if you commonly work with real credentials in tool output and want a safety net. Set to `true` explicitly to turn on.
- `tirith_enabled` — when `true`, terminal commands are scanned by [Tirith](https://github.com/StackGuardian/tirith) before execution to detect potentially dangerous operations.
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

## Checkpoints

Automatic filesystem snapshots before destructive file operations. See the [Checkpoints & Rollback](/docs/user-guide/checkpoints-and-rollback) for details.

```yaml
checkpoints:
  enabled: true                  # Enable automatic checkpoints (also: hermes --checkpoints)
  max_snapshots: 50              # Max checkpoints to keep per directory
```


## Delegation

Configure subagent behavior for the delegate tool:

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
```

**Subagent provider:model override:** By default, subagents inherit the parent agent's provider and model. Set `delegation.provider` and `delegation.model` to route subagents to a different provider:model pair — e.g., use a cheap/fast model for narrowly-scoped subtasks while your primary agent runs an expensive reasoning model.

**Direct endpoint override:** If you want the obvious custom-endpoint path, set `delegation.base_url`, `delegation.api_key`, and `delegation.model`. That sends subagents directly to that OpenAI-compatible endpoint and takes precedence over `delegation.provider`. If `delegation.api_key` is omitted, Hermes falls back to `OPENAI_API_KEY` only.

The delegation provider uses the same credential resolution as CLI/gateway startup. All configured providers are supported: `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. When a provider is set, the system automatically resolves the correct base URL, API key, and API mode — no manual credential wiring needed.

**Precedence:** `delegation.base_url` in config → `delegation.provider` in config → parent provider (inherited). `delegation.model` in config → parent model (inherited). Setting just `model` without `provider` changes only the model name while keeping the parent's credentials (useful for switching models within the same provider like OpenRouter).

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
- All loaded context files are capped at 20,000 characters with smart truncation.

See also:
- [Personality & SOUL.md](/docs/user-guide/features/personality)
- [Context Files](/docs/user-guide/features/context-files)

## Working Directory

| Context | Default |
|---------|---------|
| **CLI (`hermes`)** | Current directory where you run the command |
| **Messaging gateway** | Home directory `~` (override with `MESSAGING_CWD`) |
| **Docker / Singularity / Modal / SSH** | User's home directory inside the container or remote machine |

Override the working directory:
```bash
# In ~/.hermes/.env or ~/.hermes/config.yaml:
MESSAGING_CWD=/home/myuser/projects    # Gateway sessions
TERMINAL_CWD=/workspace                # All terminal sessions
```
