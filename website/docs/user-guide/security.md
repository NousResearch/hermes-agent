---
sidebar_position: 8
title: "Security"
description: "Security model, dangerous command approval, user authorization, container isolation, and production deployment best practices"
---

# Security

Hermes Agent is designed with a defense-in-depth security model. This page covers every security boundary — from command approval to container isolation to user authorization on messaging platforms.

## Overview

The security model has five layers:

1. **User authorization** — who can talk to the agent (allowlists, DM pairing)
2. **Dangerous command approval** — human-in-the-loop for destructive operations
3. **Container isolation** — Docker/Singularity/Modal sandboxing with hardened settings
4. **MCP credential filtering** — environment variable isolation for MCP subprocesses
5. **Context file scanning** — prompt injection detection in project files

## Dangerous Command Approval

Before executing any command, Hermes checks it against a curated list of dangerous patterns. If a match is found, the user must explicitly approve it.

### Approval Modes

The approval system supports three modes, configured via `approvals.mode` in `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: manual    # manual | smart | off
  timeout: 60     # seconds to wait for user response (default: 60)
```

| Mode | Behavior |
|------|----------|
| **manual** (default) | Always prompt the user for approval on dangerous commands |
| **smart** | Use an auxiliary LLM to assess risk. Low-risk commands (e.g., `python -c "print('hello')"`) are auto-approved. Genuinely dangerous commands are auto-denied. Uncertain cases escalate to a manual prompt. |
| **off** | Disable all approval checks — equivalent to running with `--yolo`. All commands execute without prompts. |

:::warning
Setting `approvals.mode: off` disables all safety prompts. Use only in trusted environments (CI/CD, containers, etc.).
:::

### YOLO Mode

YOLO mode bypasses **all** dangerous command approval prompts for the current session. It can be activated three ways:

1. **CLI flag**: Start a session with `hermes --yolo` or `hermes chat --yolo`
2. **Slash command**: Type `/yolo` during a session to toggle it on/off
3. **Environment variable**: Set `HERMES_YOLO_MODE=1`

The `/yolo` command is a **toggle** — each use flips the mode on or off:

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLO mode is available in both CLI and gateway sessions. Internally, it sets the `HERMES_YOLO_MODE` environment variable which is checked before every command execution.

:::danger
YOLO mode disables **all** dangerous command safety checks for the session. Use only when you fully trust the commands being generated (e.g., well-tested automation scripts in disposable environments).
:::

### Approval Timeout

When a dangerous command prompt appears, the user has a configurable amount of time to respond. If no response is given within the timeout, the command is **denied** by default (fail-closed).

Configure the timeout in `~/.hermes/config.yaml`:

```yaml
approvals:
  timeout: 60  # seconds (default: 60)
```

### What Triggers Approval

The following patterns trigger approval prompts (defined in `tools/approval.py`):

| Pattern | Description |
|---------|-------------|
| `rm -r` / `rm --recursive` | Recursive delete |
| `rm ... /` | Delete in root path |
| `chmod 777/666` / `o+w` / `a+w` | World/other-writable permissions |
| `chmod --recursive` with unsafe perms | Recursive world/other-writable (long flag) |
| `chown -R root` / `chown --recursive root` | Recursive chown to root |
| `mkfs` | Format filesystem |
| `dd if=` | Disk copy |
| `> /dev/sd` | Write to block device |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM` (without WHERE) | SQL DELETE without WHERE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | Overwrite system config |
| `systemctl stop/disable/mask` | Stop/disable system services |
| `kill -9 -1` | Kill all processes |
| `pkill -9` | Force kill processes |
| Fork bomb patterns | Fork bombs |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | Shell command execution via `-c` flag (including combined flags like `-lc`) |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | Script execution via `-e`/`-c` flag |
| `curl ... \| sh` / `wget ... \| sh` | Pipe remote content to shell |
| `bash <(curl ...)` / `sh <(wget ...)` | Execute remote script via process substitution |
| `tee` to `/etc/`, `~/.ssh/`, `~/.hermes/.env` | Overwrite sensitive file via tee |
| `>` / `>>` to `/etc/`, `~/.ssh/`, `~/.hermes/.env` | Overwrite sensitive file via redirection |
| `xargs rm` | xargs with rm |
| `find -exec rm` / `find -delete` | Find with destructive actions |
| `cp`/`mv`/`install` to `/etc/` | Copy/move file into system config |
| `sed -i` / `sed --in-place` on `/etc/` | In-place edit of system config |
| `pkill`/`killall` hermes/gateway | Self-termination prevention |
| `gateway run` with `&`/`disown`/`nohup`/`setsid` | Prevents starting gateway outside service manager |

:::info
**Container bypass**: When running in `docker`, `singularity`, `modal`, or `daytona` backends, dangerous command checks are **skipped** because the container itself is the security boundary. Destructive commands inside a container can't harm the host.
:::

### Approval Flow (CLI)

In the interactive CLI, dangerous commands show an inline approval prompt:

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

The four options:

- **once** — allow this single execution
- **session** — allow this pattern for the rest of the session
- **always** — add to permanent allowlist (saved to `config.yaml`)
- **deny** (default) — block the command

### Approval Flow (Gateway/Messaging)

On messaging platforms, the agent sends the dangerous command details to the chat and waits for the user to reply:

- Reply **yes**, **y**, **approve**, **ok**, or **go** to approve
- Reply **no**, **n**, **deny**, or **cancel** to deny

The `HERMES_EXEC_ASK=1` environment variable is automatically set when running the gateway.

### Permanent Allowlist

Commands approved with "always" are saved to `~/.hermes/config.yaml`:

```yaml
# Permanently allowed dangerous command patterns
command_allowlist:
  - rm
  - systemctl
```

These patterns are loaded at startup and silently approved in all future sessions.

:::tip
Use `hermes config edit` to review or remove patterns from your permanent allowlist.
:::

## User Authorization (Gateway)

When running the messaging gateway, Hermes controls who can interact with the bot through a layered authorization system.

### Authorization Check Order

The `_is_user_authorized()` method checks in this order:

1. **Per-platform allow-all flag** (e.g., `DISCORD_ALLOW_ALL_USERS=true`)
2. **DM pairing approved list** (users approved via pairing codes)
3. **Platform-specific allowlists** (e.g., `TELEGRAM_ALLOWED_USERS=12345,67890`)
4. **Global allowlist** (`GATEWAY_ALLOWED_USERS=12345,67890`)
5. **Global allow-all** (`GATEWAY_ALLOW_ALL_USERS=true`)
6. **Default: deny**

### Platform Allowlists

Set allowed user IDs as comma-separated values in `~/.hermes/.env`:

```bash
# Platform-specific allowlists
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# Cross-platform allowlist (checked for all platforms)
GATEWAY_ALLOWED_USERS=123456789

# Per-platform allow-all (use with caution)
DISCORD_ALLOW_ALL_USERS=true

# Global allow-all (use with extreme caution)
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
If **no allowlists are configured** and `GATEWAY_ALLOW_ALL_USERS` is not set, **all users are denied**. The gateway logs a warning at startup:

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM Pairing System

For more flexible authorization, Hermes includes a code-based pairing system. Instead of requiring user IDs upfront, unknown users receive a one-time pairing code that the bot owner approves via the CLI.

**How it works:**

1. An unknown user sends a DM to the bot
2. The bot replies with an 8-character pairing code
3. The bot owner runs `hermes pairing approve <platform> <code>` on the CLI
4. The user is permanently approved for that platform

Control how unauthorized direct messages are handled in `~/.hermes/config.yaml`:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` is the default. Unauthorized DMs get a pairing code reply.
- `ignore` silently drops unauthorized DMs.
- Platform sections override the global default, so you can keep pairing on Telegram while keeping WhatsApp silent.

**Security features** (based on OWASP + NIST SP 800-63-4 guidance):

| Feature | Details |
|---------|---------|
| Code format | 8-char from 32-char unambiguous alphabet (no 0/O/1/I) |
| Randomness | Cryptographic (`secrets.choice()`) |
| Code TTL | 1 hour expiry |
| Rate limiting | 1 request per user per 10 minutes |
| Pending limit | Max 3 pending codes per platform |
| Lockout | 5 failed approval attempts → 1-hour lockout |
| File security | `chmod 0600` on all pairing data files |
| Logging | Codes are never logged to stdout |

**Pairing CLI commands:**

```bash
# List pending and approved users
hermes pairing list

# Approve a pairing code
hermes pairing approve telegram ABC12DEF

# Revoke a user's access
hermes pairing revoke telegram 123456789

# Clear all pending codes
hermes pairing clear-pending
```

**Storage:** Pairing data is stored in `~/.hermes/pairing/` with per-platform JSON files:
- `{platform}-pending.json` — pending pairing requests
- `{platform}-approved.json` — approved users
- `_rate_limits.json` — rate limit and lockout tracking

## Container Isolation

When using the `docker` terminal backend, Hermes applies strict security hardening to every container.

### Docker Security Flags

Every container runs with these flags (defined in `tools/environments/docker.py`):

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # Drop ALL Linux capabilities
    "--cap-add", "DAC_OVERRIDE",                  # Root can write to bind-mounted dirs
    "--cap-add", "CHOWN",                         # Package managers need file ownership
    "--cap-add", "FOWNER",                        # Package managers need file ownership
    "--security-opt", "no-new-privileges",         # Block privilege escalation
    "--pids-limit", "256",                         # Limit process count
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # Size-limited /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # No-exec /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # No-exec /run
]
```

### Resource Limits

Container resources are configurable in `~/.hermes/config.yaml`:

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
  container_cpu: 1        # CPU cores
  container_memory: 5120  # MB (default 5GB)
  container_disk: 51200   # MB (default 50GB, requires overlay2 on XFS)
  container_persistent: true  # Persist filesystem across sessions
```

### Filesystem Persistence

- **Persistent mode** (`container_persistent: true`): Bind-mounts `/workspace` and `/root` from `~/.hermes/sandboxes/docker/<task_id>/`
- **Ephemeral mode** (`container_persistent: false`): Uses tmpfs for workspace — everything is lost on cleanup

:::tip
For production gateway deployments, use `docker`, `modal`, or `daytona` backend to isolate agent commands from your host system. This eliminates the need for dangerous command approval entirely.
:::

:::warning
If you add names to `terminal.docker_forward_env`, those variables are intentionally injected into the container for terminal commands. This is useful for task-specific credentials like `GITHUB_TOKEN`, but it also means code running in the container can read and exfiltrate them.
:::

## Terminal Backend Security Comparison

| Backend | Isolation | Dangerous Cmd Check | Best For |
|---------|-----------|-------------------|----------|
| **local** | None — runs on host | ✅ Yes | Development, trusted users |
| **ssh** | Remote machine | ✅ Yes | Running on a separate server |
| **docker** | Container | ❌ Skipped (container is boundary) | Production gateway |
| **singularity** | Container | ❌ Skipped | HPC environments |
| **modal** | Cloud sandbox | ❌ Skipped | Scalable cloud isolation |
| **daytona** | Cloud sandbox | ❌ Skipped | Persistent cloud workspaces |

## Environment Variable Passthrough {#environment-variable-passthrough}

Both `execute_code` and `terminal` strip sensitive environment variables from child processes to prevent credential exfiltration by LLM-generated code. However, skills that declare `required_environment_variables` legitimately need access to those vars.

### How It Works

Two mechanisms allow specific variables through the sandbox filters:

**1. Skill-scoped passthrough (automatic)**

When a skill is loaded (via `skill_view` or the `/skill` command) and declares `required_environment_variables`, any of those vars that are actually set in the environment are automatically registered as passthrough. Missing vars (still in setup-needed state) are **not** registered.

```yaml
# In a skill's SKILL.md frontmatter
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

After loading this skill, `TENOR_API_KEY` passes through to `execute_code`, `terminal` (local), **and remote backends (Docker, Modal)** — no manual configuration needed.

:::info Docker & Modal
Prior to v0.5.1, Docker's `forward_env` was a separate system from the skill passthrough. They are now merged — skill-declared env vars are automatically forwarded into Docker containers and Modal sandboxes without needing to add them to `docker_forward_env` manually.
:::

**2. Config-based passthrough (manual)**

For env vars not declared by any skill, add them to `terminal.env_passthrough` in `config.yaml`:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### Credential File Passthrough (OAuth tokens, etc.) {#credential-file-passthrough}

Some skills need **files** (not just env vars) in the sandbox — for example, Google Workspace stores OAuth tokens as `google_token.json` under the active profile's `HERMES_HOME`. Skills declare these in frontmatter:

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

When loaded, Hermes checks if these files exist in the active profile's `HERMES_HOME` and registers them for mounting:

- **Docker**: Read-only bind mounts (`-v host:container:ro`)
- **Modal**: Mounted at sandbox creation + synced before each command (handles mid-session OAuth setup)
- **Local**: No action needed (files already accessible)

You can also list credential files manually in `config.yaml`:

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

Paths are relative to `~/.hermes/`. Files are mounted to `/root/.hermes/` inside the container.

### What Each Sandbox Filters

| Sandbox | Default Filter | Passthrough Override |
|---------|---------------|---------------------|
| **execute_code** | Blocks vars containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH` in name; only allows safe-prefix vars through | ✅ Passthrough vars bypass both checks |
| **terminal** (local) | Blocks explicit Hermes infrastructure vars (provider keys, gateway tokens, tool API keys) | ✅ Passthrough vars bypass the blocklist |
| **terminal** (Docker) | No host env vars by default | ✅ Passthrough vars + `docker_forward_env` forwarded via `-e` |
| **terminal** (Modal) | No host env/files by default | ✅ Credential files mounted; env passthrough via sync |
| **MCP** | Blocks everything except safe system vars + explicitly configured `env` | ❌ Not affected by passthrough (use MCP `env` config instead) |

### Security Considerations

- The passthrough only affects vars you or your skills explicitly declare — the default security posture is unchanged for arbitrary LLM-generated code
- Credential files are mounted **read-only** into Docker containers
- Skills Guard scans skill content for suspicious env access patterns before installation
- Missing/unset vars are never registered (you can't leak what doesn't exist)
- Hermes infrastructure secrets (provider API keys, gateway tokens) should never be added to `env_passthrough` — they have dedicated mechanisms

## MCP Credential Handling

MCP (Model Context Protocol) server subprocesses receive a **filtered environment** to prevent accidental credential leakage.

### Safe Environment Variables

Only these variables are passed through from the host to MCP stdio subprocesses:

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

Plus any `XDG_*` variables. All other environment variables (API keys, tokens, secrets) are **stripped**.

Variables explicitly defined in the MCP server's `env` config are passed through:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # Only this is passed
```

### Credential Redaction

Error messages from MCP tools are sanitized before being returned to the LLM. The following patterns are replaced with `[REDACTED]`:

- GitHub PATs (`ghp_...`)
- OpenAI-style keys (`sk-...`)
- Bearer tokens
- `token=`, `key=`, `API_KEY=`, `password=`, `secret=` parameters

### Website Access Policy

You can restrict which websites the agent can access through its web and browser tools. This is useful for preventing the agent from accessing internal services, admin panels, or other sensitive URLs.

```yaml
# In ~/.hermes/config.yaml
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

When a blocked URL is requested, the tool returns an error explaining the domain is blocked by policy. The blocklist is enforced across `web_search`, `web_extract`, `browser_navigate`, and all URL-capable tools.

See [Website Blocklist](/docs/user-guide/configuration#website-blocklist) in the configuration guide for full details.

### SSRF Protection

All URL-capable tools (web search, web extract, vision, browser) validate URLs before fetching them to prevent Server-Side Request Forgery (SSRF) attacks. Blocked addresses include:

- **Private networks** (RFC 1918): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **Loopback**: `127.0.0.0/8`, `::1`
- **Link-local**: `169.254.0.0/16` (includes cloud metadata at `169.254.169.254`)
- **CGNAT / shared address space** (RFC 6598): `100.64.0.0/10` (Tailscale, WireGuard VPNs)
- **Cloud metadata hostnames**: `metadata.google.internal`, `metadata.goog`
- **Reserved, multicast, and unspecified addresses**

SSRF protection is always active and cannot be disabled. DNS failures are treated as blocked (fail-closed). Redirect chains are re-validated at each hop to prevent redirect-based bypasses.

### Tirith Pre-Exec Security Scanning

Hermes integrates [tirith](https://github.com/sheeki03/tirith) for content-level command scanning before execution. Tirith detects threats that pattern matching alone misses:

- Homograph URL spoofing (internationalized domain attacks)
- Pipe-to-interpreter patterns (`curl | bash`, `wget | sh`)
- Terminal injection attacks

Tirith auto-installs from GitHub releases on first use with SHA-256 checksum verification (and cosign provenance verification if cosign is available).

```yaml
# In ~/.hermes/config.yaml
security:
  tirith_enabled: true       # Enable/disable tirith scanning (default: true)
  tirith_path: "tirith"      # Path to tirith binary (default: PATH lookup)
  tirith_timeout: 5          # Subprocess timeout in seconds
  tirith_fail_open: true     # Allow execution when tirith is unavailable (default: true)
```

When `tirith_fail_open` is `true` (default), commands proceed if tirith is not installed or times out. Set to `false` in high-security environments to block commands when tirith is unavailable.

Tirith's verdict integrates with the approval flow: safe commands pass through, while both suspicious and blocked commands trigger user approval with the full tirith findings (severity, title, description, safer alternatives). Users can approve or deny — the default choice is deny to keep unattended scenarios secure.

### Context File Injection Protection

Context files (AGENTS.md, .cursorrules, SOUL.md) are scanned for prompt injection before being included in the system prompt. The scanner checks for:

- Instructions to ignore/disregard prior instructions
- Hidden HTML comments with suspicious keywords
- Attempts to read secrets (`.env`, `credentials`, `.netrc`)
- Credential exfiltration via `curl`
- Invisible Unicode characters (zero-width spaces, bidirectional overrides)

Blocked files show a warning:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## Best Practices for Production Deployment

### Gateway Deployment Checklist

1. **Set explicit allowlists** — never use `GATEWAY_ALLOW_ALL_USERS=true` in production
2. **Use container backend** — set `terminal.backend: docker` in config.yaml
3. **Restrict resource limits** — set appropriate CPU, memory, and disk limits
4. **Store secrets securely** — keep API keys in `~/.hermes/.env` with proper file permissions
5. **Enable DM pairing** — use pairing codes instead of hardcoding user IDs when possible
6. **Review command allowlist** — periodically audit `command_allowlist` in config.yaml
7. **Set `MESSAGING_CWD`** — don't let the agent operate from sensitive directories
8. **Run as non-root** — never run the gateway as root
9. **Monitor logs** — check `~/.hermes/logs/` for unauthorized access attempts
10. **Keep updated** — run `hermes update` regularly for security patches

### Securing API Keys

```bash
# Set proper permissions on the .env file
chmod 600 ~/.hermes/.env

# Keep separate keys for different services
# Never commit .env files to version control
```

### Network Isolation

For maximum security, run the gateway on a separate machine or VM:

```yaml
terminal:
  backend: ssh
  ssh_host: "agent-worker.local"
  ssh_user: "hermes"
  ssh_key: "~/.ssh/hermes_agent_key"
```

This keeps the gateway's messaging connections separate from the agent's command execution.
