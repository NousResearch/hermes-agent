---
sidebar_position: 26
title: "Running Hermes on a Personal or Work Machine"
description: "A security-posture walkthrough for running Hermes Agent on the machine you live on — what the defaults protect, how to tighten further, and how to undo mistakes"
---

# Running Hermes on a Personal or Work Machine

You're about to run an agent on the machine you live on — a personal laptop or an employer-managed workstation. What's the safe posture?

Short answer: the defaults already do most of the work. Hermes ships secure-by-default, with a defense-in-depth model covering command approval, file-write safety, and credential handling. This page walks through what's on out of the box, which knobs to tighten for a shared or work machine, and how to undo mistakes when they happen. Every control here is covered in depth in the [Security](/user-guide/security) guide.

## What the Defaults Already Protect

Fresh install, no configuration — these protections are active:

**Dangerous commands require approval.** Before executing any command, Hermes checks it against a curated list of dangerous patterns — recursive deletes, writes to `/etc/`, disk operations, pipe-to-shell, and more. The default `approvals.mode: smart` uses an auxiliary LLM to assess risk: low-risk commands are auto-approved for that command only, genuinely dangerous commands are auto-denied, and uncertain cases escalate to a manual prompt.

**Approval prompts fail closed.** If you don't respond to an approval prompt within the timeout (default 300 seconds), the command is **denied**. Walking away from your desk never silently approves anything.

**A hardline blocklist is the always-on floor.** Some commands — `rm -rf /`, fork bombs, zeroing a physical disk — are refused **regardless** of approval mode, `--yolo`, or an explicit "allow always". The blocklist trips before the approval layer even sees the command, and there is no override flag.

**File writes to sensitive paths are blocked.** The `write_file` and `patch` tools cannot touch OS credential stores (`~/.ssh/`, `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `~/.netrc`), Hermes credential stores (`auth.json`, `.env`, pairing data), or project secret files (`.env`, `.env.local`, `.envrc`) anywhere on disk. Blocked writes return an error immediately — there is no approval prompt and no way to override from the chat UI.

**Secrets are redacted from output.** `security.redact_secrets` is on by default: patterns that look like API keys, tokens, and passwords in tool output are redacted before they enter the conversation context and logs.

**Your data goes only where you point it.** API calls go **only to the LLM provider you configure**. Hermes Agent does not collect telemetry, usage data, or analytics. Your conversations, memory, and skills are stored locally in `~/.hermes/`. See the [FAQ](/reference/faq#is-my-data-sent-anywhere).

:::info
There's more below the surface — SSRF protection on all URL-capable tools, filtered environments for MCP subprocesses, prompt-injection scanning of context files. The [Security](/user-guide/security) page documents every layer.
:::

## Tightening for a Shared or Work Machine

On a machine with employer data, production credentials, or other people's files, layer these on top of the defaults.

### Switch approvals to manual

`smart` mode auto-approves low-risk commands. If you want to see every flagged command yourself:

```yaml
approvals:
  mode: manual
```

Manual mode always prompts you before executing a flagged command.

### Add your own deny rules

`approvals.deny` is a list of glob patterns that block matching terminal commands unconditionally — even under `--yolo`, `/yolo`, or `mode: off`. It's the user-editable counterpart to the built-in hardline blocklist. Use it to declare things that must never run on this machine:

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

Patterns are case-insensitive [fnmatch](https://docs.python.org/3/library/fnmatch.html) globs matched against the whole command text, and matching runs over the same normalized/deobfuscated variants the dangerous-pattern detector uses, so simple quoting tricks don't slip past a rule. Always quote patterns — a bare leading `*` is a YAML parse error. Changes take effect immediately, no restart needed. Details: [User-Defined Deny Rules](/user-guide/security#user-defined-deny-rules-approvalsdeny).

### Sandbox file writes

`HERMES_WRITE_SAFE_ROOT` restricts `write_file` and `patch` to the directory prefix(es) you list — anything outside is hard-blocked. Multiple roots are separated by `:` on Unix:

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

Sensitive paths inside the safe root are still blocked — pointing it at `$HOME` does not allow writing `~/.ssh/id_rsa`.

:::caution
Don't add this to `~/.hermes/.env` casually. If you set it to a project directory only, the agent cannot write to `~/.hermes/cron/jobs.json`, profile skills, or other Hermes state outside that prefix. Include your Hermes home as a second root, as above.
:::

### Move command execution off the host

The strongest isolation is not running commands on your machine at all. The terminal tool supports multiple [backends](/user-guide/features/tools#terminal-backends):

| Backend | Isolation |
|---------|-----------|
| `local` | None — runs on host (dangerous-command checks apply) |
| `docker` | Container — the container itself is the security boundary |
| `ssh` | Remote machine — keeps execution on a separate server |

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
```

Every Docker container runs with hardened settings — all Linux capabilities dropped (with a minimal add-back set), `no-new-privileges`, a process-count limit, and size-limited tmpfs mounts. With a container backend, destructive commands inside the container can't harm the host, which is why dangerous-command checks are skipped there.

For `ssh`, set `terminal.backend: ssh` in `config.yaml` and provide host details via `TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER`, and `TERMINAL_SSH_KEY` in `~/.hermes/.env`. See [Network Isolation](/user-guide/security#network-isolation).

### If messaging is on: allowlists and pairing

Running the [gateway](/user-guide/security#user-authorization-gateway) on this machine? The default is already deny: if no allowlists are configured and `GATEWAY_ALLOW_ALL_USERS` is not set, **all users are denied**. Keep it explicit:

```bash
# ~/.hermes/.env
TELEGRAM_ALLOWED_USERS=123456789
GATEWAY_ALLOWED_USERS=123456789
```

Or use DM pairing instead of hardcoding IDs: unknown users receive a one-time pairing code, and you approve them from the CLI with `hermes pairing approve <platform> <code>`. Never set `GATEWAY_ALLOW_ALL_USERS=true` on a machine you care about.

## The Undo Layer: Checkpoints and `/rollback`

Approval gates prevent damage; [checkpoints](/user-guide/checkpoints-and-rollback) reverse it. When enabled, Hermes automatically snapshots your project before destructive operations — `write_file`, `patch`, and destructive terminal commands like `rm`, `mv`, `sed -i`, and `git reset` — into a shadow git store under `~/.hermes/checkpoints/store/`. Your real project `.git` is never touched.

Checkpoints are opt-in. Enable per-session:

```bash
hermes chat --checkpoints
```

Or globally:

```yaml
checkpoints:
  enabled: true
```

Then, in a session:

| Command | Description |
|---------|-------------|
| `/rollback` | List all checkpoints with change stats |
| `/rollback diff <N>` | Preview what changed since checkpoint N |
| `/rollback <N>` | Restore to checkpoint N (also undoes the last chat turn) |
| `/rollback <N> <file>` | Restore a single file from checkpoint N |

:::tip
Preview with `/rollback diff <N>` before restoring, and combine checkpoints with git worktrees for maximum safety — each Hermes session in its own worktree, with checkpoints as an extra layer.
:::

## What This Threat Model Is — and Isn't

Be clear-eyed about what these controls defend against. As the [Security](/user-guide/security#user-defined-deny-rules-approvalsdeny) guide puts it:

> Deny rules are a guardrail against an honest-but-wrong agent, the same threat model as the dangerous-pattern detector. They are not a sandbox against a deliberately adversarial process — for that, use an isolated backend (Docker, Modal) or an egress-restricted environment.

The same applies to the file-write guards: they apply to `write_file` and `patch` only, while the `terminal` tool runs as the same OS user. The denylist reduces accidental damage and gives models a clear stop signal; it does not sandbox a hostile or compromised agent. If your requirement is containment rather than guardrails, the answer is an isolated terminal backend — that's the boundary designed for it.

## A Cautious Starting Config

Everything above, assembled. Adjust to taste in `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: manual                  # See every flagged command yourself
  timeout: 300                  # Unanswered prompts are denied (fail-closed)
  deny:                         # Never-run list — survives even /yolo
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"

security:
  redact_secrets: true          # Already the default; stated here for clarity

checkpoints:
  enabled: true                 # Snapshot before destructive operations

terminal:
  backend: docker               # Or ssh — keep execution off the host
  docker_forward_env: []        # No host secrets inside the container
```

And in `~/.hermes/.env`, if you want the write sandbox:

```bash
HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

## See Also

- **[Security](/user-guide/security)** — the full defense-in-depth reference: every approval pattern, container hardening flags, gateway authorization, MCP credential filtering
- **[Checkpoints & Rollback](/user-guide/checkpoints-and-rollback)** — configuration, store maintenance, and restore workflows
- **[Tools & Toolsets](/user-guide/features/tools)** — all terminal backends and their configuration
- **[Configuration](/user-guide/configuration)** — the complete `config.yaml` reference
