---
sidebar_position: 12
title: "Pipe Script Output to Messaging Platforms"
description: "Send text from any shell script, cron job, CI hook, or monitoring daemon to Telegram, Discord, Slack, Signal, and other platforms using `hermes send`."
---

# Pipe Script Output to Messaging Platforms

`hermes send` is a small, scriptable CLI that pushes a message to any
messaging platform Hermes is already configured for. Think of it as a
cross-platform `curl` for notifications — you don't need a running
gateway, you don't need an LLM, and you don't need to re-paste bot tokens
into each of your scripts.

Use it for:

- System monitoring (memory, disk, GPU temp, long-running job finished)
- CI/CD notifications (deploy done, test failure)
- Cron scripts that need to ping you with results
- Quick one-shot messages from a terminal
- Piping any tool's output anywhere (`make | hermes send --to slack:#builds`)

The command reuses the same credentials and platform adapters that `hermes
gateway` already uses, so there's no second configuration surface to
maintain.

---

## Quick Start

```bash
# Plain text to the home channel for a platform
hermes send --to telegram "deploy finished"

# Pipe in stdout from anything
echo "RAM 92%" | hermes send --to telegram:-1001234567890

# Send a file
hermes send --to discord:#ops --file /tmp/report.md

# Attach a subject/header line
hermes send --to slack:#eng --subject "[CI] build.log" --file build.log

# Thread target (Telegram topic, Discord thread)
hermes send --to telegram:-1001234567890:17585 "threaded reply"

# List every configured target
hermes send --list

# Filter by platform
hermes send --list telegram
```

---

## Argument Reference

| Flag | Description |
|------|-------------|
| `-t, --to TARGET` | Destination. See [target formats](#target-formats). |
| `message` (positional) | Message text. Omit to read from `--file` or stdin. |
| `-f, --file PATH` | Read the body from a file. `--file -` forces stdin. |
| `-s, --subject LINE` | Prepend a header/subject line before the body. |
| `-l, --list` | List available targets. Optional positional platform filter. |
| `-q, --quiet` | No stdout on success (exit code only — ideal for scripts). |
| `--json` | Emit the raw JSON result of the send. |
| `-h, --help` | Show the built-in help text. |

### Target Formats

| Format | Example | Meaning |
|--------|---------|---------|
| `platform` | `telegram` | Send to the platform's configured home channel |
| `platform:chat_id` | `telegram:-1001234567890` | Specific numeric chat / group / user |
| `platform:chat_id:thread_id` | `telegram:-1001234567890:17585` | Specific thread or Telegram forum topic |
| `platform:#channel` | `discord:#ops` | Human-friendly channel name (resolved against the channel directory) |
| `platform:+E164` | `signal:+15551234567` | Phone-addressed platforms: Signal, SMS, WhatsApp |

Any platform Hermes ships adapters for works as a target:
`telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`,
`mattermost`, `feishu`, `dingtalk`, `wecom`, `weixin`, `email`, and
others.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Send (or list) succeeded |
| `1` | Delivery failed at the platform level (auth, permissions, network) |
| `2` | Usage / argument / config error |

Exit codes follow the standard Unix convention so your scripts can
branch on them the same way they would on `curl` or `grep`.

---

## Message Body Resolution

`hermes send` resolves the message body in this order:

1. **Positional argument** — `hermes send --to telegram "hi"`
2. **`--file PATH`** — `hermes send --to telegram --file msg.txt`
3. **Piped stdin** — `echo hi | hermes send --to telegram`

When stdin is a TTY (no pipe), Hermes does **not** wait for input — you'll
get a clear usage error instead. This keeps scripts from hanging if they
accidentally omit the body.

---

## Real-World Examples

### Monitoring: Memory / Disk Alerts

Replace ad-hoc `curl https://api.telegram.org/...` calls in your watchdogs
with a single portable line:

```bash
#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  hermes send --to telegram --subject "⚠ MEMORY WARNING" \
    "RAM ${ram_pct}% on $(hostname)"
fi
```

Because `hermes send` reuses your Hermes config, the same script works on
any host where Hermes is installed — no need to export bot tokens into
each machine's environment manually.

:::tip Don't alert the gateway about itself
For watchdogs that might fire when the gateway itself is struggling (OOM
alerts, disk-full alerts), keep using a minimal `curl` call instead of
`hermes send`. If the Python interpreter can't load because the box is
thrashing, you still want that alert to go out.
:::

### CI / CD: Build and Test Results

```bash
# In .github/workflows/deploy.yml or any CI script
if ./scripts/deploy.sh; then
  hermes send --to slack:#deploys "✅ ${CI_COMMIT_SHA:0:7} deployed"
else
  tail -n 100 deploy.log | hermes send \
    --to slack:#deploys --subject "❌ deploy failed"
  exit 1
fi
```

### Cron: Daily Report

```bash
# Crontab entry
0 9 * * * /usr/local/bin/generate-metrics.sh \
  | /home/me/.hermes/bin/hermes send \
      --to telegram --subject "Daily metrics $(date +%Y-%m-%d)"
```

### Long-Running Tasks: Ping When Done

```bash
./train.py --epochs 200 && \
  hermes send --to telegram "training done" || \
  hermes send --to telegram "training failed (exit $?)"
```

### Scripting with `--json` and `--quiet`

```bash
# Hard-fail a script if delivery fails; don't clutter logs on success
hermes send --to telegram --quiet "keepalive" || {
  echo "Telegram delivery failed" >&2
  exit 1
}

# Capture the message ID for later editing / threading
msg_id=$(hermes send --to discord:#ops --json "build started" \
  | jq -r .message_id)
```

---

## Does `hermes send` Need the Gateway Running?

**Usually no.** For any bot-token platform — Telegram, Discord, Slack,
Signal, SMS, WhatsApp Cloud API, and most others — `hermes send` calls
the platform's REST endpoint directly using credentials from
`~/.hermes/.env` and `~/.hermes/config.yaml`. It's a standalone subprocess
that exits as soon as the message is delivered.

A live gateway is only required for **plugin platforms** that rely on a
persistent adapter connection (for example, a custom plugin that keeps
a long-lived WebSocket open). In that case you'll get a clear error
pointing at the gateway; start it with `hermes gateway start` and retry.

---

## Listing and Discovering Targets

Before sending to a specific channel, you can inspect what's available:

```bash
# Every target across every configured platform
hermes send --list

# Just Telegram targets
hermes send --list telegram

# Machine-readable
hermes send --list --json
```

The listing is built from `~/.hermes/channel_directory.json`, which the
gateway refreshes every few minutes while it's running. If you see
"no channels discovered yet", start the gateway once (`hermes gateway
start`) so it can populate the cache.

Human-friendly names (`discord:#ops`, `slack:#engineering`) are resolved
against this cache at send time, so you don't need to memorize numeric
IDs.

---

## Comparison with Other Approaches

| Approach | Multi-platform | Reuses Hermes creds | Needs gateway | Best for |
|----------|----------------|---------------------|---------------|----------|
| `hermes send` | ✅ | ✅ | No (bot-token) | Everything below |
| Raw `curl` to each platform | Each scripted separately | Manual | No | Critical watchdogs |
| `cron` job with `--deliver` | ✅ | ✅ | No | Scheduled agent tasks |
| `send_message` agent tool | ✅ | ✅ | No | Inside an agent loop |

`hermes send` is intentionally the simplest possible surface. If you need
an agent to decide what to say, use the `send_message` tool from within a
chat or cron job. If you need a scheduled run with LLM-generated content,
use `cronjob(action='create', prompt=...)` with `deliver='telegram:...'`.
If you just need to pipe a raw string, reach for `hermes send`.

---

## Related

- [Automate Anything with Cron](/docs/guides/automate-with-cron) —
  scheduled jobs whose output auto-delivers to any platform.
- [Gateway Internals](/docs/developer-guide/gateway-internals) —
  the delivery router that `hermes send` shares with cron delivery.
- [Messaging Platform Setup](/docs/user-guide/messaging/) —
  one-time configuration for each platform.
