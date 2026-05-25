# Telegram / Comms Policy MCP server

Hermes ships a small stdio MCP server for Brandon-style notification policy:
concise Telegram formatting, SQLite-backed dedupe, and stale-alert suppression.
It delegates actual delivery to Hermes' existing `send_message` tool, so Telegram
secrets, home-channel routing, and platform adapters stay in the normal gateway
configuration instead of being duplicated in MCP config.

## Tools

When configured under the server name `comms_policy`, Hermes discovers these as
`mcp_comms_policy_*` tools after restart:

- `send_success_notice` — format, dedupe, and send a concise success notice.
- `send_failure_notice` — format, dedupe, and send a concise failure / Kanban
  error notice.
- `send_briefing` — format, dedupe, and send a concise briefing.
- `dedupe_notification` — check whether a message is duplicate or stale.
- `check_last_sent` — inspect durable dedupe state for a target/key/message.
- `queue_notification` — store an allowed notification without sending yet.
- `format_ready_question_notice` — render the ready-question convention without
  sending.

## Configuration

Add this to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  comms_policy:
    command: python
    args: ["-m", "mcp_servers.comms_policy"]
    timeout: 30
    connect_timeout: 30
```

If Hermes is installed in a virtual environment and `python` does not resolve to
that environment, set `command` to the absolute venv interpreter path instead.
No Telegram token or chat ID belongs in this snippet; delivery uses the existing
Hermes gateway/send_message configuration.

Restart Hermes, then check discovery:

```bash
hermes mcp list
hermes mcp test comms_policy
```

In a new agent session, the discovered tools are available with the
`mcp_comms_policy_` prefix.

## State

Dedupe state is stored in a profile-scoped SQLite database:

```text
$HERMES_HOME/comms_policy/notifications.sqlite
```

Rows record target, category, dedupe key, SHA-256 message hash, status
(`sent`, `queued`, `suppressed`, or `failed`), event timestamp, queued time, sent
time, and JSON metadata. Repeated identical notifications within the dedupe
window are suppressed, and notifications whose event timestamp is older than
`stale_after_seconds` are suppressed as stale.

## Formatting conventions

Success:

```text
✅ Backup complete
Task: t_123
• 14 files copied
Next: Verify offsite sync
```

Failure / Kanban error:

```text
❌ Kanban worker failed
Task: t_123
Error: pytest timed out
Next: Retry with a narrower test target
```

Ready question:

```text
❓ Which target should receive the briefing?
• Telegram home
• Discord weekly review
Reply with the choice or details.
```
