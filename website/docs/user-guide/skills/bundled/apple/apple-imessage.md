---
title: "Imessage — Send and receive iMessages/SMS via the imsg CLI on macOS"
sidebar_label: "Imessage"
description: "Send and receive iMessages/SMS via the imsg CLI on macOS"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Imessage

Send and receive iMessages/SMS via the imsg CLI on macOS.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/apple/imessage` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos |
| Tags | `iMessage`, `SMS`, `messaging`, `macOS`, `Apple` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# iMessage

Use `imsg` to read and send iMessage/SMS via macOS Messages.app.

## Prerequisites

- **macOS** with Messages.app signed in
- Install: `brew install steipete/tap/imsg`
- Grant Full Disk Access for terminal (System Settings → Privacy → Full Disk Access)
- Grant Automation permission for Messages.app when prompted
- Running through the gateway as a launchd service? The permissions belong to the gateway's Python interpreter, not to Terminal — see [Pitfalls](#pitfalls)

## When to Use

- User asks to send an iMessage or text message
- Reading iMessage conversation history
- Checking recent Messages.app chats
- Sending to phone numbers or Apple IDs

## When NOT to Use

- Telegram/Discord/Slack/WhatsApp messages → use the appropriate gateway channel
- Group chat management (adding/removing members) → not supported
- Bulk/mass messaging → always confirm with user first

## Quick Reference

### List Chats

```bash
imsg chats --limit 10 --json
```

### View History

```bash
# By chat ID
imsg history --chat-id 1 --limit 20 --json

# With attachments info
imsg history --chat-id 1 --limit 20 --attachments --json
```

### Send Messages

```bash
# Text only
imsg send --to "+14155551212" --text "Hello!"

# With attachment
imsg send --to "+14155551212" --text "Check this out" --file /path/to/image.jpg

# Force iMessage or SMS
imsg send --to "+14155551212" --text "Hi" --service imessage
imsg send --to "+14155551212" --text "Hi" --service sms
```

### Verify a Send

`imsg send` only reports that Messages.app accepted the message. To confirm it was
recorded in the conversation, read the newest entry of that chat back:

```bash
imsg history --chat-id 1 --limit 1 --json
# expect "is_from_me": true and the text you just sent
```

This proves the message is in `chat.db`, not that the recipient's device received it —
`imsg` exposes no delivery receipt. Report it to the user in those terms.

### Watch for New Messages

```bash
imsg watch --chat-id 1 --attachments
```

## Service Options

- `--service imessage` — Force iMessage (requires recipient has iMessage)
- `--service sms` — Force SMS (green bubble)
- `--service auto` — Let Messages.app decide (default)

## Rules

1. **Always confirm recipient and message content** before sending
2. **Never send to unknown numbers** without explicit user approval
3. **Verify file paths** exist before attaching
4. **Don't spam** — rate-limit yourself

## Pitfalls

### Permissions under the launchd gateway

When the gateway runs as a launchd service (`hermes gateway install` on macOS), the
agent's terminal is a child of the gateway's Python process, not of Terminal.app.
macOS attributes privacy permissions to that interpreter, so granting Full Disk
Access to Terminal does not help. Symptoms:

- `imsg chats` fails with `authorization denied (code: 23)`
- `imsg send` hangs for several minutes, then fails with
  `applescript send failed (not_started): AppleScript error -1712`

Fix: add the gateway's interpreter (for a git install
`~/.hermes/hermes-agent/venv/bin/python`) under System Settings → Privacy & Security
to **Full Disk Access** and to **Automation** (allow Messages), then restart the
gateway with `hermes gateway restart` — permissions are read at process start.

### `imsg chats --json` prints one object per line

The output is newline-delimited JSON, not an array, and the field is `display_name`.
Filter it with `select(...)` directly; `.[]` fails with `Cannot index number with string`.

## Example Workflow

User: "Text mom that I'll be late"

```bash
# 1. Find mom's chat
imsg chats --limit 20 --json | jq -c 'select((.display_name // "") | contains("Mom"))'

# 2. Confirm with user: "Found Mom at +1555123456. Send 'I'll be late' via iMessage?"

# 3. Send after confirmation
imsg send --to "+1555123456" --text "I'll be late"

# 4. Verify it landed in the conversation (see "Verify a Send")
imsg history --chat-id <id from step 1> --limit 1 --json
```
