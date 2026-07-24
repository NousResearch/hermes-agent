---
sidebar_position: 24
title: "Carbon Voice"
description: "Set up Hermes Agent as a Carbon Voice bot (voice-first messaging via Socket.IO)"
---

# Carbon Voice Setup

Hermes connects to [Carbon Voice](https://carbonvoice.app/), a voice-first
messaging platform. Carbon Voice transcribes voice messages to text (STT)
before delivery and can re-synthesize the agent's replies as voice memos
(TTS), so from Hermes' side it is a **text-in / text-out** platform.

The adapter connects over **Socket.IO** (primary) with a **REST polling
fallback** — no public webhook or tunnel required. It persists a cursor to
disk, so messages received while Hermes was offline are processed on the next
startup.

:::info Dependencies
The adapter uses `httpx` (already a core Hermes dependency). For real-time
Socket.IO delivery, install `python-socketio`:

```bash
pip install 'python-socketio[asyncio_client]'
```

Without it, the adapter still works in polling-only mode.
:::

---

## Prerequisites

- A **Carbon Voice account** and a **Personal Access Token** (`cv_pat_...`),
  created at [developer.carbonvoice.app](https://www.developer.carbonvoice.app/).

---

## Setup

Run the gateway setup wizard and pick **Carbon Voice**:

```bash
hermes gateway setup
```

Or set the environment variable directly in `~/.hermes/.env`:

```bash
CARBONVOICE_PAT=cv_pat_xxxxxxxxxxxxxxxx
```

The platform auto-enables whenever `CARBONVOICE_PAT` is present. Start the
gateway:

```bash
hermes gateway run
```

You should see `✓ carbonvoice connected` in the logs.

---

## Access control (deny-by-default)

Access is **deny-by-default**. A user may reach the agent only if **any** of:

1. They are the **owner** — the Carbon Voice user who created the bot account
   (`whoami.created_by`). Auto-detected at startup; always allowed, no setup
   needed.
2. They are listed in `CARBONVOICE_ALLOWED_USERS` (comma-separated `user_guid`s).
3. The owner approved them at runtime.

### Interactive onboarding

When an unauthorized user messages the bot, it asks the **owner** in the home
channel (`CARBONVOICE_HOME_CHANNEL`):

> 👤 *Teammate (Abc123…) wants to talk to me but isn't authorized.*
> *React 💯 to allow · 👎 to block — or reply* `/cv-allow-user Abc123…`

**One-tap approval:** just **react 💯** on that prompt to allow, or **👎** to
block — no typing. Only the owner's reaction counts, so a stranger can't
self-approve. Text commands work too (owner-only, in the home channel):
`/cv-allow-user <id>`, `/cv-deny-user <id>`, `/cv-list-allow-users`.

To open access entirely (not recommended), set
`CARBONVOICE_ALLOW_ALL_USERS=true`.

---

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `CARBONVOICE_PAT` | _(required)_ | Personal Access Token (`cv_pat_...`). |
| `CARBONVOICE_BASE_URL` | `https://api.carbonvoice.app` | API base URL. |
| `CARBONVOICE_ALLOWED_USERS` | _(unset)_ | Extra allowed `user_guid`s, beyond the owner. |
| `CARBONVOICE_ALLOW_ALL_USERS` | `false` | Disable gating (open access). |
| `CARBONVOICE_HOME_CHANNEL` | _(unset)_ | Channel for cron delivery + approving unknown senders. |
| `CARBONVOICE_APPROVAL_COOLDOWN_S` | `1800` | Min seconds between owner-approval prompts per unknown user. |
| `CARBONVOICE_APPROVE_REACTION_ID` | `affirmative` | Reaction the owner taps to allow (💯). |
| `CARBONVOICE_REJECT_REACTION_ID` | `negative` | Reaction the owner taps to block (👎). |
| `CARBONVOICE_STUCK_MAX_AGE_S` | `300` | How long a transcript-less message is retried before being skipped. |
| `CARBONVOICE_SEND_DEDUP_WINDOW_S` | `90` | Drop an identical outbound reply to the same channel within this window. |
| `CARBONVOICE_REQUIRE_MENTION` | `true` | In group channels, only respond when @-mentioned (DMs always pass). |
| `CARBONVOICE_VOICE_OUT` | `false` | Auto-convert text replies to voice memos via Hermes' TTS pipeline. |

---

## Notes

- **Voice in/out:** inbound voice is transcribed by Carbon Voice before Hermes
  sees it. To reply with voice memos, set `CARBONVOICE_VOICE_OUT=true` and
  configure a TTS provider (`voice.auto_tts: true` in `config.yaml`). Carbon
  Voice transcribes the outgoing audio server-side and renders the transcript
  inline with the voice memo, so the user gets **one bubble** — audio plus
  text together, never a duplicate text message.
- **Images:** inbound image attachments are downloaded and forwarded to the
  agent's vision pipeline.
- **Cron delivery:** set `CARBONVOICE_HOME_CHANNEL` and deliver cron results
  with `carbonvoice:<channel_guid>` (or `all`).
