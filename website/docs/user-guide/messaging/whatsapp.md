---
sidebar_position: 5
title: "WhatsApp"
description: "Choose WhatsApp Agent Platform, the Baileys phone bridge, or Business Cloud API, then set up Hermes"
---

# WhatsApp Setup

Hermes has three WhatsApp connection routes. Choose the one that matches what you already have before entering credentials or scanning a QR code.

## Choose your integration

| What you have or want | Choose in Hermes | Start here |
|---|---|---|
| An agent created in **WhatsApp → Settings → Agents**, with its API key | **WhatsApp Agent Platform** (`whatsapp_agent_platform`), a community plugin | [Terminal and Desktop setup](./whatsapp-agent-platform.md). No linked-device QR, business number, or public webhook. |
| A WhatsApp **phone account** you want to link, either a dedicated bot account or Message Yourself | **WhatsApp** (`whatsapp`), the bundled Baileys bridge | [Phone-account setup below](#two-modes). Uses **Linked Devices** and a terminal QR code. |
| A business phone number and Meta Business API credentials | **WhatsApp Business Cloud API** (`whatsapp_cloud`) | [Business Cloud API setup](./whatsapp-cloud.md). Requires a public HTTPS webhook. |

:::tip Already have an Agent API key?
Follow the [Agent Platform guide](./whatsapp-agent-platform.md). Install and enable `whatsapp-agent-platform` before selecting **WhatsApp Agent Platform** in `hermes gateway setup` or Desktop Messaging. Choosing ordinary **WhatsApp** starts phone linking instead.
:::

Meta supports both an [Agent Platform API](https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf) and a separate Business Cloud API. The remaining sections on this page describe the **Baileys phone bridge**. Its QR pairing, session files, groups, and streaming behavior do not apply to Agent Platform.

:::warning Baileys is an unofficial WhatsApp Web bridge
Baileys emulates a linked WhatsApp Web session. Account restrictions and protocol breakage remain possible. Use a dedicated account for a bot, avoid bulk or unsolicited messaging, and follow WhatsApp's rules. Official API access also remains subject to Meta's policies; it does not guarantee immunity from restrictions. See the [Business Messaging Policy](https://whatsappbusiness.com/policy/) and [third-party agent terms](https://www.whatsapp.com/legal/third-party-agents-terms).
:::

## Before changing settings

Work on the same **machine, backend, and profile** that will run the gateway. Find the active files instead of assuming `~/.hermes`:

```bash
hermes config path
hermes config env-path
```

On Windows, the default home is `%LOCALAPPDATA%\hermes`; on Unix it is usually `~/.hermes`. `HERMES_HOME` and [named profiles](../profiles.md) can change these paths. In this guide, **`HERMES_HOME` means the active profile's home**.

For a named profile, add the selector to every command, for example `hermes -p work whatsapp`, `hermes -p work config path`, and `hermes -p work gateway status`.

## Two Modes

| Baileys mode | Where you send a message to Hermes | Account to link |
|---|---|---|
| **Separate bot number** | From your own account, send a DM to the bot's number | A dedicated, registered WhatsApp phone account |
| **Personal self-chat** | Open **Message Yourself** and send a message there | Your personal WhatsApp account; other conversations do not become Hermes chats |

### Prepare the account first {#step-2-getting-a-second-phone-number-bot-mode}

For bot mode, use an existing dedicated WhatsApp account or register a separate mobile number **before** scanning the QR. WhatsApp Business **the phone app** can run alongside personal WhatsApp; using that app for a linked-device bot does not turn it into the Business Cloud API.

Use a number that WhatsApp supports and that you can retain and recover. WhatsApp's [registration help](https://faq.whatsapp.com/684051319521343) lists VoIP numbers as unsupported, so Google Voice, TextNow, and similar services are not reliable supported setup choices. A prepaid SIM can work, but cost, refill requirements, and number retention depend on the carrier and plan; a periodic call alone does not guarantee retention.

For self-chat, no second number is needed. Confirm that you intend to link your personal account, then use Message Yourself for Hermes conversations.

## Prerequisites

- A phone with the **intended WhatsApp account already registered**, for scanning the QR.
- A Node.js runtime and bridge dependencies. The setup wizard can prepare Hermes's managed runtime. **Node 18 is not supported by the current bridge dependencies**; when supplying your own runtime, use the versions accepted by the [package-management guide](../../reference/package-management.md) and bridge lockfile.
- A working Hermes model/provider for replies.

The Baileys bridge does not require Chromium or Puppeteer.

## Step 1: Run the Setup Wizard

```bash
hermes whatsapp
```

`hermes gateway setup` → **WhatsApp** also opens the phone-linking setup. The dedicated `hermes whatsapp` command is the direct terminal route.

1. Choose **bot** or **self-chat** to match the account prepared above.
2. Set the permitted sender phone numbers when prompted: include the country code, without `+`, spaces, parentheses, or dashes (for example `15551234567`).
3. Let the wizard prepare the bridge dependencies and display the QR code.
4. On the **account you intend to link**, open **Settings → Linked Devices → Link a Device**, then scan the terminal QR.
5. Wait for pairing to complete and the session to be saved.

If you already have a session, the wizard may ask whether to **re-pair**. Keeping it preserves the existing login; deliberately re-pairing replaces it. See [Re-pairing](#re-pairing).

:::tip QR display
Use a terminal at least 60 columns wide with Unicode support. If the code is garbled, try another terminal. Scan from the bot account for bot mode, or your personal account for self-chat.
:::

## Step 2: Configure Hermes {#step-3-configure-hermes}

The wizard saves the mode and enables the Baileys integration after pairing. Check the active profile's enablement before starting the gateway: an explicit `platforms.whatsapp.enabled: false` in `config.yaml` still disables it even if the legacy `WHATSAPP_ENABLED=true` flag is present. A legacy `WHATSAPP_ENABLED=false` also disables it.

### Choose who can use the bot

The wizard can save `WHATSAPP_ALLOWED_USERS` for you. For a private bot, list only the people you intend to admit. Existing `WHATSAPP_ALLOWED_USERS=*` or `WHATSAPP_ALLOW_ALL_USERS=true` settings open access to all senders; use them only for an intentional public bot. [Access-control reference](../security.md#dm-pairing-system).

Behavior can be configured in the `config.yaml` printed above. For example, this restricts DMs to listed numbers and makes unauthorized DMs silent:

```yaml
whatsapp:
  dm_policy: allowlist
  allow_from:
    - "15551234567"
  unauthorized_dm_behavior: ignore
```

Preserve the rest of your configuration. Review any existing environment overrides before changing policy, since legacy environment settings can take precedence.

To approve senders individually, use the default **DM pairing** policy instead. An unknown sender gets a Hermes approval code; **the operator approves it in the matching profile's terminal**:

```bash
hermes pairing list
hermes pairing approve whatsapp CODE_FROM_REQUEST
```

Replace `CODE_FROM_REQUEST` with the pending code. This is permission to talk to Hermes, not a Linked Devices QR code or an Agent API key. Previously approved senders remain authorized alongside configured allowlists. In self-chat mode, the bridge restricts intake to your own self-chat.

### Group chats (bot mode)

Groups are gated by **group policy**, not by the DM allowlist. `WHATSAPP_GROUP_POLICY` / `whatsapp.group_policy`
defaults to `pairing`, which forwards nothing from groups. `allowlist` plus `WHATSAPP_GROUP_ALLOWED_USERS` /
`whatsapp.group_allow_from` (comma-separated **group JIDs**, e.g. `120363001234567890@g.us`) admits the listed
groups; `open` admits every group the bot is a member of. The sender is then checked like any other gateway
principal: with `WHATSAPP_ALLOWED_USERS` set, a participant must be on it (or paired) — a sender WhatsApp
addresses by LID matches through the phone number Baileys supplies alongside it, so a first contact with no
`lid-mapping` file yet is not dropped; with no sender allowlist,
`allowlist` trusts the group-JID list alone and admits every participant of a listed group, while `open` still
needs the participant paired or `WHATSAPP_ALLOW_ALL_USERS=true`. By default the bot answers every admitted group
message; set `require_mention: true` / `WHATSAPP_REQUIRE_MENTION=true` to answer only @mentions, replies to the
bot, or `/commands` (groups in `free_response_chats` are exempt).

### Connect and send the first message

Run `hermes gateway status` first. If a gateway already serves this profile, restart that existing gateway using its owner: Desktop's restart control, `hermes gateway restart` for a CLI-managed service, or restarting your foreground command. If none is running:

```bash
hermes gateway
```

Keep the terminal open. The gateway starts the bridge using the saved session. For persistent operation, follow [Service Management](./index.md#service-management): `hermes gateway install` installs a user service, and `hermes gateway start` starts it. Windows uses its Scheduled Task/Startup integration; `sudo hermes gateway install --system` is a Linux-only alternative.

Wait for **WhatsApp connected**, then test the matching mode:

- **Bot:** send a new DM to the bot number from an allowed or approved account.
- **Self-chat:** open **Message Yourself** and send a new message there.

Confirm a reply. A saved session, enabled switch, or configured indicator alone does not prove the gateway is connected and your model can answer.

## Set up from Hermes Desktop

1. Select the backend and profile that will own WhatsApp.
2. Use a terminal **on that backend**, with the same profile, to run `hermes whatsapp` and complete QR pairing above. The current Desktop Messaging form does not display a WhatsApp Linked Devices QR.
3. Open **Messaging → WhatsApp**, check the mode and sender access settings, save changes, and enable the platform.
4. Use **Restart now** when requested, wait for the live connection state, and send the first-message test above.

The Desktop chat backend and messaging gateway are separate processes. If you have an **Agent API key**, use [WhatsApp Agent Platform in Desktop](./whatsapp-agent-platform.md#set-up-from-hermes-desktop) instead of this phone-linking route.

## Session Persistence

Sessions survive gateway restarts; you normally do not need another QR scan. Hermes supports both the legacy **`HERMES_HOME/whatsapp/session`** used by the CLI wizard and **`HERMES_HOME/platforms/whatsapp/session`** used by newer adapter layouts. The adapter reuses a populated legacy session, and an explicit `session_path` can select another directory. Check your active profile's paths and gateway diagnostics before assuming a missing directory means you are unpaired.

These files contain device credentials and encryption keys. Do not share or commit them. For containers, persist the active session directory in a volume. See [Multiple profiles](#multiple-profiles) for port and session ownership.

## Re-pairing

Temporary network interruptions are handled by reconnection. A protocol update may require a Hermes/bridge update; it does not automatically require deleting valid credentials.

If the device was unlinked, the phone account was reset, or the session is invalid, stop the gateway that owns it and run:

```bash
hermes whatsapp
```

When an existing session is found, choose **yes** at the **Re-pair?** prompt to clear that login and generate a new QR. The default **no** keeps it and returns. Scan using the intended account, then restart the owning gateway and test a fresh message.

## Voice Messages

Hermes supports voice on WhatsApp:

- **Incoming:** Voice messages (`.ogg` opus) are automatically transcribed using the configured STT provider: local `faster-whisper`, Groq Whisper (`GROQ_API_KEY`), or OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)
- **Outgoing:** TTS responses are sent as MP3 audio file attachments
- **Self-chat replies** use the "☤ **Hermes Agent**" prefix by default; separate-number bot replies do not. Customize or disable the self-chat prefix in the active `config.yaml`:

```yaml
# Active profile config.yaml (hermes config path)
whatsapp:
  reply_prefix: ""                          # Empty string disables the header
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # Custom prefix (supports \n for newlines)
  send_read_receipts: false                 # Mark accepted inbound messages as read (blue ticks)
```

When `send_read_receipts` is `true`, the adapter marks policy-accepted inbound messages as read after DM/group/mention filtering passes. Rejected messages (e.g., from non-allowlisted senders) are not marked read. Disabled by default for privacy. Changing this setting automatically restarts the bridge subprocess on the next connection.

---

## Message Formatting & Delivery

The **Baileys bridge** supports **streaming (progressive) responses** — the bot edits its message in real-time as the AI generates text, just like Discord and Telegram. Internally, WhatsApp is classified as a TIER_MEDIUM platform for delivery capabilities.

### Chunking

Long responses are automatically split into multiple messages at **4,096 characters** per chunk (WhatsApp's practical display limit). You don't need to configure anything — the gateway handles splitting and sends chunks sequentially.

### WhatsApp-Compatible Markdown

Standard Markdown in AI responses is automatically converted to WhatsApp's native formatting:

| Markdown | WhatsApp | Renders as |
|----------|----------|------------|
| `**bold**` | `*bold*` | **bold** |
| `~~strikethrough~~` | `~strikethrough~` | ~~strikethrough~~ |
| `# Heading` | `*Heading*` | Bold text (no native headings) |
| `[link text](url)` | `link text (url)` | Inline URL |

Code blocks and inline code are preserved as-is since WhatsApp supports triple-backtick formatting natively.

### Tool Progress

When the agent calls tools (web search, file operations, etc.), WhatsApp displays real-time progress indicators showing which tool is running. This is enabled by default — no configuration needed.

### Native Polls, Clarify-as-Poll, and Locations

The Baileys-bridge adapter (bot mode) supports several native WhatsApp message types:

- **Polls** — the agent can send a native WhatsApp poll (question + options) via the bridge's `/send-poll` endpoint. Poll votes flow back into the conversation.
- **Clarify questions as polls** — when the agent asks a multiple-choice clarify question, it's rendered as a native single-select poll; tapping an option answers the question. If the poll fails to send, the adapter falls back to a plain text question. Approval prompts are **never** mapped onto polls — polls are only used for genuine multiple-choice clarifies.
- **Location pins** — the agent can send a native location pin (latitude/longitude, optional name/address) via `/send-location`, and incoming shared locations (including live locations) are delivered to the agent as location messages.

All of this works out of the box in bot (Baileys) mode; no configuration needed.

### Message Batching (Debounce)

WhatsApp delivers each message individually, so a rapid burst (forwarded batches, paste-splits, multi-line text) would otherwise trigger a separate agent invocation per fragment — wasting tokens and producing several disjointed replies. The adapter buffers successive text messages from the same chat and dispatches them as one combined request after a short quiet period (default **0.3s**, extended to **1s** for very long fragments; capped at 2s / 4s). Tune via `config.yaml`:

```yaml
# Active profile config.yaml (hermes config path)
gateway:
  platforms:
    whatsapp:
      extra:
        text_batch_delay_seconds: 0.3         # quiet period before flushing a batch (max 2.0)
        text_batch_split_delay_seconds: 1.0   # extended delay near the split threshold (max 4.0)
```

Set `text_batch_delay_seconds: 0` to dispatch each message immediately (disables batching).

### Quoted Replies

Replying to (quoting) an earlier message gives the agent the quoted text as context. Quoting an image, voice note, video or document also attaches that file to the turn, so "what is this?" under a quoted image works — whether the attachment came from another person or from the bot itself (a cron-delivered chart, a generated image). WhatsApp only ships a thumbnail stub with a quote, so the file is resolved from the bridge's download cache (inbound media, in-memory for the bridge's lifetime) or from a local index of the bot's own sends (last 1000 messages); quotes of anything older arrive without the attachment.

---

## Troubleshooting

If you selected **Agent Platform** or **Cloud API**, use those guides' troubleshooting sections. QR and phone-session fixes below apply to **Baileys**.

| Problem | What to check |
|---------|---------------|
| **QR code not scanning** | Use a wide Unicode terminal and scan from the intended account under **Linked Devices**. |
| **QR code expires** | Wait for the refreshed QR or rerun `hermes whatsapp` if setup has timed out. |
| **Session not persisting** | Confirm the active home/profile, the legacy or newer session path, and any `session_path` override. Persist that directory in containers. |
| **Logged out unexpectedly** | Linked devices can work with the primary phone offline, but WhatsApp logs them out if the primary phone is unused for over 14 days. Open WhatsApp on the primary phone regularly; re-pair if the link was removed. [WhatsApp linked-device help](https://faq.whatsapp.com/378279804439436/?cms_platform=android&locale=en_US). |
| **Bridge crashes or reconnect loops** | Read the gateway/bridge error, check Node and dependencies, then update Hermes if protocol compatibility changed. Re-pair only if the login is invalid. |
| **Enabled in the wizard, disabled in Desktop** | Check the same backend/profile and `platforms.whatsapp.enabled`; an explicit YAML disable survives the legacy enable flag. |
| **macOS: Node works in a terminal, not the service** | launchd does not inherit your shell PATH. Reinstall the gateway service to refresh its PATH and start it. See [macOS launchd](./index.md#macos-launchd). |
| **Connected, but messages are ignored** | Check bot vs self-chat mode, sender allowlist/approval, group and mention policy, and the selected profile. If intake succeeds, check the model/provider for generation errors. |
| **Bot sends pairing codes to strangers** | Set `whatsapp.unauthorized_dm_behavior: ignore` in the active `config.yaml` for silent rejection. Review existing DM approvals as well as allowlists. |

For bridge diagnostics, `WHATSAPP_DEBUG=true` in the active profile's `.env` enables raw events in `bridge.log` after restart. Those events can include personal data: disable debugging afterward and redact logs before sharing them.

## Security

Configure sender access before going live. Under the default DM policy, unknown senders cannot start an agent turn until allowlisted or approved, but may receive an approval-code reply. An empty allowlist does not revoke earlier pairing approvals. Use `unauthorized_dm_behavior: ignore` if a private bot should stay silent to strangers; `*` and allow-all flags are explicit public-access choices.

- Protect the **actual session directory** like a password; it grants access to the linked account. Keep it out of public repositories, screenshots, and shared diagnostics.
- Use a dedicated phone account for a separate-number bot, and keep that number active and recoverable.
- If the link is compromised or unwanted, remove it in **WhatsApp → Settings → Linked Devices**.
- Review pending/approved Hermes senders with `hermes pairing list` in the owning profile.
- Review log retention; even partially redacted identifiers and message traces can contain personal information.

## Multiple profiles

The host multiplexer can serve a separate paired WhatsApp session for each profile.
Run `hermes -p work whatsapp` to pair a secondary profile, then enable WhatsApp
for that profile. An enabled profile without `creds.json` is skipped with the
`whatsapp_unpaired` status and its pairing command.

An explicit `platforms.whatsapp.extra.bridge_port` takes precedence. Otherwise,
a secondary selects the first free port in 3001 to 3999 that no other profile's
record claims, and saves it in its own `platforms/whatsapp/bridge_port` file for
subsequent starts. Operators can pre-create that file with a port number; delete
it to have a new port allocated. The launch profile uses port 3000 unless it
already has that file (from serving as a secondary), in which case its gateway
and `hermes send --to whatsapp:<chat_id>` both keep using the recorded port.

A secondary adopts a bridge already running on its port only when its own
session pidfile identifies that process (pid, kernel start time, and the port it
was started on), which is
what a gateway crash leaves behind. An unhealthy one is reaped by that same
identity and restarted. Any other process bound on the port is a fatal error
for that profile only. Set `platforms.whatsapp.extra.bridge_port` to a
distinct free port, or stop the process holding it. Other
profiles continue running. `hermes gateway status --profile work` reports the
profile's own WhatsApp adapter rather than shared ingress.

Profiles that each run their own gateway, rather than one multiplexed gateway,
all use port 3000 unless configured otherwise. Give each one a distinct port in
that profile's `config.yaml`:

```yaml
platforms:
  whatsapp:
    extra:
      bridge_port: 3001        # one distinct port per profile
```

A gateway identifies a running bridge by the session directory the bridge
reports in `/health`. A bridge serving another profile's session is never
adopted and never stopped: the second profile's WhatsApp fails to start with
`whatsapp_bridge_foreign_session`, naming the port and the other session. A
process that holds the port but does not answer `/health` in time is left
running too, and WhatsApp fails with the retryable
`whatsapp_bridge_unresponsive`. `hermes send --to whatsapp:<chat_id>` and cron
delivery check the same field and send nothing through another profile's bridge
or through one whose `/health` fails. If you
override `session_path`, keep it distinct per profile, or the profiles share one
WhatsApp login. Bridges started by an older Hermes report no session directory
and are restarted once, as after a bridge update.
