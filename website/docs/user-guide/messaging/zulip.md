---
sidebar_position: 12
title: "Zulip"
description: "Set up Hermes Agent as a Zulip bot"
---

# Zulip Setup

Hermes Agent integrates with Zulip as a bundled platform plugin, letting you chat with your AI assistant through direct messages or stream topics. Zulip is an open-source team chat platform — you can use Zulip Cloud (hosted at zulipchat.com) or run it on your own infrastructure. The bot connects via the official `zulip` Python package using Zulip's REST API and long-polling event queue, processes messages through the Hermes Agent pipeline (including tool use, memory, and reasoning), and responds in real time. It supports text, images (including pasted/attached /user_uploads/ images, automatically fetched for vision-capable models), documents, video uploads, and typing indicators.

The `zulip` Python package is optional and is only needed when you configure the
Zulip plugin. You do not need to install it before running the setup wizard.
After Zulip credentials are configured, Hermes installs the pinned Zulip SDK
automatically on first use when `security.allow_lazy_installs` is enabled (the
default).

For locked-down or offline environments, preinstall the Zulip extra into the
same environment that runs Hermes. From a source checkout created by
`setup-hermes.sh`, use:

```bash
UV_PROJECT_ENVIRONMENT=venv uv sync --locked --extra all --extra zulip
```

If your checkout uses `.venv` instead, set `UV_PROJECT_ENVIRONMENT=.venv`.

For a pip-installed Hermes package, use:

```bash
python -m pip install 'hermes-agent[zulip]'
```

`setup-hermes.sh` installs Hermes with the curated `all` extra. Zulip is not
eager-installed by `all`; it is a lazy platform extra, like other optional
platform SDKs.

:::info
The Zulip adapter lives under `plugins/platforms/zulip/`, so setup, status,
authorization, cron delivery, and standalone sending are wired through the
platform plugin registry instead of core gateway files.
:::

Before setup, here's the part most people want to know: how Hermes behaves once it's in your Zulip organization.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs** | Hermes responds to every message. No `@mention` needed. Each DM has its own session. |
| **Stream messages** | Hermes responds when you `@mention` it. Without a mention, Hermes ignores the message. |
| **Topics** | Each stream+topic combination gets its own session. Changing the topic starts a fresh conversation. |
| **Group DMs** | Hermes responds to every message in group DMs. Each group DM has its own session. |
| **Shared streams with multiple users** | By default, Hermes isolates session history per user inside the stream. Two people talking in the same stream do not share one transcript unless you explicitly disable that. |

:::tip
If you want Hermes to respond in certain streams without an @mention, use `ZULIP_FREE_RESPONSE_STREAMS` to list stream names or IDs. This is useful for bot-dedicated channels.
:::

### Session Model in Zulip

By default:

- each DM gets its own session
- each stream+topic gets its own session
- each user in a shared stream gets their own session inside that stream+topic

This is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation for the entire stream:

```yaml
group_sessions_per_user: false
```

Shared sessions can be useful for a collaborative stream, but they also mean:

- users share context growth and token costs
- one person's long tool-heavy task can bloat everyone else's context
- one person's in-flight run can interrupt another person's follow-up in the same stream

### Agent-created topics

Enable the `zulip-history` toolset to let Hermes use
`zulip_send_topic_message`. It posts to any stream/topic the bot can access;
Zulip creates the topic when it does not yet exist. Hermes also creates or
reuses the matching topic session and records the sent seed message, so a later
reply in that topic starts with the right context rather than an empty session.

```text
zulip_send_topic_message(
  stream="projects",
  topic="new work lane",
  message="Let's investigate the deployment failure.",
)
```

In a Zulip-origin conversation, Hermes automatically uses the current sender
as the session owner. From the CLI, cron, or another messaging platform, pass
`session_user_email` so Hermes knows whose per-user Zulip session to seed.

This guide walks you through the full setup process — from creating your bot on Zulip to sending your first message.

## Step 1: Create a Bot Account

1. Log in to your Zulip organization (cloud or self-hosted).
2. Go to **Settings** → **Your bots**.
3. Click **Add a new bot**.
4. Fill in the details:
   - **Bot type**: choose **Generic bot**.
   - **Bot email**: e.g., `hermes-bot@your-org.zulipchat.com`
   - **Full name**: e.g., `Hermes Agent`
   - **Role**: can be a normal user or admin, depending on your needs
5. Click **Create bot**.
6. Zulip will display the **bot's API key**. **Copy it immediately.**

:::warning[API key shown only once]
The bot's API key is only displayed once when you create the bot. If you lose it, you'll need to regenerate it from the bot's settings page. Never share your API key publicly or commit it to Git — anyone with this key has full control of the bot.
:::

:::info
For self-hosted Zulip, make sure the bot is enabled after creation. Navigate to the bot in **Settings** → **Your bots** and verify its status.
:::

Store the API key somewhere safe (a password manager, for example). You'll need it in Step 3.

## Step 2: Subscribe the Bot to Streams

The bot needs to be subscribed to any stream where you want it to respond:

1. Open the stream where you want the bot.
2. Click the **stream name** → **Stream settings**.
3. Go to the **Subscribers** tab.
4. Search for the bot's email address and add it.

For DMs, simply open a direct message with the bot — it will be able to respond immediately without subscribing to any streams.

## Step 3: Configure Hermes Agent

### Option A: Interactive Setup (Recommended)

Run the guided setup command:

```bash
hermes gateway setup
```

Select **Zulip** when prompted, then enter your server URL, bot email, API key, and allowed user emails when asked.

If the Zulip SDK is not already installed, the first configured gateway start
will install it automatically unless lazy installs are disabled.

### Option B: Manual Configuration

Add the following to your `~/.hermes/.env` file:

```bash
# Required
ZULIP_SITE_URL=https://your-org.zulipchat.com
ZULIP_BOT_EMAIL=hermes-bot@your-org.zulipchat.com
ZULIP_API_KEY=***

# Required unless ZULIP_ALLOW_ALL_USERS=true
ZULIP_ALLOWED_USERS=you@example.com

# Multiple allowed users (comma-separated)
# ZULIP_ALLOWED_USERS=you@example.com,colleague@example.com
```

Optional settings in `~/.hermes/.env`:

```bash
# Allow all users without an allowlist (NOT recommended for bots with terminal access)
# ZULIP_ALLOW_ALL_USERS=true

# Default stream for outbound messages
ZULIP_DEFAULT_STREAM=general

# Home topic for cron/reminder delivery when ZULIP_HOME_CHANNEL is not set
# ZULIP_HOME_TOPIC=notifications

# Mention gating (default: true)
# ZULIP_REQUIRE_MENTION=false

# TLS for self-hosted/local Zulip with private CA or self-signed certs
# Preferred: trust your local CA explicitly
# ZULIP_CERT_BUNDLE=/path/to/ca.pem
# Temporary local-dev fallback only:
# ZULIP_ALLOW_INSECURE=true   # disables TLS verification

# Streams where @mention is not required (comma-separated names or IDs)
# ZULIP_FREE_RESPONSE_STREAMS=bot-commands,42

# Missed-message catch-up — back-fill messages that arrived while the gateway
# was down (default: off). See "Missed-Message Catch-Up" below.
# ZULIP_CATCHUP=true
# ZULIP_CATCHUP_MAX_MESSAGES=100   # per-stream replay cap per (re-)register
```

Optional behavior settings in `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true

display:
  platforms:
    zulip:
      # Enables live response streaming by editing Zulip messages in place.
      # Before setting true, disable edit history in Zulip organization settings.
      streaming: false
```

- `group_sessions_per_user: true` keeps each participant's context isolated inside shared streams and group DMs
- `display.platforms.zulip.streaming: true` opts Zulip into edit-based live response streaming

### Tool Progress Display

Shared Zulip streams can get noisy if every tool call is echoed into the
conversation. To keep tool progress quiet in Zulip while leaving final answers
unchanged, set a per-platform display override:

```yaml
display:
  platforms:
    zulip:
      tool_progress: off      # or log
```

Use `tool_progress: log` to keep chat silent while writing tool-call activity to
Hermes logs. To show progress in chat, use `new`, `all`, or `verbose`.

### Start the Gateway

Once configured, start the gateway in the foreground:

```bash
hermes gateway run
```

The bot should connect to your Zulip server within a few seconds. You'll see a log message like:

```
Zulip: authenticated as hermes-bot@your-org.zulipchat.com (user_id=123) on https://your-org.zulipchat.com
```

Send it a DM or @mention it in a stream to test.

:::tip
Use `hermes gateway run` for a foreground test run. Once that works, you can install the systemd/launchd service for persistent operation.
:::

## Home Channel

You can designate a "home stream+topic" where the bot sends proactive messages (such as cron job output, reminders, and notifications). There are two ways to set it.

### Using the Slash Command

Type `/sethome` in any Zulip stream or DM where the bot is present. That stream+topic becomes the home channel.

### Manual Configuration

Add either a combined stream+topic target:

```bash
ZULIP_HOME_CHANNEL=general:notifications
```

The format is `stream_name:topic`. Hermes resolves the stream name to the correct Zulip stream ID before sending, so manual config stays human-readable.

Or use separate variables:

```bash
ZULIP_DEFAULT_STREAM=general
ZULIP_HOME_TOPIC=notifications
```

`ZULIP_HOME_CHANNEL` takes precedence when both forms are set.

## Known Core Follow-Ups

The Zulip adapter is intentionally shipped through the plugin path, so this PR
does not change core target parsing or cron bookkeeping. Two generic Hermes
edges are worth tracking separately:

- `hermes send --to zulip "message"` works when `ZULIP_HOME_CHANNEL` is set.
  `hermes send --to "zulip:5:Sauve Cloud Status" "message"` currently goes
  through the shared channel-name resolver before the Zulip adapter sees it.
  A core follow-up can teach `send_message_tool` that Zulip `stream:topic` and
  `dm:...` strings are explicit adapter-native targets.
- `hermes cron run` for a finite `--repeat 1` smoke job can report failure
  after the job removes itself, even when the script ran and delivery succeeded.
  The saved output and gateway logs are authoritative for that edge case. Use
  `--repeat 2` during manual smoke testing if you need the job to remain visible
  after the first run.

Candidate patch for the first follow-up:

```diff
diff --git a/tools/send_message_tool.py b/tools/send_message_tool.py
@@
     if platform_name == "ntfy":
         topic = target_ref.strip()
         if topic:
             return topic, None, True
+    if platform_name == "zulip":
+        target = target_ref.strip()
+        if target.startswith("dm:") and len(target) > 3:
+            return target, None, True
+        if ":" in target:
+            return target, None, True
     if platform_name == "email":
```

Candidate shape for the second follow-up:

```diff
diff --git a/cron/scheduler.py b/cron/scheduler.py
@@
         if success and not final_response.strip():
             success = False
             error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"
+
+        # mark_job_run() removes finite one-shot jobs immediately after their
+        # repeat limit is reached. Preserve the run result in memory so
+        # manual callers can report the actual outcome after removal.
+        job["_execution_success"] = bool(success)
+        job["_execution_error"] = error

         mark_job_run(job["id"], success, error, delivery_error=delivery_error)
diff --git a/tools/cronjob_tools.py b/tools/cronjob_tools.py
@@
         processed = run_one_job(job)
         refreshed = get_job(job_id) or {}
-        ok = refreshed.get("last_status") == "ok"
+        ok = refreshed.get("last_status") == "ok" if refreshed else bool(job.get("_execution_success"))
+        error = refreshed.get("last_error") if refreshed else job.get("_execution_error")
         return {
             "claimed": True,
             "success": bool(processed and ok),
-            "error": refreshed.get("last_error"),
+            "error": error,
         }
```

## Missed-Message Catch-Up

The Zulip events API only delivers events from the moment the gateway registers
its event queue onward. If the gateway is down — a restart, a `BAD_EVENT_QUEUE_ID`
expiry after a long idle, or a network drop — any messages that arrive during the
gap are never delivered, and the bot silently never sees them.

Catch-up closes that gap. When enabled, on every (re-)register the adapter
back-fills each stream from a persisted per-stream watermark and feeds the missed
messages through the **same path live messages take** — so dedup, mention-gating,
and per-topic sessions all behave identically to messages received in real time.

It is **off by default**: enabling it on a bot that has been offline for a while
replays the accumulated backlog (up to the per-stream cap), which is usually
surprising. Turn it on deliberately.

```bash
ZULIP_CATCHUP=true                 # opt in (default: false)
ZULIP_CATCHUP_MAX_MESSAGES=100     # per-stream replay cap per (re-)register
```

Or in `~/.hermes/config.yaml` under the Zulip platform's `extra`:

```yaml
catchup_enabled: true
catchup_max_messages: 100
```

Behavior details:

- **First run never floods.** The first time catch-up sees a stream (no stored
  watermark) it records the newest message id as a baseline and replays nothing —
  a clean start does not drag in history.
- **Bounded.** At most `catchup_max_messages` messages per stream are replayed per
  (re-)register, so even a long downtime can't trigger an unbounded replay.
- **Exactly-once in practice.** Replayed messages flow through the same dedup the
  live queue uses, so a message caught by both the sweep and the live queue is
  processed once.
- **Forward-only.** The watermark only advances; it is persisted next to the bot's
  state and survives restarts.

## Mention Gating

By default, Hermes only responds in streams when it is @mentioned. This prevents the bot from processing every message in a busy stream.

### Disabling Mention Requirement

Set `ZULIP_REQUIRE_MENTION=false` in your `~/.hermes/.env` to make the bot respond to all messages in every stream:

```bash
ZULIP_REQUIRE_MENTION=false
```

### Per-Stream Exemptions

Use `ZULIP_FREE_RESPONSE_STREAMS` to exempt specific streams from the mention requirement while keeping it active elsewhere:

```bash
ZULIP_FREE_RESPONSE_STREAMS=bot-commands,ai-assistant
```

You can use stream names or stream IDs (comma-separated). This is useful for dedicated bot channels where you want a conversational experience without the @mention overhead.

### Historical Context on @mention

By default, the bot only sees the message where it was @mentioned — it has no idea what was discussed beforehand. It's like walking into a room mid-conversation.

Set `ZULIP_CONTEXT_DEPTH` in your `~/.hermes/.env` to instruct the bot to fetch recent messages from the same stream+topic when summoned:

```bash
ZULIP_CONTEXT_DEPTH=20
```

When someone types `@bot what do you think about the proposal above?`, the bot fetches the last 20 messages from Zulip's `/messages` REST API and injects them as context before the current message. The agent sees something like:

```
Recent messages in this topic:
Alice: I think we should use PostgreSQL
Bob: Agreed, but what about migrations?
Charlie: We can use Alembic for that
---
@**Hermes Bot** what do you think about the proposal above?
```

**Key properties:**

- **Survives disconnects** — uses Zulip as the source of truth via REST API, not the event queue. If the bot was offline for an hour, it still sees what happened.
- **On-demand** — only fetches when @mentioned (or in free-response streams). Zero overhead on unmentioned messages.
- **No local storage** — context is fetched fresh each time, never persisted between turns.
- **Privacy-preserving** — the bot only reads the stream+topic when explicitly summoned. It never silently observes or stores messages.
- **Bot's own messages skipped** — the bot filters itself out of the context so it doesn't see its own previous responses.

Set it to `0` (the default) to disable context fetching:

:::info
DMs and group DMs always bypass mention gating — the bot responds to every message in private conversations.
:::

### Message History Search Tool

The bot has access to a `zulip_search_messages` tool that lets it search and paginate through Zulip message history. This wraps Zulip's `/messages` API — the bot can use it to:

- **Fetch context** around a specific message ID (e.g., a message someone replied to)
- **Paginate** further back when the initial auto-fetched context isn't enough
- **Search** for specific content (e.g., "find where we discussed PostgreSQL")
- **Filter by sender** using Zulip's search operators

**Tool parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `stream` | Stream name | `"general"` |
| `topic` | Topic name | `"database"` |
| `query` | Full-text search (Zulip operators) | `"postgresql"`, `"sender:alice@example.com"` |
| `anchor` | Message ID or `"newest"`/`"oldest"` | `"newest"`, `"42"` |
| `num_before` | Messages to fetch before anchor | `20` |
| `num_after` | Messages to fetch after anchor | `5` |

**Common patterns the bot uses:**

```python
# Recent conversation context
zulip_search_messages(stream="general", topic="database", anchor="newest", num_before=20)

# Context around a specific message (e.g., a reply target)
zulip_search_messages(stream="general", anchor="42", num_before=5, num_after=5)

# Find where PostgreSQL was discussed
zulip_search_messages(stream="general", query="postgresql")

# Show older page of history
zulip_search_messages(stream="general", topic="database", anchor="<oldest_message_id>", num_before=20)
```

:::warning[Search scope is restricted to the current conversation]
When the bot is talking to you **through Zulip** (DM, group DM, or stream), the `zulip_search_messages` tool is automatically restricted to the **current conversation only**. A user in a private DM cannot ask the bot to search messages from streams or other DMs the bot is subscribed to. This prevents the bot from being used as a proxy to exfiltrate content from conversations you don't have access to.

When the bot is invoked from the **CLI** or other platforms, the full search scope is available (subject to the bot's own Zulip permissions).
:::

The response includes pagination hints (`oldest_message_id`, `newest_message_id`) so the bot can continue browsing without guesswork.

### Downloading historical attachments

`zulip_search_messages` returns the original Markdown for historical messages.
When it finds a `/user_uploads/...` link, the bot can call
`zulip_download_attachment(message_id, filename?, attachment_index?)` to save
that attachment into Hermes's local media cache. The tool returns a local path
for `read_file` or terminal tools to inspect.

In a Zulip conversation, the download tool has the same stream/topic or DM
restriction as message-history search. It accepts a message ID rather than an
arbitrary upload URL, so it cannot be used to retrieve a file from another
conversation that the requesting user cannot access.

:::tip
If the bot says "I don't have enough context," tell it: "use zulip_search_messages to look further back in this topic." It will paginate until it finds what it needs.
:::

## Live Response Streaming Edits

Zulip streaming uses message edits: Hermes sends the first partial response as a normal Zulip message, then edits that same message as more text is generated. This mode is disabled by default because Zulip may preserve every intermediate edit in message edit history.

To enable it for Zulip only, use the standard per-platform gateway streaming switch in `~/.hermes/config.yaml`:

```yaml
streaming:
  enabled: false
  transport: auto
  # Desktop Zulip can visibly repaint on each edit. A high threshold lets the
  # interval control the cadence instead of flushing for every short segment.
  edit_interval: 2.0
  buffer_threshold: 10000

display:
  platforms:
    zulip:
      # First disable edit history in Zulip organization settings.
      streaming: true
```

You can also enable streaming globally with `streaming.enabled: true`; `display.platforms.zulip.streaming: true` is the narrower opt-in for Zulip. Before enabling Zulip streaming, disable edit history in your Zulip organization settings. Otherwise intermediate streamed content, including partial assistant drafts, may remain visible in the message's edit history.

Some Zulip desktop clients visibly repaint an edited message. If frequent
updates look like blinking, set a 1–2 second `edit_interval` and a high
`buffer_threshold`, as above: the interval is the minimum time between edits,
while the threshold is a character-count force-flush. Restart the gateway after
changing the setting.

When Zulip streaming edits are first used, Hermes logs a warning once. With Zulip streaming unset or false, Zulip sends normal whole-message replies and does not log the warning.

## Long responses

Hermes sends a Zulip response as one message up to 10,000 Unicode code points,
the standard Zulip message limit. Responses above that limit are split at
natural text boundaries so that code blocks remain valid Markdown.

For a self-hosted Zulip realm, make sure its `MAX_MESSAGE_LENGTH` in
`/etc/zulip/settings.py` is at least `10000`, then restart the server. Docker
deployments can set the equivalent `SETTING_MAX_MESSAGE_LENGTH=10000`. Zulip
Cloud users cannot change this server setting. Zulip publishes the effective
per-realm maximum through its registration API, so a lower realm limit will
still reject a message that exceeds it.

## Sending Messages Cross-Platform

You can send messages to Zulip from other platforms using the `send_message` tool. The plugin registers a standalone sender, so `send_message` and cron delivery can work even when the gateway is not running in the same process.

| Target | Description |
|--------|-------------|
| `zulip` | Sends to the home stream+topic |

Out-of-process standalone sending is text-focused. Media attachments are supported by the live gateway adapter, but standalone cron/tool sends return a clear unsupported-media error when media files are provided. The standalone sender uses the same Zulip dependency check as the gateway adapter, so configured cron/status delivery can trigger the normal lazy install path.

## Cron Delivery

Cron jobs can deliver results to Zulip. Use `deliver="zulip"` to send to the configured home stream+topic:

```
deliver="zulip"
```

For cron-only deployments where the gateway is not running in the same process,
make sure the Zulip credentials and home channel are available to the cron
process. If runtime lazy installs are disabled, preinstall the Zulip extra before
running cron jobs.

## Media attachments

The Zulip adapter supports uploading and sending media files:

| Type | Behavior |
|------|----------|
| **Images** | Uploaded to Zulip and rendered inline using `![alt](/user_uploads/...)` |
| **Documents** | Uploaded and sent as a clickable Markdown link `[filename](/user_uploads/...)` |
| **Video** | Uploaded and sent as a downloadable link (Zulip does not inline video playback) |
| **Voice/audio** | Uploaded and sent as a downloadable Markdown link. Zulip has no native voice bubble support. |

Media delivery works in both streams and DMs.

Incoming attachments from the triggering message are also downloaded
automatically. Images go through the normal vision path; documents, archives,
text files, audio, and video are cached locally and surfaced to the agent as
paths it can inspect with its existing tools. The current-message download is
automatic, so no model tool call is required.

## Troubleshooting

### Bot is not responding to messages

**Cause**: The bot is not subscribed to the stream, or `ZULIP_ALLOWED_USERS` doesn't include your email.

**Fix**: Subscribe the bot to the stream (stream settings → Subscribers → add the bot's email). Verify your email is in `ZULIP_ALLOWED_USERS`. Restart the gateway.

### 401 Unauthorized errors

**Cause**: The API key, bot email, or server URL is incorrect.

**Fix**: Verify all three values in your `.env` file. Check that `ZULIP_SITE_URL` includes `https://` and has no trailing slash.

Run `hermes gateway run` in the foreground and check for the authenticated log
line.

### Bot ignores stream messages

**Cause**: `ZULIP_REQUIRE_MENTION` is `true` (the default) and the bot isn't @mentioned.

**Fix**: Either @mention the bot (e.g., `@**Hermes Agent** hello`), or add the stream to `ZULIP_FREE_RESPONSE_STREAMS`, or set `ZULIP_REQUIRE_MENTION=false`.

### "zulip package not installed" on startup

**Cause**: The `zulip` Python package is not installed, and Hermes could not
install it at runtime because lazy installs are disabled or the environment is
offline.

**Fix**: Normally Hermes installs this automatically on first configured use. If
lazy installs are disabled, install the Zulip extra into the same environment
that runs Hermes. From a source checkout created by `setup-hermes.sh`:

```bash
UV_PROJECT_ENVIRONMENT=venv uv sync --locked --extra all --extra zulip
hermes gateway run
```

If your checkout uses `.venv` instead, set `UV_PROJECT_ENVIRONMENT=.venv`.

For a pip-installed Hermes package:

```bash
python -m pip install 'hermes-agent[zulip]'
hermes gateway run
```

### Event queue disconnects / reconnection loops

**Cause**: Network instability, Zulip server restarts, or firewall issues with long-polling connections.

**Fix**: The adapter automatically reconnects with exponential backoff (2s → 60s). Check your network connectivity. If you're behind a proxy, ensure it supports long-lived HTTP connections.

### Bot is offline

**Cause**: The Hermes gateway isn't running, or it failed to connect.

**Fix**: Check that `hermes gateway run` is running. Look at the terminal output or `~/.hermes/logs/gateway.log` / `~/.hermes/logs/gateway.error.log` for error messages. Common issues: wrong server URL, stale API key, Zulip server unreachable, or self-signed TLS without `ZULIP_CERT_BUNDLE` / `ZULIP_ALLOW_INSECURE`.

### "User not allowed" / Bot ignores you

**Cause**: Your email isn't in `ZULIP_ALLOWED_USERS`.

**Fix**: Add your email to `ZULIP_ALLOWED_USERS` in `~/.hermes/.env` and restart the gateway. Remember: this is your **email address**, not your Zulip username.

## Security

:::warning
Always set `ZULIP_ALLOWED_USERS` to restrict who can interact with the bot. Without it, the gateway denies all users by default as a safety measure. Only add emails of people you trust — authorized users have full access to the agent's capabilities, including tool use and system access.
:::

If you want to allow all users in your Zulip organization, set `ZULIP_ALLOW_ALL_USERS=true`. This is only appropriate for private organizations where all members are trusted.

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).

## Notes

- **Zulip Cloud and self-hosted**: Works with both zulipchat.com cloud organizations and self-hosted Zulip servers.
- **Official client**: Uses the `zulip` Python package for reliable API access.
- **Long-polling**: The event queue uses Zulip's long-polling mechanism — no WebSocket or incoming webhook needed.
- **Stream topics**: Each topic in a stream gets its own session, which maps naturally to Zulip's topic-based conversation model.
- **DM pairing**: Unknown users who DM the bot receive a one-time pairing code (see the [Messaging Gateway](index.md) docs for details on the pairing flow).
