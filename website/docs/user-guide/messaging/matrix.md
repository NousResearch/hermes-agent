---
sidebar_position: 9
title: "Matrix"
description: "Set up Hermes Agent as a Matrix bot"
---

# Matrix Setup

Python dependency commands on this page use a
[PM-prepared source checkout](../../reference/package-management.md#developer-workflow).
After a dependency change, reactivate the checkout and restart Hermes.

Hermes Agent integrates with Matrix, the open, federated messaging protocol. Matrix lets you run your own homeserver or use a public one like matrix.org — either way, you keep control of your communications. The bot connects via the `mautrix` Python SDK, processes messages through the Hermes Agent pipeline (including tool use, memory, and reasoning), and responds in real time. It supports text, file attachments, images, audio, video, and optional end-to-end encryption (E2EE).

Hermes works with any Matrix homeserver — Synapse, Conduit, Dendrite, or matrix.org.

Before setup, here's the part most people want to know: how Hermes behaves once it's connected.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs** | Hermes treats a room as a private bot chat when exactly two users have joined: the bot and one other user. It responds to every message from an authorised user without an `@mention`, and each chat has its own session. Set `MATRIX_DM_MENTION_THREADS=true` to start a thread when the bot is `@mentioned` in a private bot chat. The room name and `m.direct` metadata do not change this classification. |
| **Rooms** | By default, Hermes requires an `@mention` to respond. Set `MATRIX_REQUIRE_MENTION=false` or add room IDs to `MATRIX_FREE_RESPONSE_ROOMS` for free-response rooms. Room invites are auto-accepted. A room with more than two joined users, or whose joined membership cannot be determined, follows room rules. Only private bot chats bypass `MATRIX_ALLOWED_ROOMS` and the room mention settings. |
| **Threads** | Hermes supports Matrix threads (MSC3440). If you reply in a thread, Hermes keeps the thread context isolated from the main room timeline. Threads where the bot has already participated do not require a mention. |
| **Auto-threading** | By default, Hermes auto-creates a thread for each message it responds to in a room. This keeps conversations isolated. Set `MATRIX_AUTO_THREAD=false` to disable. Set `MATRIX_DM_AUTO_THREAD=true` (default false) to also auto-create threads for private bot chats. This is distinct from `MATRIX_DM_MENTION_THREADS`, which starts a thread when the bot is `@mentioned` in a private bot chat. These two-person chats follow `MATRIX_DM_AUTO_THREAD`. |
| **Commands** | Hermes accepts normal `/commands` when your Matrix client sends them. If your client reserves `/` for local commands, use `!commands` instead; Hermes normalizes known `!command` aliases to `/command`. |
| **Emotes and stickers** | Native `/me` actions include the sender's Matrix ID. Their text is conversational, even when it begins with a command. Stickers include their description and image, subject to the same room, sender and mention rules as other messages. |
| **Interactive controls** | Dangerous-command approval and `/model` selection can use Matrix reactions. Approval reactions can be limited to the user who requested the action. |
| **Thinking and tool activity** | Matrix uses threaded, editable thinking/tool-activity panes when gateway progress is enabled, so updates do not flood the main room timeline. |
| **Shared rooms with multiple users** | By default, Hermes isolates session history per user inside the room. Two people talking in the same room do not share one transcript unless you explicitly disable that. |
| **LaTeX math** | `$...$` (inline) and `$$...$$` (display) in replies are sent as Element `data-mx-maths` markup, so clients with **Settings → Labs → Render LaTeX maths in messages** typeset them with KaTeX. Unpaired dollars (`$5 or $10`) stay literal, and the plain-text `body` keeps the raw TeX for other clients. |

:::tip
The bot automatically joins rooms when invited. Just invite the bot's Matrix user to any room and it will join and start responding.
:::

## Capability Matrix

This table is backed by the Matrix adapter capability declaration and Matrix test
coverage. E2EE is mode-based because deployments choose whether encrypted rooms
are disabled, opportunistic, or required.

| Capability | Matrix |
|------------|--------|
| text | yes |
| threads | yes |
| reactions | yes |
| approvals | yes |
| model picker | yes |
| thinking panes | yes |
| images | yes |
| multiple images | yes |
| files | yes |
| voice/audio | yes |
| video | yes |
| E2EE | off / optional / required |
| diagnostics | yes |

### Native Actions and Stickers

Hermes receives native Matrix emotes (`m.emote`) and stickers (`m.sticker`) in
plain and encrypted rooms. Emotes appear as
`[emote by https://matrix.to/#/@user:server] action`. The sender's Matrix ID is
written as a permalink so that Hermes does not read it as an `@` context
reference such as `@file:`. Stickers appear as `[sticker: description]`, with
cached image pixels when the attachment can be downloaded within the configured
media limit. If the image is too large or unavailable, Hermes receives the
description and an attachment status instead. Hermes ignores a sticker that the
server has already redacted when Hermes receives it.

Replies, thread history and explicit Matrix context reads use the same action
and sticker descriptions. When an emote or sticker is edited or redacted before
model dispatch, Hermes rechecks that event's contribution to the new input. A
withdrawn sticker loses its cached pixels or image description; other stickers
in a queued burst keep theirs. This recheck affects the new input. Earlier model requests
and the conversation's existing transcript remain unchanged.

### Session Model in Matrix

By default:

- each DM gets its own session
- each thread gets its own session namespace
- each user in a shared room gets their own session inside that room

This is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation for the entire room:

```yaml
group_sessions_per_user: false
```

Shared sessions can be useful for a collaborative room, but they also mean:

- users share context growth and token costs
- one person's long tool-heavy task can bloat everyone else's context
- one person's in-flight run can interrupt another person's follow-up in the same room

### Mention and Threading Configuration

You can configure mention and auto-threading behavior via environment variables or `config.yaml`:

```yaml
matrix:
  require_mention: true           # Require @mention in rooms (default: true)
  allowed_users:                  # Matrix users allowed to trigger agent turns
    - "@alice:matrix.org"
  allowed_rooms:                  # Matrix rooms allowed to trigger agent turns
    - "!abc123:matrix.org"
  channel_prompts: {}             # Optional prompts keyed by internal room ID
  channel_skill_bindings: []      # Optional skills keyed by internal room ID
  free_response_rooms:            # Rooms exempt from mention requirement
    - "!abc123:matrix.org"
  ignore_user_patterns:           # Bridge/appservice ghost users to ignore
    - "^@telegram_"
    - "^@whatsapp_"
  process_notices: false          # Ignore m.notice by default
  session_scope: room             # auto|room|thread; room is recommended for project rooms
  auto_thread: true               # Auto-create threads for responses (default: true)
  dm_mention_threads: false       # Create thread when @mentioned in DM (default: false)
  max_message_length: 16000       # Outbound chunk size in chars (default: 16000, max: 65535)
  room_backfill_limit: 20         # Earlier room events to scan on a mention (0 disables)
  thread_backfill_limit: 20       # Earlier thread events to scan (0 disables)
```

In shared rooms and threads that require a mention, an admitted @mention includes the
messages that did not mention the bot since its previous turn there. The scan stops at
the bot's own last reply or the last admitted mention, whichever is later, and each
limit bounds the number of events scanned. The bot's status notices, such as restart
notices and progress updates, are not replies: the scan continues past them, and
neither catch-up nor thread history includes them. The previous turn can belong to an
earlier session. In the main timeline, catch-up after `/new` starts after the bot's
reply to `/new`, so the conversation that the reset discarded stays out of the new
session. Thread messages stay within their thread; room catch-up excludes thread
replies. The first message of a new thread session includes the thread root and up to
`thread_backfill_limit` earlier thread messages instead. That history does not stop at
the bot's own messages, because the new session has no transcript that contains them.
In a thread, the first message after `/new` starts a new thread session, so it gets
that history, including the messages from before `/new`. Free-response rooms and rooms
with `require_mention: false` start a turn for every message, so they have no catch-up.
Threads that the bot already takes part in also start a turn for every message unless
`thread_require_mention` is `true`. With that setting, those threads require a mention
and get catch-up as well. Set either limit to `0` to disable that source of earlier
messages. Catch-up and thread history also list recent reactions to the included
messages. Redacted reactions are excluded. Ordinary reactions do not start an agent
turn.

Or via environment variables:

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_ALLOWED_USERS=@alice:matrix.org
MATRIX_ALLOWED_ROOMS=!abc123:matrix.org
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@whatsapp_'
MATRIX_PROCESS_NOTICES=false
MATRIX_SESSION_SCOPE=room       # recommended for stable project-room context
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_ALLOW_ROOM_MENTIONS=false
```

### Read receipts and lifecycle reactions

Configure automatic feedback in the owning profile's `config.yaml`:

```yaml
matrix:
  read_receipts: immediate
  reactions: true
```

| `read_receipts` | Behaviour |
|-----------------|-----------|
| `immediate` | Default. Mark an accepted inbound message as read on arrival. |
| `after_processing` | Mark the message as read when processing completes, including failure. Expected cancellation skips the receipt. |
| `disabled` | Send no automatic read receipts. |

Unrecognised receipt modes use `immediate`.

`send_read_receipts` is the read-receipt switch that WhatsApp and BlueBubbles
also use. On Matrix, `send_read_receipts: false` turns automatic receipts off
whatever `read_receipts` says. When it is `true` or unset, `read_receipts`
chooses when the receipt is sent.

`reactions: false` disables the processing reactions (👀/✅/❌). Approval,
model picker and choice picker reactions remain available. Reactions default to
`true`; completion keeps the existing success, failure and cancellation behaviour.

Each served profile resolves its own settings. The existing `MATRIX_REACTIONS`
environment override takes precedence over that profile's YAML value. Receipt
timing is configured only through `read_receipts` in YAML. Both settings also
work under `platforms.matrix` or `gateway.platforms.matrix`.

:::tip Room-wide mentions
Hermes sends structured Matrix user mentions for explicit Matrix IDs such as `@alice:example.org`. Room-wide `@room` notifications are disabled by default; set `MATRIX_ALLOW_ROOM_MENTIONS=true` only in rooms where the bot is allowed to notify everyone.
:::

:::note
If you are upgrading from a version that did not have `MATRIX_REQUIRE_MENTION`, the bot previously responded to all messages in rooms. To preserve that behavior, set `MATRIX_REQUIRE_MENTION=false`.
:::

### Project Room Isolation

If you use the same Matrix bot in multiple project rooms, configure stable
room-scoped sessions:

```bash
MATRIX_SESSION_SCOPE=room
MATRIX_AUTO_THREAD=false
```

`MATRIX_SESSION_SCOPE` accepts:

| Scope | Behavior |
|-------|----------|
| `auto` | Backward-compatible default. Existing `MATRIX_AUTO_THREAD` behavior controls synthetic threads. |
| `room` | Unthreaded room messages stay in one stable room session. Real Matrix threads still use their thread root. |
| `thread` | Unthreaded room messages synthesize a thread/session from the triggering event ID. |

Hermes now includes the current Matrix room name, room ID, topic, message ID,
and a Matrix room-boundary note in the agent prompt. `/status` also shows the
current Matrix room/session scope, and `/resume` will not silently resume a
named session from another Matrix room unless you explicitly use
`/resume --cross-room <session name>`.

`MATRIX_SESSION_SCOPE=room` controls the room/thread lane. The existing
`group_sessions_per_user` setting still controls whether users inside that room
share the lane. With `group_sessions_per_user: true` (default), Alice and Bob get
separate Project B sessions. With `group_sessions_per_user: false`, the room has
one shared Project B transcript.

During a live Matrix turn, the agent can use `matrix_read` to inspect the
current room's name, topic, alias, join rule, history visibility, encryption,
joined members, power levels, and pinned messages. Member and pin results are
bounded by the tool's `limit` argument. Pinned messages use the bot's Matrix
client and report individual fetch or decryption errors. The tool applies the
same room and sender access rules as Matrix history reads.

Enable the `matrix_admin` toolset for Matrix in `hermes tools` to allow room
administration. This toolset is disabled by default. Its actions use the
current session's connected client and the same room and sender access rules
as Matrix reads. The `matrix_pin` action changes the current room's
`m.room.pinned_events` state. The requesting user must have the room power
level needed to change pins, so a member cannot use the bot's power to change
pins that they could not change themselves. The homeserver then checks the
bot's own permission, and the tool reports its error if the bot cannot send
that state event. The other administration actions are described in
[Matrix Tools and Controls](#matrix-tools-and-controls).

This guide walks you through the full setup process — from creating your bot account to sending your first message.

## Step 1: Create a Bot Account

You need a Matrix user account for the bot. There are several ways to do this:

### Option A: Register on Your Homeserver (Recommended)

If you run your own homeserver (Synapse, Conduit, Dendrite):

1. Use the admin API or registration tool to create a new user:

```bash
# Synapse example
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. Choose a username like `hermes` — the full user ID will be `@hermes:your-server.org`.

### Option B: Use matrix.org or Another Public Homeserver

1. Go to [Element Web](https://app.element.io) and create a new account.
2. Pick a username for your bot (e.g., `hermes-bot`).

### Option C: Use Your Own Account

You can also run Hermes as your own user. This means the bot posts as you — useful for personal assistants.

## Step 2: Get an Access Token

Hermes needs an access token to authenticate with the homeserver. You have two options:

### Option A: Access Token (Recommended)

The most reliable way to get a token:

**Via Element:**
1. Log in to [Element](https://app.element.io) with the bot account.
2. Go to **Settings** → **Help & About**.
3. Scroll down and expand **Advanced** — the access token is displayed there.
4. **Copy it immediately.**

**Via the API:**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

The response includes an `access_token` field — copy it.

:::warning[Keep your access token safe]
The access token gives full access to the bot's Matrix account. Never share it publicly or commit it to Git. If compromised, revoke it by logging out all sessions for that user.
:::

### Option B: Password Login

Instead of providing an access token, you can give Hermes the bot's user ID and password. Hermes will log in automatically on startup. This is simpler but means the password is stored in your `.env` file.

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## Step 3: Find Your Matrix User ID

Hermes Agent uses your Matrix User ID to control who can interact with the bot. Matrix User IDs follow the format `@username:server`.

To find yours:

1. Open [Element](https://app.element.io) (or your preferred Matrix client).
2. Click your avatar → **Settings**.
3. Your User ID is displayed at the top of the profile (e.g., `@alice:matrix.org`).

:::tip
Matrix User IDs always start with `@` and contain a `:` followed by the server name. For example: `@alice:matrix.org`, `@bob:your-server.com`.
:::

## Step 4: Configure Hermes Agent

### Option A: Interactive Setup (Recommended)

Run the guided setup command:

```bash
hermes gateway setup
```

Select **Matrix** when prompted, then provide your homeserver URL, access token (or user ID + password), and allowed user IDs when asked.

### Option B: Manual Configuration

Add the following to your `~/.hermes/.env` file:

**Using an access token:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Optional: restrict which rooms can trigger the bot
MATRIX_ALLOWED_ROOMS=!abc123:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**Using password login:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# Security
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

## Private Deployment Hardening

For private Matrix deployments, set both user and room allowlists. If
`MATRIX_ALLOWED_USERS` is unset, any sender who can reach the bot in a joined
room can trigger an agent turn. If `MATRIX_ALLOWED_ROOMS` is unset, any room the
bot joins can trigger an agent turn. A locked-down deployment should set both:

```bash
MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
MATRIX_ALLOWED_ROOMS=!ops:matrix.example.org,!dmroom:matrix.example.org
```

Bridge and appservice deployments need extra loop protection. Hermes always
ignores its own events, Matrix appservice-style users whose localpart starts
with `_`, duplicate event IDs, old startup events, edit replacement events, and
`m.notice` events by default. Add deployment-specific bridge ghost patterns when
your bridge uses a different naming convention:

```bash
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@slack_,^@whatsapp_'
```

Only enable notices when a trusted human workflow really sends `m.notice`:

```bash
MATRIX_PROCESS_NOTICES=true
```

Outbound whole-room notifications are disabled by default. Keep
`MATRIX_ALLOW_ROOM_MENTIONS=false` unless the bot is explicitly allowed to wake
the whole room with `@room`.

Diagnostics and debug payloads redact Matrix access tokens, recovery keys,
device identifiers, and message bodies. Media downloads are limited to Matrix
`mxc://` content URIs and rejected when they exceed `MATRIX_MAX_MEDIA_BYTES`.
Treat federated rooms and untrusted homeservers as untrusted input: keep room
allowlists tight, prefer DMs or private rooms for tool-heavy work, and avoid
authorizing bridge ghosts or appservice puppets as allowed users.

Optional behavior settings in `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` keeps each participant's context isolated inside shared rooms

### Start the Gateway

Once configured, start the Matrix gateway:

```bash
hermes gateway
```

The bot should connect to your homeserver and start syncing within a few seconds. Send it a message — either a DM or in a room it has joined — to test.

:::tip
You can run `hermes gateway` in the background or as a systemd service for persistent operation. See the deployment docs for details.
:::

### Restarts and messages sent while offline

Hermes saves its Matrix sync position for each profile, homeserver account and
device. After a restart it resumes from that position and handles every message
that arrived while it was stopped, however old the message is. After a long
outage the bot can therefore answer a backlog of messages in several rooms.

On the first start of a new device, or when the homeserver rejects the saved
position (for example after a homeserver database restore), Hermes starts from
a fresh sync. It then ignores messages that were sent more than five seconds
before startup, so it does not answer the room history again.

## End-to-End Encryption (E2EE)

Hermes supports Matrix end-to-end encryption, so you can chat with your bot in encrypted rooms.

### Requirements

E2EE requires the `mautrix` library with encryption extras and the `libolm` C library:

```bash
# Request the declared Matrix dependencies
python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
```

You also need `libolm` installed on your system:

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### Enable E2EE

Add to your `~/.hermes/.env`:

```bash
MATRIX_E2EE_MODE=required
```

`MATRIX_E2EE_MODE` accepts:

| Mode | Behavior |
|------|----------|
| `off` | Do not initialize Matrix E2EE. |
| `optional` | Try E2EE when dependencies are available, but keep unencrypted rooms working if crypto cannot initialize. |
| `required` | Fail closed if E2EE dependencies or crypto setup are not available. |

Optional mode may fall back to non-E2EE operation when crypto setup is unavailable. Required mode fails closed instead of silently downgrading.

For backwards compatibility, `MATRIX_ENCRYPTION=true` still enables required E2EE behavior.

When E2EE is enabled, Hermes:

- Stores encryption keys in `~/.hermes/platforms/matrix/store/` (legacy installs: `~/.hermes/matrix/store/`)
- Uploads device keys on first connection
- Decrypts incoming messages and encrypts outgoing messages automatically
- Auto-joins encrypted rooms when invited

### Matrix Tools and Controls

The `matrix_read` tool, in the `matrix_read` toolset, reads recent messages in the current room, one thread, or one event, and returns at most 50 events. Each call checks that the room is joined and allowed and that the user who sent the current message passes the Matrix user policy. Encrypted events are decrypted with the gateway's Matrix session. Each message in the result lists its reactions with their sender and target event.

Optional reaction menus give the agent a structured way to offer choices.

The `matrix_image_packs` toolset lists and sends image-pack stickers, as described in the Image-pack stickers section below.

The `matrix_reaction` tool, in the `matrix_reaction` toolset, adds an emoji reaction to a message or removes the reactions that the agent added. A reaction targets the current inbound message unless the agent supplies a Matrix event ID. The gateway records agent reactions in memory for the 1,000 most recently used messages, so the agent can remove only reactions that it added since the gateway process started. `MATRIX_REACTIONS=false` disables automatic processing reactions but does not block an agent-requested reaction.

The read and reaction tools operate in the current room. Each call checks that the room is joined and allowed and that the user who sent the current message passes the Matrix user policy.

`matrix_followup` can enable one reaction-triggered follow-up for the current turn. The agent can restrict it to specific emoji. For ten minutes after the final reply is delivered, a new matching reaction from the requester to any part of that reply starts one follow-up turn in the same room and thread. The action is disabled by default, and the last call in the turn decides its setting.

The `matrix_read`, `matrix_reaction`, `matrix_followup` and `matrix_image_packs` toolsets are enabled for Matrix sessions. Turn any of them off in the Matrix checklist of `hermes tools`, or run `hermes tools disable <toolset> --platform matrix`, for example `hermes tools disable matrix_reaction --platform matrix`. A saved Matrix toolset list that names individual toolsets and was saved before these toolsets existed does not include them; run `hermes tools enable <toolset> --platform matrix` to add each one.

Room administration requires the additional `matrix_admin` toolset. Enable it
for Matrix in `hermes tools`, or select it in the owning profile's config:

```yaml
platform_toolsets:
  matrix: [hermes-matrix, matrix_admin]
```

When a shared bot routes a conversation to another profile, administration
still uses the bot that received the event. Enable `matrix_admin` in both the
runtime profile and the receiving bot's profile. Each operation rechecks both
selections before writing.

| Tool | Action |
|---|---|
| `matrix_room_admin`, `action: create` | Create a private room with a name, topic, invite list, and optional E2EE |
| `matrix_room_admin`, `action: invite` | Invite a full Matrix user ID to the current room |
| `matrix_room_admin`, `action: leave` | Leave the current room |
| `matrix_room_admin`, `action: forget` | Forget the current room after the bot has left |
| `matrix_room_admin`, `action: redact` | Redact an event in the current room |
| `matrix_pin` | Pin or unpin an event in the current room |

These tools require a live Matrix session and an authorised actor. Existing
room actions are limited to the session's current room. Invite and pin also
require the actor's corresponding Matrix power level. Every redaction requires
the room's level for sending `m.room.redaction` events, and redacting another
user's event, including the bot's, also requires the redact level. Leave and
forget require the room's kick level, because removing the bot affects every
member of the room. A room v12 creator passes every power check. In a private
chat with the bot, where the actor is the only other joined member, leave and
forget need no power level. A refusal reports the required and actual levels.
The homeserver checks the bot's permissions and its errors are reported in the
tool result. Create uses the configured actor authorisation and no room power
level. It can invite only the actor and users whom the gateway authorises,
such as those in `MATRIX_ALLOWED_USERS`.

New rooms are private. `encrypted: true` requires the owning client's active
crypto store and adds Megolm encryption state during creation. With
`is_direct: true`, the room must invite exactly one user, and the bot records
it as a direct chat with that user in its `m.direct` account data. The bot sees
the new room through normal sync. Forget removes the room from the bot's own
account after leaving; other members retain their room. It does not purge the
room from the homeserver. A leave/forget sequence must run during the current
turn because the bot cannot receive a new turn in a room after leaving. An
authorised user can invite the bot back later, and it rejoins through its
normal invite handling.

Changing the admin selection affects tool discovery for new sessions. Active
admin operations recheck the current selection, profile, client, room policy,
and actor authorisation before a write. If the turn is interrupted or the
operation times out after the request was sent to the homeserver, the result
reports that the outcome is unknown and tells the agent how to check before
retrying. Enabling tools does not rebuild the system prompt or rewrite the
history of an existing conversation.

The adapter also uses reactions and redactions internally to power approval prompts and pickers.

If `MATRIX_ALLOWED_ROOMS` is set, Hermes only responds in those rooms and in private bot chats with exactly two joined users, including the bot.

Reaction controls use:

- ✅ approve once
- ♾️ approve always
- ❌ deny
- number reactions for `/model` choices

Set `MATRIX_APPROVAL_REQUIRE_SENDER=false` if you intentionally want any authorized Matrix user in the room to operate an approval/model picker prompt. The default is requester-bound when Hermes knows who requested the action.

#### Agent reaction menus

Enable **Reaction Menus** through `hermes tools`, or add `reaction_menu` to
`platform_toolsets.matrix` in `config.yaml`. The toolset is disabled by default
and is available only in Matrix sessions. Restart the gateway or start a new
session after changing toolsets.

The agent can call `present_menu` with a question and one to five choices. Each
choice has a distinct emoji, visible label and payload for the next turn. The
menu arrives in the current room or thread. Only the person who requested the
menu can select a choice, and that person must still pass the gateway's user
policy. The first valid selection starts one follow-up in the original session.

A new menu replaces the previous menu in the same session, and Hermes removes
its own reactions from the replaced menu. Menus expire after five minutes. A
reaction on an expired menu gets a reply saying that it has expired, and Hermes
removes its own reactions from that menu. A choice from a conversation that has
since been reset with `/new` is refused with a notice. Context compression continues the same conversation, so it
does not affect a pending menu. Menus are local to the running gateway and
disappear on restart; ask the agent for another menu if a choice is no longer
active.

A reaction on an approval prompt or model picker affects only that prompt, even
when a menu is pending in the same room.

### Image-pack stickers

In a Matrix session, `matrix_image_packs` lists sticker images from current-room
packs and accessible room packs referenced by the bot account. It supports
[MSC2545 image packs][image-packs-msc] and the legacy Cinny room and account
formats. Private account packs belong to the bot account. The tool does not
read the requester's private packs or join referenced rooms.

Use `action: "list"`, then pass an item's `selection_id` to `action: "send"`.
Selections expire after five minutes and are limited to the same conversation,
requester, room, bot and profile. `/new` and compression that creates a child
session invalidate earlier selections; in-place compression preserves them. The
tool reads the selected pack again after preparing encryption and rejects
changed or removed images and account references. It sends a native `m.sticker`
event with the current reply and thread relations. In encrypted rooms, the
owning Matrix client's crypto encrypts the sticker event. Pack media remains
public, as specified by MSC2545.

Pack labels, descriptions and image bodies are untrusted data. Catalogs return
at most 20 packs, 100 inspected images and 20 account references, with explicit
errors and truncation indicators. When the current room has more than 1,000
state events, or its state cannot be read, the catalog reports an error for the
room's own packs and still lists the bot account's private pack and the packs
that the bot account references. Matrix's state endpoint has no pagination, so
the SDK receives the complete state response before Hermes applies this
processing limit. Declared image sizes must fit the adapter's media limit;
sending an existing MXC URL does not download or re-upload the image.

The `matrix_image_packs` toolset is enabled for Matrix sessions. Turn it off in
the Matrix checklist of `hermes tools`, or run
`hermes tools disable matrix_image_packs --platform matrix`. A saved Matrix
toolset list that names individual toolsets and was saved before this toolset
existed does not include it; run
`hermes tools enable matrix_image_packs --platform matrix` to add it.

[image-packs-msc]: https://github.com/matrix-org/matrix-spec-proposals/blob/0923024fe35068874e59b5536b793dfcfcfc96c0/proposals/2545-emotes.md

### Media Limits

Hermes uploads and downloads Matrix images, files, audio, and video through Matrix media APIs. Multiple generated images are sent as one ordered logical batch, preserving captions and thread context across the batch.

By default, Matrix media over 100 MB is rejected before upload/download. Override with:

```bash
MATRIX_MAX_MEDIA_BYTES=104857600
```

Inbound media must use Matrix `mxc://` content URIs. Hermes rejects arbitrary
HTTP(S) media URLs in Matrix events to avoid turning a federated room into an
unrestricted downloader.

### Cross-Signing Verification (Recommended)

If your Matrix account has cross-signing enabled (the default in Element), set the recovery key so the bot can self-sign its device on startup. Without this, other Matrix clients may refuse to share encryption sessions with the bot after a device key rotation.

```bash
MATRIX_RECOVERY_KEY=EsT... your recovery key here
```

**Where to find it:** In Element, go to **Settings** → **Security & Privacy** → **Encryption** → your recovery key (also called the "Security Key"). This is the key you were asked to save when you first set up cross-signing.

On each startup, if `MATRIX_RECOVERY_KEY` is set, Hermes imports cross-signing keys from the homeserver's secure secret storage and signs the current device. This is idempotent and safe to leave enabled permanently.

If Hermes bootstraps a new Matrix recovery key, it never logs the raw key. Set
`MATRIX_RECOVERY_KEY_OUTPUT_FILE=/secure/path/matrix-recovery-key.txt` before
startup to write a generated key once with file mode `0600`; the file is not
overwritten if it already exists.

:::warning[Deleting the crypto store]
If you delete `~/.hermes/platforms/matrix/store/crypto.db`, the bot loses its encryption identity. Simply restarting with the same device ID will **not** fully recover — the homeserver still holds one-time keys signed with the old identity key, and peers cannot establish new Olm sessions.

Hermes detects this condition on startup and refuses to enable E2EE, logging: `device XXXX has stale one-time keys on the server signed with a previous identity key`.

**Easiest recovery: generate a new access token** (which gets a fresh device ID with no stale key history). See the "Upgrading from a previous version with E2EE" section below. This is the most reliable path and avoids touching the homeserver database.

**Manual recovery** (advanced — keeps the same device ID):

1. Stop Synapse and delete the old device from its database:
   ```bash
   sudo systemctl stop matrix-synapse
   sudo sqlite3 /var/lib/matrix-synapse/homeserver.db "
     DELETE FROM e2e_device_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_one_time_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_fallback_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM devices WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
   "
   sudo systemctl start matrix-synapse
   ```
   Or via the Synapse admin API (note the URL-encoded user ID):
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   Note: deleting a device via the admin API may also invalidate the associated access token. You may need to generate a new token afterward.

2. Delete the local crypto store and restart Hermes:
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # restart hermes
   ```

Other Matrix clients (Element, matrix-commander) may cache the old device keys. After recovery, type `/discardsession` in Element to force a new encryption session with the bot.
:::

:::info
If `mautrix[encryption]` is not installed or `libolm` is missing, the bot falls back to a plain (unencrypted) client automatically. You'll see a warning in the logs.
:::


## Home Room

You can designate a "home room" where the bot sends proactive messages (such as cron job output, reminders, and notifications). There are two ways to set it:

### Using the Slash Command

Type `/sethome` in any Matrix room where the bot is present. That room becomes the home room.
If your Matrix client intercepts slash commands, type `!sethome` instead.

### Manual Configuration

Add this to your `~/.hermes/.env`:

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

## Room allowlist (`allowed_rooms`)

Restrict the bot to a fixed set of Matrix rooms. When set, the bot responds in listed rooms and private bot chats. It ignores messages from other rooms, even if the bot is mentioned.

**Private bot chats are exempt** from this filter when exactly two users have joined, including the bot. Other rooms require an allowlist entry, even if a Matrix client marks them as direct.

```yaml
matrix:
  allowed_rooms:
    - "!abc123def456:matrix.example.org"
    - "!opsroom789:matrix.example.org"
```

Or via env var (comma-separated):

```bash
MATRIX_ALLOWED_ROOMS="!abc123def456:matrix.example.org,!opsroom789:matrix.example.org"
```

Behavior:

- Empty / unset → no restriction (default).
- Non-empty → room ID must be on the list. The check runs **before** any other gating (mention requirement, sender allowlist, etc.).
- Use the room's **internal ID** (`!abc...:server`), not its alias (`#room:server`). You can find a room's internal ID in Element via Room → Settings → Advanced.

See also: [admin/user slash command split](../../reference/slash-commands.md#permissions-and-adminuser-split).


:::tip
To find a Room ID: in Element, go to the room → **Settings** → **Advanced** → the **Internal room ID** is shown there (starts with `!`).
:::

## Commands in Matrix

Hermes supports the same gateway commands in Matrix that it supports on other
messaging platforms, including `/commands`, `/model`, `/stop`, `/queue`,
`/steer`, `/goal`, `/subgoal`, `/bg`, `/btw`, `/tasks`, and
`/yolo`.

Some Matrix clients reserve leading `/` for local client commands and may not
send unknown slash commands to the room. In that case, use `!` as a Matrix-safe
alias:

```text
!commands
!model
!model gpt-5.5 --provider openrouter
!queue continue with the next task
!stop
```

Hermes only normalizes `!command` when the command is known to the gateway, a
registered plugin command, or an installed skill command. Ordinary exclamations
such as `!important` remain normal chat messages.

## Per-Room Prompts

Use `matrix.channel_prompts` to give individual Matrix rooms persistent
instructions. Keys must be full internal room IDs—not room names, aliases,
user IDs, or Matrix thread event IDs. Matrix threads use the entry for their
containing room.

```yaml
matrix:
  channel_prompts:
    "!abc123:matrix.org": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "!def456:matrix.org": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

On every future turn in a matching room, Hermes injects the configured prompt
ephemerally. Changes take effect without a new session or history rewrite.
Unmatched rooms and blank or whitespace-only entries apply no prompt.

## Per-Room Skill Bindings

Use `matrix.channel_skill_bindings` to auto-load one or more installed skills
when a new session starts in a specific room:

```yaml
matrix:
  channel_skill_bindings:
    - id: "!abc123:matrix.org"
      skills:
        - arxiv
        - writing-plans
    - id: "!def456:matrix.org"
      skill: code-review
```

Binding IDs are full internal room IDs—not names, aliases, user IDs, or Matrix
thread event IDs. Matrix threads use their containing room's binding. Skills
load only at session start, so after changing a binding, run `/new` or wait for
the session to reset. Unmatched rooms and empty bindings load no skill. You can
combine a skill binding with `channel_prompts` for per-turn room instructions.

## Troubleshooting

### Bot is not responding to messages

**Cause**: The bot hasn't joined the room, `MATRIX_ALLOWED_USERS` doesn't include your User ID, `MATRIX_ALLOWED_ROOMS` doesn't include the room, or a room message did not mention the bot.

**Fix**: Invite the bot to the room — it auto-joins on invite. Verify your User ID is in `MATRIX_ALLOWED_USERS` (use the full `@user:server` format) and the room ID is in `MATRIX_ALLOWED_ROOMS` if that allowlist is configured. In rooms, mention the bot or add the room to `MATRIX_FREE_RESPONSE_ROOMS`. Restart the gateway.

### Bot joins rooms but silently drops every message (clock skew)

**Cause**: The host's system clock is set ahead of real time. The Matrix adapter applies a 5-second startup-grace filter (`event_ts < startup_ts - 5`) to ignore events replayed from initial sync. When the wall clock is ahead, every incoming event looks "older than startup" and is dropped before reaching the message handler — the bot appears connected but never replies. See [#12614](https://github.com/NousResearch/hermes-agent/issues/12614).

**Symptom**: Gateway log shows `Matrix: dropped N live events as 'too old' more than 30s after startup`.

**Fix**: Sync the host clock with NTP and restart the bot:

```bash
# Debian/Ubuntu
sudo timedatectl set-ntp true
timedatectl status   # confirm "System clock synchronized: yes"

# macOS
sudo sntp -sS time.apple.com
```

### "Failed to authenticate" / "whoami failed" on startup

**Cause**: The access token or homeserver URL is incorrect.

**Fix**: Verify `MATRIX_HOMESERVER` points to your homeserver (include `https://`, no trailing slash). Check that `MATRIX_ACCESS_TOKEN` is valid — try it with curl:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

If this returns your user info, the token is valid. If it returns an error, generate a new token.

### "mautrix not installed" error

**Cause**: The `mautrix` Python package is not installed.

**Fix**: Install it:

```bash
python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
```

Or with Hermes extras:

```bash
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
```

### Encryption errors / "could not decrypt event"

**Cause**: Missing encryption keys, `libolm` not installed, or the bot's device isn't trusted.

**Fix**:
1. Verify `libolm` is installed on your system (see the E2EE section above).
2. Make sure `MATRIX_ENCRYPTION=true` is set in your `.env`.
3. In your Matrix client (Element), go to the bot's profile -> Sessions -> verify/trust the bot's device.
4. If the bot just joined an encrypted room, it can only decrypt messages sent *after* it joined. Older messages are inaccessible.

### Upgrading from a previous version with E2EE

:::tip
If you also manually deleted `crypto.db`, see the "Deleting the crypto store" warning in the E2EE section above — there are additional steps to clear stale one-time keys from the homeserver.
:::

If you previously used Hermes with `MATRIX_ENCRYPTION=true` and are upgrading to
a version that uses the new SQLite-based crypto store, the bot's encryption
identity has changed. Your Matrix client (Element) may cache the old device keys
and refuse to share encryption sessions with the bot.

**Symptoms**: The bot connects and shows "E2EE enabled" in the logs, but all
messages show "could not decrypt event" and the bot never responds.

**What's happening**: The old encryption state (from the previous `matrix-nio` or
serialization-based `mautrix` backend) is incompatible with the new SQLite crypto
store. The bot creates a fresh encryption identity, but your Matrix client still
has the old keys cached and won't share the room's encryption session with a
device whose keys changed. This is a Matrix security feature -- clients treat
changed identity keys for the same device as suspicious.

**Fix** (one-time migration):

1. **Generate a new access token** to get a fresh device ID. The simplest way:

   ```bash
   curl -X POST https://your-server/_matrix/client/v3/login \
     -H "Content-Type: application/json" \
     -d '{
       "type": "m.login.password",
       "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
       "password": "***",
       "initial_device_display_name": "Hermes Agent"
     }'
   ```

   Copy the new `access_token` and update `MATRIX_ACCESS_TOKEN` in `~/.hermes/.env`.

2. **Delete old encryption state**:

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. **Set your recovery key** (if you use cross-signing — most Element users do). Add to `~/.hermes/.env`:

   ```bash
   MATRIX_RECOVERY_KEY=EsT... your recovery key here
   ```

   This lets the bot self-sign with cross-signing keys on startup, so Element trusts the new device immediately. Without this, Element may see the new device as unverified and refuse to share encryption sessions. Find your recovery key in Element under **Settings** → **Security & Privacy** → **Encryption**.

4. **Force your Matrix client to rotate the encryption session**. In Element,
   open the DM room with the bot and type `/discardsession`. This forces Element
   to create a new encryption session and share it with the bot's new device.

5. **Restart the gateway**:

   ```bash
   hermes gateway run
   ```

   If `MATRIX_RECOVERY_KEY` is set, you should see `Matrix: cross-signing verified via recovery key` in the logs.

6. **Send a new message**. The bot should decrypt and respond normally.

:::note
After migration, messages sent *before* the upgrade cannot be decrypted -- the old
encryption keys are gone. This only affects the transition; new messages work
normally.
:::

:::tip
**New installations are not affected.** This migration is only needed if you had
a working E2EE setup with a previous version of Hermes and are upgrading.

**Why a new access token?** Each Matrix access token is bound to a specific device
ID. Reusing the same device ID with new encryption keys causes other Matrix
clients to distrust the device (they see changed identity keys as a potential
security breach). A new access token gets a new device ID with no stale key
history, so other clients trust it immediately.
:::

## Proxy Mode (E2EE on macOS)

The `matrix` extra is gated to Linux. On macOS or Windows, run the Matrix
adapter and encryption dependencies in a Linux container and forward requests
to the native agent. The example below uses a macOS host; the same separation
applies to Windows with the corresponding host address and authentication.

### How It Works

```
macOS (Host):
  └─ hermes gateway
       ├─ api_server adapter ← listens on 0.0.0.0:8642
       ├─ AIAgent ← single source of truth
       ├─ Sessions, memory, skills
       └─ Local file access (Obsidian, projects, etc.)

Linux VM (Docker):
  └─ hermes gateway (proxy mode)
       ├─ Matrix adapter ← E2EE decryption/encryption
       └─ HTTP forward → macOS:8642/v1/chat/completions
           (no LLM API keys, no agent, no inference)
```

The Docker container only handles Matrix protocol + E2EE. When a message arrives, it decrypts it and forwards the text to the host via a standard HTTP request. The host runs the agent, calls tools, generates a response, and streams it back. The container encrypts and sends the response to Matrix. All sessions are unified — CLI, Matrix, Telegram, and any other platform share the same memory and conversation history.

### Step 1: Configure the Host (macOS)

Enable the API server so the host accepts incoming requests from the Docker container.

Add to `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=your-secret-key-here
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0` binds to all interfaces so the Docker container can reach it.
- `API_SERVER_KEY` is required for non-loopback binding. Pick a strong random string.
- The API server runs on port 8642 by default (change with `API_SERVER_PORT` if needed).

Start the gateway:

```bash
hermes gateway
```

You should see the API server start alongside any other platforms you have configured. Verify it's reachable from the VM:

```bash
# From the Linux VM
curl http://<mac-ip>:8642/health
```

### Step 2: Configure the Docker Container (Linux VM)

The container needs Matrix credentials and the proxy URL. It does NOT need LLM API keys.

**`docker-compose.yml`:**

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

Use the repository's [Docker build](../docker.md), which includes the
Matrix extra on compatible Linux targets and the required native libraries.
Do not install dependencies into the sealed image at runtime. The container
needs Matrix credentials and proxy access, not inference-provider API keys.

### Step 3: Start Both

1. Start the host gateway first:
   ```bash
   hermes gateway
   ```

2. Start the Docker container:
   ```bash
   docker compose up -d
   ```

3. Send a message in an encrypted Matrix room. The container decrypts it, forwards it to the host, and streams the response back.

### Configuration Reference

Proxy mode is configured on the **container side** (the thin gateway):

| Setting | Description |
|---------|-------------|
| `GATEWAY_PROXY_URL` | URL of the remote Hermes API server (e.g., `http://192.168.1.100:8642`) |
| `GATEWAY_PROXY_KEY` | Bearer token for authentication (must match `API_SERVER_KEY` on the host) |
| `gateway.proxy_url` | Same as `GATEWAY_PROXY_URL` but in `config.yaml` |

The host side needs:

| Setting | Description |
|---------|-------------|
| `API_SERVER_ENABLED` | Set to `true` |
| `API_SERVER_KEY` | Bearer token (shared with the container) |
| `API_SERVER_HOST` | Set to `0.0.0.0` for network access |
| `API_SERVER_PORT` | Port number (default: `8642`) |

### Works for Any Platform

Proxy mode is not limited to Matrix. Any platform adapter can use it — set `GATEWAY_PROXY_URL` on any gateway instance and it will forward to the remote agent instead of running one locally. This is useful for any deployment where the platform adapter needs to run in a different environment from the agent (network isolation, E2EE requirements, resource constraints).

:::tip
Session continuity is maintained via the `X-Hermes-Session-Id` header. The host's API server tracks sessions by this ID, so conversations persist across messages just like they would with a local agent.
:::

:::note
**Limitations (v1):** Tool progress messages from the remote agent are not relayed back — the user sees the streamed final response only, not individual tool calls. Dangerous command approval prompts are handled on the host side, not relayed to the Matrix user. These can be addressed in future updates.
:::

### Bot connects and sends, but ignores inbound messages

**Cause**: Matrix event handlers only fire when sync payloads are dispatched through
mautrix's `handle_sync()` machinery. A raw `client.sync()` poll that never calls
`handle_sync()` can leave the adapter connected (send works) while inbound
messages never reach `_on_room_message`.

**Fix**: Hermes uses an explicit sync loop that calls `client.handle_sync()` on
both the initial sync and every incremental sync response. This matches the
diagnosis in upstream issue #7914 and closed PR #37807, but keeps Hermes's own
background maintenance tasks (joined-room tracking, invite handling, E2EE key
share) instead of delegating the full lifecycle to `client.start()`. If inbound
messages still fail after a gateway restart, verify handlers are registered before
the first sync and check logs for `sync event dispatch error`.

### Sync issues / bot falls behind

**Cause**: Long-running tool executions can delay the sync loop, or the homeserver is slow.

**Fix**: The sync loop automatically retries every 5 seconds on error. Check the Hermes logs for sync-related warnings. If the bot consistently falls behind, ensure your homeserver has adequate resources.

### Bot is offline

**Cause**: The Hermes gateway isn't running, or it failed to connect.

**Fix**: Check that `hermes gateway` is running. Look at the terminal output for error messages. Common issues: wrong homeserver URL, expired access token, homeserver unreachable.

### "User not allowed" / Bot ignores you

**Cause**: Your User ID isn't in `MATRIX_ALLOWED_USERS`.

**Fix**: Add your User ID to `MATRIX_ALLOWED_USERS` in `~/.hermes/.env` and restart the gateway. Use the full `@user:server` format.

### Bot ignores an entire room

**Cause**: `MATRIX_ALLOWED_ROOMS` is set and the current room ID is not listed, or the room requires a mention and the message did not mention the bot.

**Fix**: Add the room ID to `MATRIX_ALLOWED_ROOMS`, or remove the room allowlist if this is a personal deployment. To find a Room ID in Element, open room settings and check **Advanced**.

### Bridge messages loop or echo

**Cause**: A bridge/appservice puppet is relaying bot output back as a new user message, or a bridge uses non-standard ghost user IDs.

**Fix**: Keep bridge ghosts out of `MATRIX_ALLOWED_USERS`, add a matching `MATRIX_IGNORE_USER_PATTERNS` entry, and leave `MATRIX_PROCESS_NOTICES=false` unless notices are part of a trusted workflow.

## Security

:::warning
Always set `MATRIX_ALLOWED_USERS` and, for shared/private deployments, `MATRIX_ALLOWED_ROOMS`. Without them, anyone who can message the bot in a joined room may trigger the agent. Only authorize people and rooms you trust — authorized users have full access to the agent's capabilities, including tool use and system access.
:::

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).

## Notes

- **Any homeserver**: Works with Synapse, Conduit, Dendrite, matrix.org, or any spec-compliant Matrix homeserver. No specific homeserver software required.
- **Federation**: If you're on a federated homeserver, the bot can communicate with users from other servers — just add their full `@user:server` IDs to `MATRIX_ALLOWED_USERS`.
- **Auto-join**: The bot automatically accepts room invites and joins. It starts responding immediately after joining.
- **Media support**: Hermes can send and receive images, audio, video, and file attachments. Media is uploaded to your homeserver using the Matrix content repository API.
- **Native voice messages (MSC3245)**: The Matrix adapter automatically tags outgoing voice messages with the `org.matrix.msc3245.voice` flag. This means TTS responses and voice audio are rendered as **native voice bubbles** in Element and other clients that support MSC3245, rather than as generic audio file attachments. Incoming voice messages with the MSC3245 flag are also correctly identified and routed to speech-to-text transcription. No configuration is needed — this works automatically. An audio file attached with a `MEDIA:` tag (an `.mp3`/`.wav` the agent produced, not marked `[[audio_as_voice]]`) is sent as a plain `m.audio` attachment in its original format instead of a voice bubble.
