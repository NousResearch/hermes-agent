# SimpleX Chat

[SimpleX Chat](https://simplex.chat/) is a private, decentralised messaging platform where users own their contacts and groups. Unlike other platforms, SimpleX assigns no persistent user IDs — every contact is identified by an opaque internal ID generated at connection time, which makes it one of the most private messengers available.

> Run `hermes gateway setup` and pick **SimpleX** for a guided walk-through.

## Prerequisites

- The **simplex-chat** CLI installed and running as a daemon
- Python package **websockets** (`pip install websockets`)

## Install simplex-chat

Download the latest release from the [simplex-chat GitHub releases](https://github.com/simplex-chat/simplex-chat/releases) page:

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86_64 -o simplex-chat
chmod +x simplex-chat
```

The SimpleX Chat project does not publish a prebuilt Docker image for the chat client; to run it under Docker, build from source from the [simplex-chat repository](https://github.com/simplex-chat/simplex-chat).

## Start the daemon

```bash
simplex-chat -p 5225
```

The daemon listens on WebSocket at `ws://127.0.0.1:5225` by default.

## Configure Hermes

### Via setup wizard

```bash
hermes gateway setup
```

Select **SimpleX Chat** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| Variable | Required | Description |
|---|---|---|
| `SIMPLEX_WS_URL` | Yes | WebSocket URL of the simplex-chat daemon |
| `SIMPLEX_ALLOWED_USERS` | Recommended | Comma-separated allowlist. Each entry can be a numeric `contactId` **or** a display name — both forms work. |
| `SIMPLEX_ALLOW_ALL_USERS` | Optional | Set `true` to allow every contact (use carefully) |
| `SIMPLEX_AUTO_ACCEPT` | Optional | Auto-accept incoming contact requests (default: `true`) |
| `SIMPLEX_GROUP_ALLOWED` | Optional | Comma-separated group IDs the bot participates in, or `*` for any group. Omit to ignore group messages entirely |
| `SIMPLEX_HOME_CHANNEL` | Optional | Default contact/group ID for cron job delivery |
| `SIMPLEX_HOME_CHANNEL_NAME` | Optional | Human label for the home channel |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | Optional | Quiet-period seconds (default: `0.8`) used to concatenate rapid-fire inbound text messages into one event |

## Find your contact ID or display name

After starting the daemon, open a conversation with your agent contact. The numeric `contactId` appears in session logs. If you'd rather use the display name shown in the SimpleX UI, that works too — `SIMPLEX_ALLOWED_USERS` accepts either form.

## Authorization

By default **all contacts are denied**. You must either:

1. Set `SIMPLEX_ALLOWED_USERS` to a comma-separated list of `contactId`s and/or display names (e.g. `SIMPLEX_ALLOWED_USERS=4,alice` matches either contactId 4 or the contact whose display name is "alice"), or
2. Use **DM pairing** — send any message to the bot and it will reply with a pairing code. Enter that code via `hermes pairing approve simplex <CODE>`.

## Group chats

By default the adapter ignores group messages — a bot in a group otherwise
processes every member's traffic. Opt-in explicitly:

```
SIMPLEX_GROUP_ALLOWED=12,34          # specific group IDs
# or
SIMPLEX_GROUP_ALLOWED=*              # any group the bot is in
```

Address groups by prefixing the chat ID with `group:`, e.g.
`simplex:group:12` as a cron `deliver=` target or in a `hermes send` call.

## Sending with `hermes send`

SimpleX works as a standalone send target — the daemon must be running,
but a live gateway is not required for plain text:

```bash
hermes send --to simplex:alice "hello"          # DM by contact display name
hermes send --to simplex:group:12 "hello"       # group by numeric ID
hermes send --to simplex "hello"                # SIMPLEX_HOME_CHANNEL
```

While the gateway is running, the adapter enumerates your contacts and
allowed groups into the channel directory (refreshed every 5 minutes), so
`hermes send --list` shows them by name. Before the first gateway run the
platform still appears in `--list` with a "no channels discovered yet"
hint — direct targets like the ones above work regardless.

## Attachments

The adapter supports native SimpleX attachments in both directions:

- **Inbound** — incoming images, voice notes, and files are accepted via
  the daemon's XFTP flow (`rcvFileDescrReady` → `/freceive` → wait for
  `rcvFileComplete`) and surfaced as `MessageEvent.media_urls` with the
  appropriate `MessageType` (`PHOTO`, `VOICE`, `TEXT` + document).
- **Outbound** — `send_image_file`, `send_voice`, `send_document`, and
  `send_video` all use the structured `/_send` form with `filePath`, so
  the receiving SimpleX client renders images inline and plays voice
  notes inline rather than offering them as downloads.

Agent replies can also embed `MEDIA:/path/to/file` tags in plain text —
the adapter strips the tag from the body and sends the file as either a
voice note (audio extensions) or a document.

## Exec approval

When the agent wants to run a potentially dangerous command it asks first.
The prompt always shows the command, the reason, and the typed commands
that answer it: `/approve`, `/approve session`, `/approve always`, `/deny`.

**In a direct chat**, when exactly one approval is waiting, the bot also
pre-places three decision emoji on its own message so you can answer with a
long-press and a tap instead of typing, then follows the prompt with a short
line naming the taps it managed to place:

| tap | means |
|---|---|
| 👎 | deny |
| ✅ | approve once |
| 🚀 | approve for this session |

Three, not four, because the simplex-chat daemon holds at most **three
reactions per sender per item** — a fourth comes back as "too many
reactions". The limit is per sender, so it never stops you from adding your
own reaction to a message the bot has already filled. 👎 is placed first, so
if a slot is ever short it is never the refusal that goes missing.

Two more reactions are read but never pre-placed:

| reaction | means |
|---|---|
| 👍 | approve once |
| ❤️ | approve always |

Nothing else is recognised, and a reaction is only accepted for a tier the
prompt actually offered — reacting ❤️ to a prompt that did not offer the
permanent tier gets you a short reply, not a permanent approval. The emoji
are not the ones other Hermes platforms use: the simplex-chat daemon
validates reactions against a fixed set, so these are what SimpleX allows.

### "Approve always" is typed-only, on purpose

**approve always** writes the command pattern into your permanent allowlist
on disk. It applies to that pattern everywhere Hermes runs it — not just this
chat, not just this platform, and not just today. That is the most
consequential thing an approval prompt can do, so it costs a deliberate
`/approve always` rather than one tap on an emoji sitting next to "deny".
Taps are for the two decisions that expire on their own: this once, or this
session.

If you do want it as a reaction, ❤️ still resolves to approve-always when you
place it yourself. It is simply never offered.

The legend the bot posts under a prompt lists only the taps that are really
on the message. If the daemon refused one, or the slots were full, the
missing one is not advertised — you will never be told to tap something that
is not there.

### When you get a typed prompt instead

The tap lane is only offered when a tap can mean exactly one thing. These
cases fall back to typed commands, which the gateway authorizes in full:

- **Group chats.** Any member can react, and the identity the daemon
  reports for a group reactor is not one this adapter can tie to a verified
  operator, so v1 keeps reactions to direct chats.
- **Two approvals at once in the same session.** `resolve_gateway_approval`
  resolves a session's approvals oldest-first and cannot be pointed at a
  specific one (upstream issue #64001), so a tap could not say *which*
  command you meant. When a second approval arrives, the bot withdraws the
  reactions from the first message, tells you it was superseded, and sends
  the new prompt typed-only. Answer both with `/approve` or `/deny`. That
  session keeps getting typed prompts until the approval window lapses —
  answering by typing happens inside the gateway, so the adapter cannot see
  that the queue drained.
- **A prompt that could not be delivered or anchored.** If the connection to
  the daemon is down, the send fails, or the daemon never answers it, the
  approval is still waiting inside the agent even though no tappable message
  exists for it. That session gets typed prompts for one approval window,
  rather than a tap that would answer the earlier command you never saw.
- **After a gateway restart.** Prompts live in memory. Reactions left on a
  message from before a restart do nothing — type `/approve` instead.
- **A daemon that will not let the bot place reactions.** The bot says so
  once and stops pre-placing them; typed commands keep working, and a
  reaction you place yourself is still read.

`SIMPLEX_ALLOW_ALL_USERS` controls who may talk to the bot. It does not
grant anyone the right to approve a command.

A prompt stays tappable for as long as your `approvals.timeout` setting
allows. After that a tap gets an "expired" reply rather than silently doing
nothing, and a tap that lands after the command was already answered is
told the command did not run.

## Using SimpleX with cron jobs

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Or target a specific contact via the cron job's `deliver:` field, or from a shell script with the [`hermes send` CLI](/guides/pipe-script-output):

```bash
hermes send simplex:<contact-id> "Done!"
```

## Privacy notes

- SimpleX never reveals phone numbers or email addresses — contacts use opaque IDs
- The connection between Hermes and the daemon is local WebSocket (`ws://127.0.0.1:5225`) — no data leaves your machine
- Messages are end-to-end encrypted by the SimpleX protocol before reaching the daemon

## Troubleshooting

**"Cannot reach daemon"** — Ensure `simplex-chat -p 5225` is running and the port matches `SIMPLEX_WS_URL`.

**"websockets not installed"** — Run `pip install websockets`.

**Messages not received** — Check that the contact's ID is in `SIMPLEX_ALLOWED_USERS` or approve them via DM pairing.
