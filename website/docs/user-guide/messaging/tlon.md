---
sidebar_position: 18
title: "Tlon"
description: "Set up Hermes Agent as a Tlon ship adapter"
---

# Tlon Setup

Hermes can run as a Tlon ship and talk through Tlon DMs, chat channels,
notebooks, and galleries. The adapter connects to your ship over Eyre, listens
for Tlon events, and gives the agent a native `tlon` tool for group, channel,
role, gallery, contact, and message-history operations.

Use a dedicated bot ship when possible. The bot ship has real Tlon authority:
it can post, create groups, invite ships, manage channels, and assign roles.

## What Works

| Feature | Support |
|---------|---------|
| DMs | Responds to allowed ships |
| Group chats | Responds to mentions, participated threads, and owner messages when enabled |
| Channel discovery | Auto-discovers joined chat, notebook, and gallery channels |
| Blob reading | Downloads readable Tlon blob attachments into the agent context |
| Groups | Create, update, invite, join, leave, delete |
| Roles/admins | Create roles, assign roles, promote/demote admins |
| Galleries | Upload remote media to Tlon storage and post to heap/gallery channels |
| Notebooks | Create diary/notebook posts |
| History | Read DMs, channels, threads, and message context |

## Step 1: Gather Tlon Credentials

You need:

- `TLON_SHIP_URL`: the ship URL, for example `https://bot-palnet.tlon.network`
- `TLON_SHIP_NAME`: the bot ship patp, for example `~bot-palnet`
- `TLON_SHIP_CODE`: the ship login `+code`
- `TLON_OWNER_SHIP`: the human owner ship, for example `~zod`

:::warning
Keep the ship `+code` secret. Anyone with it can log in as that ship.
:::

## Step 2: Configure Hermes

### Option A: Setup Wizard

Run:

```bash
hermes gateway setup
```

Choose **Tlon**, then enter the ship URL, ship name, `+code`, owner ship, and
allowed ships. The wizard writes the values into your Hermes environment.

### Option B: Manual `.env`

Edit `~/.hermes/.env`:

```bash
TLON_SHIP_URL=https://bot-palnet.tlon.network
TLON_SHIP_NAME=~bot-palnet
TLON_SHIP_CODE=sampel-ticlyt-migfun-falmel

# Access control
TLON_OWNER_SHIP=~zod
TLON_ALLOWED_USERS=~zod
TLON_DEFAULT_AUTHORIZED_SHIPS=~zod

# Discover groups/channels the bot ship has joined
TLON_AUTO_DISCOVER=true

# Names that count as a mention in group channels
TLON_BOT_ALIASES=Hermes,Hermetic

# Let the owner talk in groups without mentioning the bot
TLON_OWNER_LISTEN_ENABLED=true
```

Hermes enables the Tlon platform automatically when `TLON_SHIP_URL`,
`TLON_SHIP_NAME`, and `TLON_SHIP_CODE` are all set.

## Step 3: Configure Model Credentials

The Tlon adapter only delivers messages. Hermes still needs a model provider.
If you have not configured one yet, run:

```bash
hermes setup model
```

or set your normal provider environment variables before starting the gateway.

## Step 4: Start the Gateway

For local debugging, run in the foreground:

```bash
hermes gateway run
```

For a background service:

```bash
hermes gateway install
hermes gateway start
```

Check status and logs:

```bash
hermes gateway status
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log
```

A healthy Tlon startup logs:

- authenticated ship name
- bot nickname, if available
- auto-discovered channel count
- monitored channel nests
- `Connected and listening`

## Step 5: Test DMs

From the owner ship, DM the bot ship:

```text
hello
```

The bot should answer without a mention. If it does not, check:

- `TLON_ALLOWED_USERS` includes your owner ship
- `TLON_OWNER_SHIP` is set
- the gateway log has a Tlon inbound DM event
- the ship URL and `+code` are current

The adapter also has a DM history fallback poller. It is enabled by default so
missed Tlon SSE DM events still get picked up:

```bash
TLON_DM_POLL_ENABLED=true
TLON_DM_POLL_INTERVAL=10
TLON_DM_POLL_INITIAL_CATCHUP_SECONDS=1800
```

## Step 6: Test Group Channels

Invite the bot ship to a Tlon group, or ask the bot to create one.

In group chat channels, Hermes responds when:

- the message mentions the bot ship, nickname, or one of `TLON_BOT_ALIASES`
- the message is in a thread where Hermes has already participated
- the sender is `TLON_OWNER_SHIP` and `TLON_OWNER_LISTEN_ENABLED=true`
- the owner sends a blob-only message that needs attachment handling

Examples:

```text
Hermes: summarize this channel
~bot-palnet what groups can you see?
```

Channel IDs are Tlon nests:

```text
chat/~host/general
diary/~host/notebook
heap/~host/gallery
```

If auto-discovery is off, specify channels manually:

```bash
TLON_AUTO_DISCOVER=false
TLON_CHANNELS=chat/~host/general,heap/~host/gallery
```

## Step 7: Create Groups and Make Admins

For "create a group and make me admin" requests, Hermes should use the
dedicated Tlon action `group_create_with_admins`. That path creates the group,
force-adds the requested ships as seats, assigns the `admin` role, and verifies
that the role is visible in group state.

Example prompt to the bot:

```text
Create a group called research and make ~zod an admin.
```

Expected behavior:

1. create the group
2. add/invite the admin ship
3. create the `admin` role if needed
4. mark the role as admin
5. assign the role to the target ship
6. verify the target ship has `["admin"]`

If an existing group needs repair, ask:

```text
Promote ~zod to admin in ~bot-palnet/research.
```

The adapter should use `group_promote`, not ask the user to join first.

## Step 8: Post to Galleries

Tlon galleries are heap channels:

```text
heap/~host/gallery-name
```

For images and files, use a reachable URL. Do not use a local file path as
gallery media because Tlon clients cannot render host-local files.

The reliable flow is:

1. start with a public or remote image URL
2. upload it to Tlon storage with the Tlon tool
3. post it to the heap/gallery channel with `gallery_post`

In natural language:

```text
Upload these image URLs to Tlon storage and post them to heap/~bot-palnet/cats.
```

For link/text gallery posts, the bot can post text or URLs directly to the heap
channel. For image posts, it should use the media URL and an optional caption.

## Step 9: Set a Home Channel

The home channel is where cron results, background notifications, and gateway
status messages go. For Tlon, use either a DM ship or a channel nest:

```bash
TLON_HOME_CHANNEL=~zod
# or
TLON_HOME_CHANNEL=chat/~host/general
```

Status messages are routed to the owner DM when possible so shutdown/restart
notices do not leak into shared groups.

## Access Control

Recommended personal setup:

```bash
TLON_OWNER_SHIP=~zod
TLON_ALLOWED_USERS=~zod
TLON_DEFAULT_AUTHORIZED_SHIPS=~zod
TLON_ALLOW_ALL_USERS=false
```

Use `TLON_ALLOW_ALL_USERS=true` only for disposable test ships. The agent can
use tools and operate the bot ship, so broad access is risky.

For group access, keep channels restricted by default and authorize specific
ships or channels through Tlon settings. Owner commands include:

```text
/pending
/approve <id>
/deny <id>
/block <id>
/unblock ~ship
```

## Troubleshooting

### Gateway starts but the bot does not respond

Check:

```bash
hermes gateway status
tail -100 ~/.hermes/logs/gateway.log
tail -100 ~/.hermes/logs/gateway.error.log
```

Look for Tlon authentication, channel discovery, and inbound DM/channel events.
If the message appears in Tlon history but not the live event stream, keep
`TLON_DM_POLL_ENABLED=true`.

### Bot responds in DMs but not groups

Verify:

- the bot ship has joined the group
- `TLON_AUTO_DISCOVER=true`, or the channel nest is in `TLON_CHANNELS`
- the message mentions the bot, unless owner-listen is enabled
- `TLON_BOT_ALIASES` includes the displayed bot name you are using

### Admin assignment does not stick

Use:

```text
Create a group called X and make ~zod admin.
```

or:

```text
Promote ~zod to admin in ~host/group.
```

The bot should use `group_create_with_admins` for new groups and
`group_promote` for existing groups. If it says you must accept an invite first,
that is the wrong workflow.

### Gallery posts report success but the gallery is empty

Use a heap channel (`heap/~host/gallery`) and the `gallery_post` path. Sending a
normal chat message to a heap channel may not create a visible gallery item.

### `group_info` or scry returns 404

Make sure you are passing the group flag, not only a channel nest:

```text
group: ~host/group-slug
channel: chat/~host/channel-slug
```

If you have a Tlon app URL, pass the full URL to `group_info`; the tool can
extract `groupId` and `channelId` from the query string.

## Maintenance

After changing `.env`, restart the gateway:

```bash
hermes gateway restart
```

After updating Hermes:

```bash
hermes update
hermes gateway restart
```

If you maintain local adapter changes, commit or stash them before running
`hermes update`, otherwise Git will refuse to overwrite modified files.
