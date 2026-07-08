# Telegram Secretary Mode (Chat Automation)

Secretary Mode allows a Telegram bot to receive and respond to messages sent to
your **personal account** — not to the bot directly. Clients DM you as usual;
the bot processes the message behind the scenes and can reply as you.

This is powered by Telegram's "Chat Automation in Profiles" feature (also known
as "Secretary Bots" or "Connected Business Bots"), introduced in May 2026.

## Setup

### 1. Enable Secretary Mode for your bot

Message **@BotFather** → `/mybots` → select your bot → **Bot Settings** →
**Secretary Mode** → **Turn On**.

This sets the `bot_business` flag that allows the bot to be connected to a user
account.

### 2. Connect the bot to your account

In Telegram: **Settings → Chat Automation** → connect your bot.

Configure which chats the bot can access:
- **New chats** — only conversations started after the connection
- **Non-contacts** — only people not in your contact list
- **Exclude contacts** — keep friends/family manual, automate the rest

The bot only sees private 1:1 chats — **group chats are not supported** by
Secretary Mode. For group automation, add the bot as a group member separately.

### 3. No special Hermes config needed

Hermes processes business_message updates automatically when they arrive. The
updates are already requested via `allowed_updates=ALL_TYPES`. Business messages
bypass the user-ID allowlist because they're pre-authorized by your Chat
Automation connection.

## How it works

When a client DMs your personal account, Telegram routes the message to your
connected bot as a `business_message` update. Hermes:

1. **Receives** the update (already requested via `allowed_updates`)
2. **Bypasses user-ID auth** — the message carries a server-set
   `business_connection_id` proving you authorized it
3. **Processes** the message through the normal agent loop
4. **Replies** using `business_connection_id` so the response appears to come
   from your personal account

Messages sent via business connection carry a `via_business_connection` flag
in Telegram's metadata. In your chat, a banner shows the bot manages this
conversation. From the client's side, the reply appears to come from you.

## Constraints

- **Private chats only** — Secretary Mode cannot access groups or channels
- **One bot per account** — Telegram allows only one connected business bot
- **No history backfill** — the bot only sees messages from connection time
  forward
- **24-hour reply window** — `can_reply` permission is limited to chats active
  in the last 24 hours (Telegram Business API constraint)
