---
sidebar_position: 12
title: Move to Hermes Cloud
description: Carry your Desktop agent's skills, memory, persona, schedules and conversations to a Hermes Cloud agent.
---

# Move to Hermes Cloud

**Move to Hermes Cloud** copies the agent you run in Hermes Desktop into a Hermes Cloud agent, so the cloud agent knows what your desktop agent knows from its first message. Open it from **⌘K → Move to Hermes Cloud**, or from the empty state of a new agent in the Nous Portal.

## What moves

| Moves | Stays on this computer |
|-------|------------------------|
| Skills you wrote or installed from the hub | Built-in skills the cloud agent already has |
| Memory (`MEMORY.md`, `USER.md`) | API keys, bot tokens, OAuth logins (`.env`, `auth.json`) |
| Persona (`SOUL.md`) | Terminal, browser, and other this-computer settings |
| Settings that are not machine-specific | MCP servers that run a program from your disk (listed in the preview) |
| Scheduled jobs (they arrive paused) | Conversations larger than 4 MB |
| Your most recent conversations | Skills larger than 25 MB |

Any key value that appears inside a conversation or a skill file is replaced with `[redacted NAME]` before the bundle leaves your computer.

## How it works

1. Desktop builds the bundle on your computer and shows what it will carry.
2. It uploads the bundle to the cloud agent in small pieces with a checksum, so a dropped connection resumes where it stopped.
3. The cloud agent shows what will change and waits for you to confirm.
4. The cloud agent merges the bundle while it keeps running. Its own settings, persona, skills and jobs always win; memory is appended without duplicates; conversations arrive as new chats.
5. Desktop lists the keys to enter again in the cloud agent's settings.

A move can be undone from the same screen, or with `hermes move undo <id>` on the agent.

## Keys and bots

Keys never travel. The checklist after a move links each one to the cloud agent's settings. A bot token (Telegram, Discord, iMessage…) can only be held by one gateway at a time: add it to the cloud agent when you are ready to switch, and remove it from this computer at the same time.

## From the command line

```bash
hermes move export -o hermes-move.zip     # on this computer
hermes move receive hermes-move.zip       # on the other Hermes
hermes move undo <id>
```

See [`hermes move`](../reference/cli-commands.md#hermes-move).
