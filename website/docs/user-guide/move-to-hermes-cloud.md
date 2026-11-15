---
sidebar_position: 12
title: Backup & Move
description: Back up your whole Hermes agent to a file, or carry it into a Hermes Cloud agent, merging or replacing what is there.
---

# Backup & Move

**Backup & Move** in Hermes Desktop (sidebar, or **⌘K → Back up this agent / Move to Hermes Cloud**) does two things:

- **Back up** writes one file with your whole Hermes: every profile, settings, memory, skills, schedules, chat history and keys. It is `hermes backup`, saved where you choose. The file holds your keys, so keep it private; `hermes import <file>` restores it.
- **Move to Hermes Cloud** copies this agent into one of your Hermes Cloud agents, so it knows what your desktop agent knows from its first message. You need to be signed in to the Nous Portal. It also opens from the empty state of a new agent in the Nous Portal.

On the Hermes Webapp that Hermes Cloud serves, the page offers Backup only: that tab already is a cloud agent. A Webapp you host yourself gets the whole page; it lists your cloud agents through that install's own Nous Portal sign-in (**Settings → Providers → Nous Portal**).

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

## Add or replace

After the upload, choose how the bundle lands:

- **Add to the cloud agent** (the default) merges. The cloud agent's own settings, persona, skills and jobs win; memory is appended without duplicates; your conversations arrive as new chats.
- **Replace the cloud agent** makes your copy win for each part you tick: memory and persona are overwritten, same-named skills are swapped, the cloud agent's scheduled jobs are removed (yours arrive paused), its conversations are archived (kept, out of the list), and your settings replace its own. You confirm with an "I understand" tick before **Replace** is enabled.

Either way, what Hermes Cloud set up for the agent is never touched: its Nous Portal sign-in, model and provider, fallbacks, the managed tool gateway, web, image and voice tools, the scheduler, the session limits, and every key.

## How it works

1. Desktop builds the bundle on your computer and shows what it will carry.
2. It uploads the bundle to the cloud agent in small pieces with a checksum, so a dropped connection resumes where it stopped.
3. The cloud agent shows what will change and waits for you to choose Add or Replace.
4. The cloud agent applies the bundle while it keeps running; conversations go through its own session store, never around it.
5. Desktop lists the keys to enter again in the cloud agent's settings.

A move can be undone from the same screen, or with `hermes move undo <id>` on the agent. Undo restores every file the move rewrote, the old job list, and unarchives the conversations a replace set aside.

## New cloud agents

Create the agent in the Nous Portal (the **New cloud agent** row links there), then come back and move into it.

## Keys and bots

Keys never travel. The checklist after a move links each one to the cloud agent's settings. A bot token (Telegram, Discord, iMessage…) can only be held by one gateway at a time: add it to the cloud agent when you are ready to switch, and remove it from this computer at the same time.

## From the command line

```bash
hermes backup -o hermes-backup.zip                 # everything, keys included
hermes move export -o hermes-move.zip              # on this computer
hermes move receive hermes-move.zip                # on the other Hermes: merge
hermes move receive hermes-move.zip --replace all  # or: the bundle wins
hermes move undo <id>
```

See [`hermes move`](../reference/cli-commands.md#hermes-move).
