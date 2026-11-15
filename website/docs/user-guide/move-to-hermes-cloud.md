---
sidebar_position: 12
title: Move an Agent
description: Carry a Hermes profile into another Hermes (a Hermes Cloud agent, a new computer), adding to it or replacing the parts you name, and undo it.
---

# Move an agent

`hermes move` carries one profile into another Hermes: a Hermes Cloud agent, or Hermes on a new computer. You export a bundle where the agent lives now, receive it on the other side, and can undo it there.

```bash
hermes move export -o hermes-move.zip                  # where the agent lives now
hermes move receive hermes-move.zip                    # on the other Hermes: add
hermes move receive hermes-move.zip --replace all      # or: the bundle wins
hermes move receive hermes-move.zip --replace memory,soul --only memory,soul,skills
hermes move list                                       # moves this profile received
hermes move undo <id>
```

To copy everything instead, keys included, use `hermes backup -o hermes-backup.zip` and `hermes import <file>`.

## What moves

| Moves | Stays behind |
|-------|--------------|
| Skills you wrote or installed from the hub | Built-in skills the other Hermes already has |
| Memory (`MEMORY.md`, `USER.md`) | API keys, bot tokens, OAuth logins (`.env`, `auth.json`) |
| Persona (`SOUL.md`) | Terminal, browser, gateway and other this-computer settings |
| Settings that are not machine-specific | MCP servers that run a program from your disk |
| Scheduled jobs (they arrive paused) | Conversations larger than 4 MB |
| Your most recent conversations (200 by default) | Skills larger than 25 MB |

Any value from `.env` or `auth.json` that appears inside a conversation, a skill or `SOUL.md` is replaced with `[redacted NAME]` before the bundle is written. The bundle file is readable only by you from its first byte, and skill scripts keep their executable bit. `hermes move export --dry-run` lists what would be carried; the bundle's manifest names the keys to enter again on the other side.

## Add or replace

`hermes move receive` shows what will change and asks before applying (`-y` skips the prompt).

- **Add** (the default) merges. The receiving agent's own settings, persona, skills and jobs win; memory is appended without duplicates; your conversations arrive as new chats.
- **Replace** (`--replace PARTS`, or `all`) makes the bundle win for each part you name: memory and `SOUL.md` are overwritten, same-named skills are swapped, the receiving agent's scheduled jobs are removed (yours arrive paused), its conversations are archived (kept, out of the list), and the bundle's settings win.

Parts are `config`, `soul`, `memory`, `skills`, `cron` and `sessions`. `--only PARTS` applies just those parts.

## Never touched

What the host set up for the receiving agent stays as it is in both modes: its Nous Portal sign-in, `model` and `providers`, fallbacks, `auxiliary`, the tool gateway, web, image and voice tools, the scheduler (`cron`), the session limits, and every key. The receiving agent keeps running while the move applies; conversations go through its own session store.

## Undo

`hermes move undo <id>` (ids from `hermes move list`) puts back what that move changed: every file it rewrote, the old job list and the conversations a replace archived, and it removes what the move added. It restores only what the move wrote: a file edited after the move is left alone and reported. Hermes records what it is about to change before changing it, so an interrupted replace can still be undone. An undo that runs into a problem exits non-zero and can be run again; it only redoes what is left.

## Keys and bots

Keys never travel; enter them again on the receiving side. A bot token (Telegram, Discord, iMessage…) can be held by only one gateway at a time: add it to the receiving agent when you are ready to switch, and remove it from the old one at the same time.

See [`hermes move`](../reference/cli-commands.md#hermes-move) for every flag.
