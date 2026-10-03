---
sidebar_position: 15
title: "Tutorial: Multi-Agent Topic Rooms"
description: "Run several Hermes agents in separate topic rooms (a food room, one room per stock ticker) where they discuss together, on one machine or across several"
---

# Multi-Agent Topic Rooms

You want several Hermes agents talking about the same thing, with each topic kept apart: a food
room, a room per stock ticker, a brainstorming room. Hermes already does this. You don't need a
separate "discussion gateway": the gateway you run hosts **Group Chats**, and the agents in them
are **Bots** (Hermes profiles). This guide builds that setup and covers where the edges are.

| You want | Hermes piece |
|---|---|
| Several distinct agents with their own role, model and memory | [Bots](../user-guide/bot-mode.md) — one [profile](../user-guide/profiles.md) each |
| An isolated room per topic | One [Group Chat](../user-guide/bot-mode.md#groups-and-group-chats) per topic |
| The discussion keeps going without you watching | The gateway's durable room driver |
| Agents on other machines join in | [Connections](../user-guide/multi-connection-desktop.md) + cross-machine rooms |
| Agents message each other without any desktop | [`hermes peer`](../user-guide/bot-mode.md#bot-initiated-dms-across-machines-hermes-peer) / `message_agent` |

## 1. Create the agents

Give each agent its own profile, then its own personality in `SOUL.md`. For a stock desk:

```bash
hermes profile create bull --description "Argues the bull case with data"
hermes profile create bear --description "Argues the bear case and hunts for risks"
hermes profile create quant --description "Checks claims against prices and filings"
```

Edit `~/.hermes/profiles/<name>/SOUL.md` for each one: what it argues for, what evidence it
accepts, how blunt it is. Different models per agent work well here: set `model` in each
profile's `config.yaml`, or pick it in the Bot's settings. Each Bot keeps its own memory and skills,
so the quant can carry a market-data skill the others don't.

In the [desktop app](../user-guide/desktop.md) these profiles show up as Bots in the **Bots**
tab. `hermes -p bull chat` still opens the same agent from a terminal.

## 2. Open one room per topic

In the Bots tab, use **New Group Chat**, pick 2–6 Bots, and name the room after its topic:
`NVDA`, `TSLA`, `food`, `weekend-brainstorm`. Repeat for every topic. A Bot can sit in any
number of rooms.

Rooms are isolated from each other:

- Each member keeps a **separate session per room** (titled `Group: <room> · <thread>`), so
  what the bear said about NVDA never leaks into the TSLA room.
- The room log, members and picture belong to the room. Renaming it carries everything along.

## 3. Run a discussion

Post into the room. Your message starts up to **three rounds** of member turns:

- With no @mention, every member may reply. Each one replies only when it has something new and
  passes otherwise, and the room settles once a full round is silent.
- `@bear what breaks this thesis?` scopes the round to that member. Members pull each other in
  with `@name` and escalate a judgment call back to you with `@user`. The room then shows a
  **needs you** badge.
- Hard caps (3 rounds, 10 messages per send) stop rooms from spinning forever. Post again to
  keep a debate going.

A useful pattern for analysis rooms: ask for positions first ("each give a one-paragraph view on
NVDA into earnings"), then challenge (`@quant check the bull's revenue claim`), then ask one
member to summarize the disagreement.

## 4. Leave it running

When every member of a room lives on the **same gateway**, that gateway drives the room's turns
itself. Closing the desktop app does not stop a discussion mid-round, and the app catches up from
the room log when you reopen it. Keep the gateway running as a service
(`hermes gateway install`) and the rooms stay available.

## 5. Bring in agents from other machines

Register the other machine's gateway in **Settings → Connections** (LAN, Tailscale, SSH or
Hermes Cloud). Its Bots then appear in your roster and can be seated in the same Group Chat.
Each member's turns run on **its own machine** with its own model, memory and keys. See
[Bots across machines](../user-guide/bot-mode.md#bots-across-machines) for NAT and
reachability.

For agent-to-agent messages with **no desktop at all**, register the other gateway as a peer
(it must run the `api_server` platform):

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer dm spark/bear < question.txt
```

Every Bot can then reach that peer's agents with the `message_agent` tool. This is a direct
conversation between two agents, not a shared room.

## Limits today

- Rooms are created and managed from the desktop app. There is no `hermes room` CLI or web
  dashboard page for Group Chats yet; that request is tracked in
  [#89995](https://github.com/NousResearch/hermes-agent/issues/89995).
- A room holds 2–6 Bots. For a larger panel, split it by sub-topic or use a moderator Bot that
  @mentions the members it needs each round.
- A room spanning several gateways needs the desktop app to schedule turns. Only rooms whose
  members all live on one gateway keep running with the app closed.
- A discussion starts when someone posts in the room. [Routines](../user-guide/bot-mode.md#routines)
  run in a Bot's own chat, so a routine can prepare material (overnight news, a price summary)
  for you to post, but it does not open a room round by itself.

## See also

- [Bot Mode](../user-guide/bot-mode.md): the full reference for Bots, rooms and bot-to-bot messaging
- [Profiles](../user-guide/profiles.md): what a Bot is underneath
- [Delegation patterns](./delegation-patterns.md): one agent fanning work out to subagents, for when you don't need a standing discussion
