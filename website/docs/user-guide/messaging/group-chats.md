---
sidebar_position: 2
title: "Group Chats from messaging"
description: "Read and control your gateway Group Chats with /group from a Telegram, Discord, Slack or other chat the owner allowed"
---

# Group Chats from messaging

When you're away from Hermes Desktop, `/group` lets you follow and steer your gateway
Group Chats from a messaging chat with one of your Bots: see what each group is doing,
post a message, stop work, and answer the Bots' approval requests.

It works with gateway Group Chats (the rooms the gateway runs, see
[Bot Mode](../bot-mode.md#groups-and-group-chats)), on the profile of the Bot you're
talking to. Classic rooms, which Desktop runs itself, aren't reachable from messaging.

## Who can use it

Two things must both be true, for every command:

1. **You allowed this exact chat**, once, on the computer where Hermes runs (below).
2. **The sender is on the Bot's admin list** for that kind of chat: `allow_admin_from` in a
   direct message, `group_allow_admin_from` in a group chat or channel. These are the same lists
   that [gate slash commands](./index.md#admins-vs-regular-users). An empty list allows nobody
   to use `/group`.

A *private* chat (you and the Bot alone) is allowed for you only. A *shared* chat (a
group, a channel, a thread, or a direct message that could hold more people) is allowed as a
whole: everyone in it can read what `/group` shows there, and only people on that chat's admin
list can use it.

Platforms whose direct messages can include several people, such as Slack and Matrix, are
always treated as shared (their direct messages still use `allow_admin_from`).

## Connect a chat

1. Send `/group` to the Bot in the chat. It answers with a one-time code.
2. On the computer where Hermes runs, with the gateway running:

   ```bash
   hermes groups allow K7Q2-M9XF
   ```

   The command shows exactly which chat the code belongs to and who will be able to read
   it, then asks you to confirm. The code works once and expires after ten minutes.
3. The chat is told it's connected. Send `/group list`.

You are the owner of a connected chat: it reaches the Group Chats your account owns,
the same ones you see in Desktop.

To see or remove connected chats:

```bash
hermes groups chats          # every connected chat, and what each always allows
hermes groups revoke 67c349dc
```

Revoking takes effect for the next command, and ends every approval that chat chose to
always allow.

## Commands

| Command | What it does |
| --- | --- |
| `/group` or `/group list [page]` | Your Group Chats, numbered. A number always means the same Group Chat in that chat. |
| `/group N` | Status, Bots, waiting approvals, what this chat always allows, and recent messages. |
| `/group N send <message>` | Post a message. It's recorded as yours, for example *Alice via Telegram*, never as typed in Desktop. |
| `/group N stop` | Stop the work in progress. |
| `/group N approve <code> once\|deny` | Answer one waiting approval; `/group N` shows its code. |
| `/group N approve <code> always` | Always allow that exact command for that Bot in that Group Chat, from this chat (see below). |
| `/group N forget <code>` | Stop always allowing a command. |
| `/group help` | The command list. |

`/group` also works while the chat's own conversation with the Bot is busy.

## Always allow in this chat

Some approval requests can be remembered: a terminal command run in the foreground,
locally or over SSH. For those, `/group N` offers `once|always|deny`.

`/group N approve <code> always` shows the exact command and the folder or connection it
runs in, and asks you to confirm with `... always confirm`. The request is then approved
once, and from then on that Bot may run **that exact command, in that folder or on that
connection**, again in that Group Chat without asking, for as long as this chat stays
connected. Anything different still asks: another command, another folder, another Bot,
another Group Chat.

It is never wider than that. It doesn't become a "session" or "always" permission for the
Bot's profile, and it ends when you forget it, when you revoke the chat, when the Group
Chat is disbanded or moves to another gateway, or when the Bot's setup in the Group Chat
changes. A command whose folder or connection changes while its approval is waiting is not
run at all.

## Good to know

- Room text is shown inert: mentions, links and formatting in Bot replies can't ping or
  trigger anything in your chat.
- If a change can't be confirmed (for example the gateway is restarting), `/group` says so
  instead of retrying. Check with `/group N` before trying again.
- Desktop doesn't yet label messages sent from messaging; the Group Chat's log records who
  sent them.
