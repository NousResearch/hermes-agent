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

Direct messages that can include several people, such as a Slack group DM or any Matrix
direct room, are treated as shared (they still use `allow_admin_from`). A one-to-one Slack
DM is private.

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
| `/group N continue` | When the group's host is offline, continue the group on this computer (see below). |
| `/group N keep <computer>` | For a group continued on two computers, choose the one that keeps it. After a careful move, go back to the computer it left (confirmed with `... confirm`). |
| `/group N ask first` | The group asks you before it moves by itself again. |
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

## When a group's host goes offline

A Group Chat runs on one computer, its *host*. Your other computers can keep a full copy of
the group, and the ones you allow in Hermes Desktop can continue it if the host goes offline.
On a gateway that supports this, `/group N` shows it:

- `Host: Mac mini.` while all is well, or `Host: Mac mini, offline since 14:05 CEST. Paused.`
  when the host can't be reached. Nothing new runs in a paused group.
- `Can continue on: Home VPS, MacBook.`, or
  `No computer can continue this group yet; choose one in Hermes Desktop.`
  Messaging shows which computers can continue a group; you choose them in Desktop.
- Whether the group moves by itself:
  `Keeps running if a computer goes offline: ready (Home VPS takes over).` With only two
  computers it waits first: `… (Home VPS takes over after about 3 minutes).` Otherwise it says
  why not right now (`Not automatic right now: MacBook offline.`), not yet
  (`Not automatic yet: add one more always-on computer in Hermes Desktop.`), or
  `Moves only when you choose.`
- In `/group list`, a group another computer hosts is marked *backup copy*. `/group N` shows
  it, but sending and stopping work happen on its host.

`/group N continue` continues a paused group on **the computer whose Bot you're talking to**.
It first shows what that involves: which Bots stay unavailable until the group moves back to
the computer they run on, how much work finished, still runs on other computers or is
unknown, and which recent messages this computer doesn't have yet. To go ahead, reply
`/group N continue confirm`, or tap **Continue on …** where your chat shows buttons. Unknown
work never runs again by itself.

Only continue if the host is really offline. If it's still running somewhere you can't
reach, both computers may keep working until they reconnect. `/group N` then says the group
was continued on two computers, and `/group N keep <computer>` chooses the one that keeps
it; messages from the other are kept and shown separately.

Continuing and keeping belong to the group's owner. They work only in your private chat with
the Bot, never in a shared chat, which still shows the group's state. A computer that can
continue a paused group may also message you in that private chat with the command to reply.

When a group moves by itself, the computer it moved to tells you once:
`“Research” moved to Home VPS because Mac mini went offline. It’s running.` A host that can't
reach enough of the group's other computers pauses the group so that it never runs in two
places, and tells you the same way: `“Research” is paused to stay safe: Mac mini can’t reach
Home VPS.` It resumes as soon as one of them is back; continuing it anyway is only possible in
Hermes Desktop and the CLI. A message you send to a paused group isn't sent, and the reply says
so. While the other computers are still deciding which one takes over, `/group N` says that
too.

A group on just two computers moves more carefully: the other computer takes over only after
the host has been silent for about 3 minutes. Silence can't prove the host stopped, so you get
a warning instead, with three choices: **Keep going** on the new host (nothing to do), **Go
back** to the computer it left (`/group N keep <computer>`, then `... confirm`), or **Ask me
first next time** (`/group N ask first`). If both computers did run it,
`“Research” ran on both Mac mini and Home VPS while they couldn’t reach each other (…)`
says which one is still running the group and asks you to choose one with
`/group N keep <computer>`; keeping the one still running is keep going, and the other's
messages are kept separately.

These messages go to your main channel: your home channel (`/sethome`) when it is your
private chat with the Bot, otherwise each of your private chats with it. Where your chat app
has buttons, the choices are buttons too. Without any private chat, a home channel that is a
one-to-one chat gets the message without the choices ("Choose in Hermes Desktop."), and a
shared home channel gets nothing.

On Matrix every direct room counts as shared, so host-loss notices aren't sent there and
continuing, keeping or asking first isn't possible from Matrix: use Hermes Desktop or
`hermes groups`.

## Good to know

- Room text is shown inert: mentions, links and formatting in Bot replies can't ping or
  trigger anything in your chat.
- If a change can't be confirmed (for example the gateway is restarting), `/group` says so
  instead of retrying. Check with `/group N` before trying again.
- Desktop doesn't yet label messages sent from messaging; the Group Chat's log records who
  sent them.
