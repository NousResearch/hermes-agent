---
sidebar_position: 8
title: "Backup & Transfer"
description: "Back up your whole agent, or replace a Hermes Cloud agent with it, from Settings"
---

# Backup & Transfer

**Settings ▸ Backup & Transfer** in Hermes Desktop (and the Webapp) does two things:

- **Back up** saves your whole agent to one file.
- **Transfer to Hermes Cloud** replaces one of your Hermes Cloud agents with this agent. You can undo it.

On a Webapp that Hermes Cloud itself serves, the section is called **Backup** and has only the backup. That tab already is a cloud agent.

## Back up

**Back up** runs [`hermes backup`](../reference/cli-commands.md#hermes-backup) on the agent Settings is showing, then asks where to save the zip. The zip holds the whole Hermes home: every profile, memory, persona, skills, conversations, scheduled jobs, settings, and your keys and logins (`.env`, `auth.json`). Keep it private. To restore it, run `hermes import <zip>`.

## Transfer to Hermes Cloud

1. Sign in to Nous Portal if the section asks you to. It then lists your Hermes Cloud agents.
2. Pick the agent to replace.
3. Read what gets replaced and what the agent keeps, tick **I understand this replaces &lt;agent&gt;**, and click **Replace &lt;agent&gt;**.

Hermes backs up this agent, uploads the backup to the cloud agent, and the cloud agent restores it with `hermes import --keep-host`. Before it changes anything, the cloud agent backs itself up. That backup is the undo point.

**What comes over:** memory, persona (SOUL.md), skills, conversations, scheduled jobs, settings, and your API keys and logins (other than Nous Portal).

**What the cloud agent keeps:** its Nous Portal sign-in, its model and provider, its Hermes Cloud tools (web, images, voice), its dashboard sign-in, and its scheduler settings. Hermes Cloud set these up, and they already work there.

**Messaging bots** (Telegram, Discord, Slack…) come over with their tokens. A bot token can only be used by one running Hermes at a time, so stop the gateway on this computer if you don't want both to answer.

**Undo** on the done screen restores the cloud agent from its undo point. **Start again** goes back to the agent list.

**New cloud agent** opens Nous Portal, where you create one. Hermes Desktop cannot create agents yet.

## From the command line

```bash
hermes backup -o my-agent.zip                 # on this computer
# copy my-agent.zip to the cloud agent, then on it:
hermes import my-agent.zip --keep-host        # prints the undo point
hermes import <undo point> --force --keep-host   # undo
```

[`hermes import --keep-host`](../reference/cli-commands.md#--keep-host-replace-an-agent-but-keep-its-hosts-setup) lists exactly what it keeps.
