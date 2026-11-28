---
sidebar_position: 18
title: "Use Hermes on Upstash Box over SSH"
description: "Install Hermes Agent in an Upstash Box and connect to it from Hermes Desktop over SSH"
---

# Use Hermes on Upstash Box over SSH

[Upstash Box](https://upstash.com/docs/box/overall/quickstart) is a managed Linux environment with SSH access. This guide installs Hermes in a Box and connects Hermes Desktop to it over SSH. Desktop opens the tunnel and starts `hermes serve` in the Box for you, so there is no service to install or keep running.

Your install, config, sessions and memory stay in the Box's `~/.hermes` between connections.

## Prerequisites

- [Upstash account](https://console.upstash.com)
- [Hermes Desktop](../user-guide/desktop.md) on your machine

## 1. Create a keep-alive Box

In the [Upstash Console](https://console.upstash.com), open **Box** and create a Box with the **Fixed (keep-alive)** plan and the **Small** size. Note the Box ID, for example `right-flamingo-14486`.

## 2. Add an SSH key

Desktop connects with an SSH key and never prompts for a password. Create a key and add it to your Upstash account as described in [SSH keys](https://upstash.com/docs/box/overall/shell#ssh-keys):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/upstash_box
```

Then add the contents of `~/.ssh/upstash_box.pub` under **Box → Settings → SSH Keys** in the console.

Check that the key works:

```bash
ssh -i ~/.ssh/upstash_box <box-id>@us-east-1.box.upstash.com
```

## 3. Install Hermes in the Box

In that SSH session, run the install script and follow the prompts:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

## 4. Connect from Hermes Desktop

In Hermes Desktop, go to **Settings → Gateways → This window** and choose **Connect via SSH**. Fill in:

- **Host:** `us-east-1.box.upstash.com`
- **User:** your Box ID
- **Identity file:** `~/.ssh/upstash_box`

Leave **Port** and **Hermes path** blank, then click **Save and reconnect**. Desktop starts `hermes serve` in the Box and connects to it through the tunnel. Send a message to check that the agent answers.

### As a registered SSH gateway

The **SSH** connection kind under **Registered gateways** ([Connecting Desktop to Many Hermes Instances](../user-guide/multi-connection-desktop.md)) has no identity file field. Add the Box to `~/.ssh/config` instead:

```
Host hermes-box
  HostName us-east-1.box.upstash.com
  User <box-id>
  IdentityFile ~/.ssh/upstash_box
  IdentitiesOnly yes
```

Then use `hermes-box` as the **SSH host**.

## Next steps

- [Configure an LLM provider](../integrations/providers.md)
- [Work with skills](./work-with-skills.md)
