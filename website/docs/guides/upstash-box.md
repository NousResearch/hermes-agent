---
sidebar_position: 18
title: "Run Hermes on Upstash Box"
description: "Set up Hermes Agent on an Upstash Box — a managed keep-alive Linux environment with full SSH access and no server administration overhead"
---

# Run Hermes on Upstash Box

[Upstash Box](https://upstash.com/docs/box/overall/quickstart) is a managed Linux environment that stays alive around the clock. You get full SSH access without having to manage a VPS.

:::note
Hermes is resource-intensive. Use a **Medium** box to ensure a smooth installation.
:::

## Prerequisites

- [Upstash account](https://console.upstash.com)
- SSH client on your local machine

## Setup

### 1. Create a keep-alive Box

Log in to the [Upstash Console](https://console.upstash.com), open **Box**, and create a new Box with:

- **Size:** Medium
- **Keep-Alive:** enabled

Note your Box ID (for example, `right-flamingo-14486`) and your [Box API key](https://upstash.com/docs/box/overall/quickstart#1-get-your-api-key).

### 2. Connect via SSH

Use your Box API key as the password when prompted.

```bash
ssh <box-id>@us-east-1.box.upstash.com
```

### 3. Install Hermes

Run the install script inside the SSH session.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Follow the CLI prompts to complete setup.

## Next steps

- [Configure an LLM provider](/docs/getting-started/installation)
- [Connect a Telegram bot](/docs/guides/team-telegram-assistant)
- [Automate with cron](/docs/guides/automate-with-cron)
