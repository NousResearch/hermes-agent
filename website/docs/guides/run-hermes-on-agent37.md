---
title: "Run Hermes Agent on Agent37"
description: "Launch a managed, always-on Hermes instance in the browser: pick a plan, launch, add a model key, and start chatting"
---

# Run Hermes Agent on Agent37

This guide walks you through running an always-on Hermes Agent on [Agent37](https://www.agent37.com) managed hosting. Agent37 provisions and operates the instance for you: Hermes runs in an isolated container with a browser dashboard, so there is no server to install, patch, or keep awake. If you would rather run Hermes on your own machine or VPS, see the [Quickstart](/getting-started/quickstart).

## Prerequisites

- Agent37 account ([signup](https://www.agent37.com/))
- A model API key (bring your own key from your model provider; some plans bundle models)
- About 5 minutes

You do **not** need: a VPS, Docker, Python, or a terminal on your own machine. The instance runs Hermes for you.

## 1. Choose a plan

Open [agent37.com](https://www.agent37.com/), pick a managed hosting plan, and complete checkout. See the Agent37 site for current plans.

## 2. Launch the agent

In the [Agent37 dashboard](https://www.agent37.com/dashboard), click **Launch Agent** and select **Hermes** as the agent type. Provisioning takes about a minute; the instance appears in your dashboard when it is ready.

## 3. Add your model key

Provide your model API key during setup, or use bundled models if your plan includes them. Keys are used by your instance directly against the model provider.

## 4. Say hello

Open **Web Chat** from the instance view and send "Hi". Hermes replies and you can start working with it immediately.

## Manage the instance

From the instance view in the dashboard:

- **Web Chat** — the hosted chat UI for your agent.
- **Terminal** — full TTY shell access to the instance for inspecting logs and files or making manual changes.
- **Files** — visual file browser for the instance workspace.
- Scheduled jobs and a live Linux desktop are also available from the dashboard.

Runtime updates and security patches are applied by Agent37 automatically. Because the instance is always on, Hermes keeps its memory, files, and sessions between conversations, and scheduled work runs while you are away.

## Connect channels

A hosted Hermes instance is most useful when you can reach it from your phone. Connect Telegram or another messaging channel from Hermes the same way as on any other install; see [Messaging](/user-guide/messaging).

## Troubleshooting

**Instance stuck provisioning** — provisioning normally completes in about a minute; if it takes much longer, refresh the dashboard, then contact Agent37 support from the dashboard.

**Hermes not replying in Web Chat** — open **Terminal** and check the logs, and confirm your model API key is valid for the configured provider.

## Related

- [Quickstart](/getting-started/quickstart) — run Hermes on your own machine
- [Messaging](/user-guide/messaging) — connect Telegram and other channels
- [Configuring models](/user-guide/configuring-models) — providers and model selection
