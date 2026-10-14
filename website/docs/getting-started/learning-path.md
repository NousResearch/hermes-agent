---
sidebar_position: 3
title: 'Learning Path'
description: 'Choose your learning path through the Hermes Agent documentation based on your experience level and goals.'
---

# Learning Path

Hermes Agent can do a lot — CLI assistant, Telegram/Discord bot, task automation, RL training, and more. This page helps you figure out where to start and what to read based on your experience level and what you're trying to accomplish.

:::tip Start Here
If you haven't installed Hermes Agent yet, begin with the [Installation guide](./installation.md) and then run through the [Quickstart](./quickstart.md). Everything below assumes you have a working installation.
:::

:::tip First-time provider setup
First-time users almost always want `hermes setup --portal` — one OAuth covers a model plus the four Tool Gateway tools (search/image/TTS/browser). See [Nous Portal](../integrations/nous-portal.md).
:::

## How to Use This Page

- **Know your level?** Jump to the [experience-level table](#by-experience-level) and follow the reading order for your tier.
- **Have a specific goal?** Skip to [By Use Case](#by-use-case) and find the scenario that matches.
- **Just browsing?** Check the [Key Features](#key-features-at-a-glance) table for a quick overview of everything Hermes Agent can do.

## By Experience Level

| Level | Goal | Recommended Reading | Time Estimate |
|---|---|---|---|
| **Beginner** | Get up and running, have basic conversations, use built-in tools | [Installation](./installation.md) → [Quickstart](./quickstart.md) → [CLI Usage](../user-guide/cli.md) → [Configuration](../user-guide/configuration.md) | ~1 hour |
| **Intermediate** | Set up messaging bots, use advanced features like memory, cron jobs, and skills | [Sessions](../user-guide/sessions.md) → [Messaging](../user-guide/messaging/index.md) → [Tools](../user-guide/features/tools.md) → [Skills](../user-guide/features/skills.md) → [Memory](../user-guide/features/memory.md) → [Cron](../user-guide/features/cron.md) | ~2–3 hours |
| **Advanced** | Build custom tools, create skills, train models with RL, contribute to the project | [Architecture](../developer-guide/architecture.md) → [Adding Tools](../developer-guide/adding-tools.md) → [Creating Skills](../developer-guide/creating-skills.md) → [Contributing](../developer-guide/contributing.md) | ~4–6 hours |

## By Use Case

Pick the scenario that matches what you want to do. Each one links you to the relevant docs in the order you should read them.

### "I want a CLI coding assistant"

Use Hermes Agent as an interactive terminal assistant for writing, reviewing, and running code.

1. [Installation](./installation.md)
2. [Quickstart](./quickstart.md)
3. [CLI Usage](../user-guide/cli.md)
4. [Code Execution](../user-guide/features/code-execution.md)
5. [Context Files](../user-guide/features/context-files.md)
6. [Tips & Tricks](../guides/tips.md)

:::tip
Pass files directly into your conversation with context files. Hermes Agent can read, edit, and run code in your projects.
:::

### "I want a Telegram/Discord bot"

Deploy Hermes Agent as a bot on your favorite messaging platform.

1. [Installation](./installation.md)
2. [Configuration](../user-guide/configuration.md)
3. [Messaging Overview](../user-guide/messaging/index.md)
4. [Telegram Setup](../user-guide/messaging/telegram.md)
5. [Discord Setup](../user-guide/messaging/discord.md)
6. [Voice Mode](../user-guide/features/voice-mode.md)
7. [Use Voice Mode with Hermes](../guides/use-voice-mode-with-hermes.md)
8. [Security](../user-guide/security.md)

For full project examples, see:
- [Daily Briefing Bot](../guides/daily-briefing-bot.md)
- [Team Telegram Assistant](../guides/team-telegram-assistant.md)

### "I want to automate tasks"

Schedule recurring tasks, run batch jobs, or chain agent actions together.

1. [Quickstart](./quickstart.md)
2. [Cron Scheduling](../user-guide/features/cron.md)
3. [Batch Processing](../user-guide/features/batch-processing.md)
4. [Delegation](../user-guide/features/delegation.md)
5. [Hooks](../user-guide/features/hooks.md)

:::tip
Cron jobs let Hermes Agent run tasks on a schedule — daily summaries, periodic checks, automated reports — without you being present.
:::

### "I want a team of specialist Bots"

Create named Bots with their own model, memory, skills, routines, and chats, then bring them together in group chats or through `@mentions`.

1. [Desktop](../user-guide/desktop.md)
2. [Profiles](../user-guide/profiles.md)
3. [Bot Mode](../user-guide/bot-mode.md)
4. [Cron Scheduling](../user-guide/features/cron.md)
5. [Multi-connection Desktop](../user-guide/multi-connection-desktop.md)

### "I want to build custom tools/skills"

Extend Hermes Agent with your own tools and reusable skill packages.

1. [Plugins](../user-guide/features/plugins.md)
2. [Build a Hermes Plugin](../developer-guide/plugins/index.md)
3. [Tools Overview](../user-guide/features/tools.md)
4. [Skills Overview](../user-guide/features/skills.md)
5. [MCP (Model Context Protocol)](../user-guide/features/mcp.md)
6. [Architecture](../developer-guide/architecture.md)
7. [Adding Tools](../developer-guide/adding-tools.md)
8. [Creating Skills](../developer-guide/creating-skills.md)

:::tip
For most custom tool creation, start with plugins. The [Adding Tools](../developer-guide/adding-tools.md)
page is for built-in Hermes core development, not the usual user/custom-tool path.
:::

### "I want to train models"

Use reinforcement learning to fine-tune model behavior with Hermes Agent's RL training pipeline (powered by [Atropos](https://github.com/NousResearch/atropos)).

1. [Quickstart](./quickstart.md)
2. [Configuration](../user-guide/configuration.md)
3. [Atropos RL Environments](https://github.com/NousResearch/atropos) (external)
4. [Provider Routing](../user-guide/features/provider-routing.md)
5. [Architecture](../developer-guide/architecture.md)

:::tip
RL training works best when you already understand the basics of how Hermes Agent handles conversations and tool calls. Run through the Beginner path first if you're new.
:::

### "I want to use it as a Python library"

Integrate Hermes Agent into your own Python applications programmatically.

1. [Installation](./installation.md)
2. [Quickstart](./quickstart.md)
3. [Python Library Guide](../guides/python-library.md)
4. [Architecture](../developer-guide/architecture.md)
5. [Tools](../user-guide/features/tools.md)
6. [Sessions](../user-guide/sessions.md)

## Key Features at a Glance

Not sure what's available? Here's a quick directory of major features:

| Feature | What It Does | Link |
|---|---|---|
| **Tools** | Built-in tools the agent can call (file I/O, search, shell, etc.) | [Tools](../user-guide/features/tools.md) |
| **Skills** | Installable plugin packages that add new capabilities | [Skills](../user-guide/features/skills.md) |
| **Memory** | Persistent memory across sessions | [Memory](../user-guide/features/memory.md) |
| **Bot Mode** | Named specialist Bots with persistent chats, routines, group chats, and `@mentions` | [Bot Mode](../user-guide/bot-mode.md) |
| **Context Files** | Feed files and directories into conversations | [Context Files](../user-guide/features/context-files.md) |
| **MCP** | Connect to external tool servers via Model Context Protocol | [MCP](../user-guide/features/mcp.md) |
| **Cron** | Schedule recurring agent tasks | [Cron](../user-guide/features/cron.md) |
| **Delegation** | Spawn sub-agents for parallel work | [Delegation](../user-guide/features/delegation.md) |
| **Code Execution** | Run Python scripts that call Hermes tools programmatically | [Code Execution](../user-guide/features/code-execution.md) |
| **Browser** | Web browsing and scraping | [Browser](../user-guide/features/browser.md) |
| **Hooks** | Event-driven callbacks and middleware | [Hooks](../user-guide/features/hooks.md) |
| **Batch Processing** | Process multiple inputs in bulk | [Batch Processing](../user-guide/features/batch-processing.md) |
| **Provider Routing** | Route requests across multiple LLM providers | [Provider Routing](../user-guide/features/provider-routing.md) |

## What to Read Next

Based on where you are right now:

- **Just finished installing?** → Head to the [Quickstart](./quickstart.md) to run your first conversation.
- **Completed the Quickstart?** → Read [CLI Usage](../user-guide/cli.md) and [Configuration](../user-guide/configuration.md) to customize your setup.
- **Comfortable with the basics?** → Explore [Tools](../user-guide/features/tools.md), [Skills](../user-guide/features/skills.md), and [Memory](../user-guide/features/memory.md) to unlock the full power of the agent.
- **Setting up for a team?** → Read [Security](../user-guide/security.md) and [Sessions](../user-guide/sessions.md) to understand access control and conversation management.
- **Ready to build?** → Jump into the [Developer Guide](../developer-guide/architecture.md) to understand the internals and start contributing.
- **Want practical examples?** → Check out the [Guides](../guides/tips.md) section for real-world projects and tips.

:::tip
You don't need to read everything. Pick the path that matches your goal, follow the links in order, and you'll be productive quickly. You can always come back to this page to find your next step.
:::
