---
sidebar_position: 3
title: 'Learning Path'
description: 'Choose your learning path through the Hermes Agent documentation based on your experience level and goals.'
---

# Learning Path

Hermes Agent can do a lot — CLI assistant, Telegram/Discord bot, task automation, RL training, and more. This page helps you figure out where to start and what to read based on your experience level and what you're trying to accomplish.

:::tip Start Here
If you haven't installed Hermes Agent yet, begin with the [Installation guide](/docs/getting-started/installation) and then run through the [Quickstart](/docs/getting-started/quickstart). Everything below assumes you have a working installation.
:::

## How to Use This Page

- **Know your level?** Jump to the [experience-level table](#by-experience-level) and follow the reading order for your tier.
- **Have a specific goal?** Skip to [By Use Case](#by-use-case) and find the scenario that matches.
- **Just browsing?** Check the [Key Features](#key-features-at-a-glance) table for a quick overview of everything Hermes Agent can do.

## By Experience Level

| Level | Goal | Recommended Reading | Time Estimate |
|---|---|---|---|
| **Beginner** | Get up and running, have basic conversations, use built-in tools | [Installation](/docs/getting-started/installation) → [Quickstart](/docs/getting-started/quickstart) → [CLI Usage](/docs/user-guide/cli) → [Configuration](/docs/user-guide/configuration) | ~1 hour |
| **Intermediate** | Set up messaging bots, use advanced features like memory, cron jobs, and skills | [Sessions](/docs/user-guide/sessions) → [Messaging](/docs/user-guide/messaging) → [Tools](/docs/user-guide/features/tools) → [Skills](/docs/user-guide/features/skills) → [Memory](/docs/user-guide/features/memory) → [Cron](/docs/user-guide/features/cron) | ~2–3 hours |
| **Advanced** | Build custom tools, create skills, train models with RL, contribute to the project | [Architecture](/docs/developer-guide/architecture) → [Adding Tools](/docs/developer-guide/adding-tools) → [Creating Skills](/docs/developer-guide/creating-skills) → [RL Training](/docs/user-guide/features/rl-training) → [Contributing](/docs/developer-guide/contributing) | ~4–6 hours |

## By Use Case

Pick the scenario that matches what you want to do. Each one links you to the relevant docs in the order you should read them.

### "I want a CLI coding assistant"

Use Hermes Agent as an interactive terminal assistant for writing, reviewing, and running code.

1. [Installation](/docs/getting-started/installation)
2. [Quickstart](/docs/getting-started/quickstart)
3. [CLI Usage](/docs/user-guide/cli)
4. [Code Execution](/docs/user-guide/features/code-execution)
5. [Context Files](/docs/user-guide/features/context-files)
6. [Tips & Tricks](/docs/guides/tips)

:::tip
Pass files directly into your conversation with context files. Hermes Agent can read, edit, and run code in your projects.
:::

### "I want a Telegram/Discord bot"

Deploy Hermes Agent as a bot on your favorite messaging platform.

1. [Installation](/docs/getting-started/installation)
2. [Configuration](/docs/user-guide/configuration)
3. [Messaging Overview](/docs/user-guide/messaging)
4. [Telegram Setup](/docs/user-guide/messaging/telegram)
5. [Discord Setup](/docs/user-guide/messaging/discord)
6. [Voice Mode](/docs/user-guide/features/voice-mode)
7. [Use Voice Mode with Hermes](/docs/guides/use-voice-mode-with-hermes)
8. [Security](/docs/user-guide/security)

For full project examples, see:
- [Daily Briefing Bot](/docs/guides/daily-briefing-bot)
- [Team Telegram Assistant](/docs/guides/team-telegram-assistant)

### "I want to automate tasks"

Schedule recurring tasks, run batch jobs, or chain agent actions together.

1. [Quickstart](/docs/getting-started/quickstart)
2. [Cron Scheduling](/docs/user-guide/features/cron)
3. [Batch Processing](/docs/user-guide/features/batch-processing)
4. [Delegation](/docs/user-guide/features/delegation)
5. [Hooks](/docs/user-guide/features/hooks)

:::tip
Cron jobs let Hermes Agent run tasks on a schedule — daily summaries, periodic checks, automated reports — without you being present.
:::

### "I want to build custom tools/skills"

Extend Hermes Agent with your own tools and reusable skill packages.

1. [Plugins](/docs/user-guide/features/plugins)
2. [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin)
3. [Tools Overview](/docs/user-guide/features/tools)
4. [Skills Overview](/docs/user-guide/features/skills)
5. [MCP (Model Context Protocol)](/docs/user-guide/features/mcp)
6. [Architecture](/docs/developer-guide/architecture)
7. [Adding Tools](/docs/developer-guide/adding-tools)
8. [Creating Skills](/docs/developer-guide/creating-skills)

:::tip
For most custom tool creation, start with plugins. The [Adding Tools](/docs/developer-guide/adding-tools)
page is for built-in Hermes core development, not the usual user/custom-tool path.
:::

### "I want to train models"

Use reinforcement learning to fine-tune model behavior with Hermes Agent's built-in RL training pipeline.

1. [Quickstart](/docs/getting-started/quickstart)
2. [Configuration](/docs/user-guide/configuration)
3. [RL Training](/docs/user-guide/features/rl-training)
4. [Provider Routing](/docs/user-guide/features/provider-routing)
5. [Architecture](/docs/developer-guide/architecture)

:::tip
RL training works best when you already understand the basics of how Hermes Agent handles conversations and tool calls. Run through the Beginner path first if you're new.
:::

### "I want to use it as a Python library"

Integrate Hermes Agent into your own Python applications programmatically.

1. [Installation](/docs/getting-started/installation)
2. [Quickstart](/docs/getting-started/quickstart)
3. [Python Library Guide](/docs/guides/python-library)
4. [Architecture](/docs/developer-guide/architecture)
5. [Tools](/docs/user-guide/features/tools)
6. [Sessions](/docs/user-guide/sessions)

## Key Features at a Glance

Not sure what's available? Here's a quick directory of major features:

| Feature | What It Does | Link |
|---|---|---|
| **Tools** | Built-in tools the agent can call (file I/O, search, shell, etc.) | [Tools](/docs/user-guide/features/tools) |
| **Skills** | Installable plugin packages that add new capabilities | [Skills](/docs/user-guide/features/skills) |
| **Memory** | Persistent memory across sessions | [Memory](/docs/user-guide/features/memory) |
| **Context Files** | Feed files and directories into conversations | [Context Files](/docs/user-guide/features/context-files) |
| **MCP** | Connect to external tool servers via Model Context Protocol | [MCP](/docs/user-guide/features/mcp) |
| **Cron** | Schedule recurring agent tasks | [Cron](/docs/user-guide/features/cron) |
| **Delegation** | Spawn sub-agents for parallel work | [Delegation](/docs/user-guide/features/delegation) |
| **Code Execution** | Run Python scripts that call Hermes tools programmatically | [Code Execution](/docs/user-guide/features/code-execution) |
| **Browser** | Web browsing and scraping | [Browser](/docs/user-guide/features/browser) |
| **Hooks** | Event-driven callbacks and middleware | [Hooks](/docs/user-guide/features/hooks) |
| **Batch Processing** | Process multiple inputs in bulk | [Batch Processing](/docs/user-guide/features/batch-processing) |
| **RL Training** | Fine-tune models with reinforcement learning | [RL Training](/docs/user-guide/features/rl-training) |
| **Provider Routing** | Route requests across multiple LLM providers | [Provider Routing](/docs/user-guide/features/provider-routing) |

## What to Read Next

Based on where you are right now:

- **Just finished installing?** → Head to the [Quickstart](/docs/getting-started/quickstart) to run your first conversation.
- **Completed the Quickstart?** → Read [CLI Usage](/docs/user-guide/cli) and [Configuration](/docs/user-guide/configuration) to customize your setup.
- **Comfortable with the basics?** → Explore [Tools](/docs/user-guide/features/tools), [Skills](/docs/user-guide/features/skills), and [Memory](/docs/user-guide/features/memory) to unlock the full power of the agent.
- **Setting up for a team?** → Read [Security](/docs/user-guide/security) and [Sessions](/docs/user-guide/sessions) to understand access control and conversation management.
- **Ready to build?** → Jump into the [Developer Guide](/docs/developer-guide/architecture) to understand the internals and start contributing.
- **Want practical examples?** → Check out the [Guides](/docs/guides/tips) section for real-world projects and tips.

:::tip
You don't need to read everything. Pick the path that matches your goal, follow the links in order, and you'll be productive quickly. You can always come back to this page to find your next step.
:::
