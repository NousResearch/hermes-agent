---
sidebar_position: 1
title: "Quickstart"
description: "Your first conversation with Hermes Agent — from install to chatting in 2 minutes"
---

# Quickstart

This guide walks you through installing Hermes Agent, setting up a provider, and having your first conversation. By the end, you'll know the key features and how to explore further.

## 1. Install Hermes Agent

Run the one-line installer:

```bash
# Linux / macOS / WSL2
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Windows Users
Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) first, then run the command above inside your WSL2 terminal.
:::

After it finishes, reload your shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

## 2. Set Up a Provider

The installer runs the setup wizard automatically. If you skipped it, run:

```bash
hermes setup
```

This walks you through selecting an inference provider:

| Provider | What it is | How to set up |
|----------|-----------|---------------|
| **Nous Portal** | Subscription-based, zero-config | OAuth login via `hermes model` |
| **OpenAI Codex** | ChatGPT OAuth, uses Codex models | Device code auth via `hermes model` |
| **OpenRouter** | 200+ models, pay-per-use | Enter your API key |
| **Custom Endpoint** | VLLM, SGLang, any OpenAI-compatible API | Set base URL + API key |

:::tip
You can switch providers at any time with `hermes model` — no code changes, no lock-in.
:::

## 3. Start Chatting

```bash
hermes
```

That's it! You'll see a welcome banner with your model, available tools, and skills. Type a message and press Enter.

```
❯ What can you help me with?
```

The agent has access to tools for web search, file operations, terminal commands, and more — all out of the box.

## 4. Try Key Features

### Ask it to use the terminal

```
❯ What's my disk usage? Show the top 5 largest directories.
```

The agent will run terminal commands on your behalf and show you the results.

### Use slash commands

Type `/` to see an autocomplete dropdown of all commands:

| Command | What it does |
|---------|-------------|
| `/help` | Show all available commands |
| `/tools` | List available tools |
| `/model` | Switch models interactively |
| `/personality pirate` | Try a fun personality |
| `/save` | Save the conversation |

### Multi-line input

Press `Alt+Enter` or `Ctrl+J` to add a new line. Great for pasting code or writing detailed prompts.

### Interrupt the agent

If the agent is taking too long, just type a new message and press Enter — it interrupts the current task and switches to your new instructions. `Ctrl+C` also works.

### Resume a session

When you exit, hermes prints a resume command:

```bash
hermes --continue    # Resume the most recent session
hermes -c            # Short form
```

## 5. Explore Further

Here are some things to try next:

### Set up a sandboxed terminal

For safety, run the agent in a Docker container or on a remote server:

```bash
hermes config set terminal.backend docker    # Docker isolation
hermes config set terminal.backend ssh       # Remote server
```

### Connect messaging platforms

Chat with Hermes from your phone via Telegram, Discord, Slack, or WhatsApp:

```bash
hermes gateway setup    # Interactive platform configuration
```

### Schedule automated tasks

```
❯ Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

The agent will set up a cron job that runs automatically via the gateway.

### Browse and install skills

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

Or use the `/skills` slash command inside chat.

### Try MCP servers

Connect to external tools via the Model Context Protocol:

```yaml
# Add to ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `hermes` | Start chatting |
| `hermes setup` | Configure providers and settings |
| `hermes model` | Switch provider or model |
| `hermes tools` | Configure which tools are enabled per platform |
| `hermes doctor` | Diagnose issues |
| `hermes update` | Update to latest version |
| `hermes gateway` | Start the messaging gateway |
| `hermes --continue` | Resume last session |

## Next Steps

- **[CLI Guide](../user-guide/cli.md)** — Master the terminal interface
- **[Configuration](../user-guide/configuration.md)** — Customize your setup
- **[Messaging Gateway](../user-guide/messaging/index.md)** — Connect Telegram, Discord, Slack, WhatsApp
- **[Tools & Toolsets](../user-guide/features/tools.md)** — Explore available capabilities
