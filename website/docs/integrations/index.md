---
title: "Integrations"
sidebar_label: "Overview"
sidebar_position: 0
---

# Integrations

Hermes Agent connects to external systems for AI inference, tool servers, IDE workflows, programmatic access, and more. These integrations extend what Hermes can do and where it can run.

## Available Integrations

- **[AI Providers](/docs/user-guide/features/provider-routing)** — Set up and configure inference providers. Hermes works with OpenRouter, Anthropic, OpenAI, Google, and any OpenAI-compatible endpoint. Use `hermes model` to configure interactively.

- **[MCP Servers](/docs/user-guide/features/mcp)** — Connect Hermes to external tool servers via Model Context Protocol. Access tools from GitHub, databases, file systems, browser stacks, internal APIs, and more without writing native Hermes tools.

- **[IDE Integration (ACP)](/docs/user-guide/features/acp)** — Use Hermes Agent inside ACP-compatible editors such as VS Code, Zed, and JetBrains. Hermes runs as an ACP server, rendering chat messages, tool activity, file diffs, and terminal commands inside your editor.

- **[API Server](/docs/user-guide/features/api-server)** — Expose Hermes as an OpenAI-compatible HTTP endpoint. Any frontend that speaks the OpenAI format — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox — can connect and use Hermes as a backend with its full toolset.

- **[Honcho Memory](/docs/user-guide/features/honcho)** — AI-native persistent memory for cross-session user modeling and personalization. Honcho adds deep user modeling via dialectic reasoning on top of Hermes's built-in memory system.

- **[Provider Routing](/docs/user-guide/features/provider-routing)** — Fine-grained control over which underlying AI providers handle your OpenRouter requests. Optimize for cost, speed, or quality with sorting, whitelists, blacklists, and explicit priority ordering.

- **[Fallback Providers](/docs/user-guide/features/fallback-providers)** — Automatic failover to backup LLM providers when your primary model encounters errors. Includes primary model fallback and independent auxiliary task fallback for vision, compression, and web extraction.
