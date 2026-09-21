---
title: "Integrations"
sidebar_label: "Overview"
sidebar_position: 0
---

# Integrations

Hermes Agent connects to external systems for AI inference, tool servers, IDE workflows, programmatic access, and more. These integrations extend what Hermes can do and where it can run.

:::tip Start here
If you only have time to set up one integration, set up [Nous Portal](./nous-portal.md) — a single OAuth login covers 300+ models plus the four Tool Gateway tools (web search, image generation, TTS, and browser automation).
:::

## AI Providers & Routing

Hermes supports multiple AI inference providers out of the box. Use `hermes model` to configure interactively, or set them in `config.yaml`.

- **[AI Providers](./providers.md)** — OpenRouter, Anthropic, OpenAI, Google, and any OpenAI-compatible endpoint. Hermes auto-detects capabilities like vision, streaming, and tool use per provider.
- **[Provider Routing](../user-guide/features/provider-routing.md)** — Fine-grained control over which underlying providers handle your OpenRouter requests. Optimize for cost, speed, or quality with sorting, whitelists, blacklists, and explicit priority ordering.
- **[Fallback Providers](../user-guide/features/fallback-providers.md)** — Automatic failover to backup LLM providers when your primary model encounters errors. Includes primary model fallback and independent auxiliary task fallback for vision, compression, and web extraction.

## Tool Servers (MCP)

- **[MCP Servers](../user-guide/features/mcp.md)** — Connect Hermes to external tool servers via Model Context Protocol. Access tools from GitHub, databases, file systems, browser stacks, internal APIs, and more without writing native Hermes tools. Supports both stdio and SSE transports, per-server tool filtering, and capability-aware resource/prompt registration.

## Web Search Backends

The `web_search` and `web_extract` tools support eight backend providers, configured via `config.yaml` or `hermes tools`:

| Backend | Env Var | Search | Extract | Crawl |
|---------|---------|--------|---------|-------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Brave** (free tier) | `BRAVE_SEARCH_API_KEY` | ✔ | — | — |
| **DuckDuckGo** (ddgs) | _(none)_ | ✔ | — | — |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **xAI** | `XAI_API_KEY` | ✔ | — | — |

Quick setup example:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | keenable | exa | parallel | xai
```

If `web.backend` is not set, the backend is auto-detected from whichever API key is available. Self-hosted Firecrawl is also supported via `FIRECRAWL_API_URL`.

## Browser Automation

Hermes includes full browser automation with multiple backend options for navigating websites, filling forms, and extracting information:

- **Browser Use Cloud** — Managed Chromium with stealth, residential proxies, CAPTCHA solving, and reusable browser profiles
- **Browserbase** — Alternative cloud browser provider with managed browsers, anti-bot tooling, CAPTCHA solving, and residential proxies
- **Local Chromium-family CDP** — Connect to your running Chrome, Brave, Chromium, or Edge browser using `/browser connect`
- **Local Chromium** — Headless local browser via the `agent-browser` CLI

See [Browser Automation](../user-guide/features/browser.md) for setup and usage.

## Voice & TTS Providers

Text-to-speech and speech-to-text across all messaging platforms:

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------|
| **Edge TTS** (default) | Good | Free | None needed |
| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Good | Paid | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | Good | Paid | `MINIMAX_API_KEY` |
| **xAI TTS** | Good | Paid | `XAI_API_KEY` |
| **NeuTTS** | Good | Free | None needed |

Speech-to-text supports eight providers: local faster-whisper (free, runs on-device), a local command wrapper, Groq, OpenAI Whisper API, Mistral, xAI, ElevenLabs Scribe, and DeepInfra. Voice message transcription works across Telegram, Discord, WhatsApp, and other messaging platforms. See [Voice & TTS](../user-guide/features/tts.md) and [Voice Mode](../user-guide/features/voice-mode.md) for details.

## IDE & Editor Integration

- **[IDE Integration (ACP)](../user-guide/features/acp.md)** — Use Hermes Agent inside ACP-compatible editors such as VS Code, Zed, and JetBrains. Hermes runs as an ACP server, rendering chat messages, tool activity, file diffs, and terminal commands inside your editor.

## Programmatic Access

- **[API Server](../user-guide/features/api-server.md)** — Expose Hermes as an OpenAI-compatible HTTP endpoint. Any frontend that speaks the OpenAI format — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox — can connect and use Hermes as a backend with its full toolset.

## Memory & Personalization

- **[Built-in Memory](../user-guide/features/memory.md)** — Persistent, curated memory via `MEMORY.md` and `USER.md` files. The agent maintains bounded stores of personal notes and user profile data that survive across sessions.
- **[Memory Providers](../user-guide/features/memory-providers.md)** — Plug in external memory backends for deeper personalization. Eight providers are supported: Honcho (dialectic reasoning), OpenViking (tiered retrieval), Mem0 (cloud extraction), Hindsight (knowledge graphs), Holographic (local SQLite), RetainDB (hybrid search), ByteRover (CLI-based), and Supermemory.

## Messaging Platforms

Hermes runs as a gateway bot on 27+ messaging platforms, all configured through the same `gateway` subsystem:

- **[Telegram](../user-guide/messaging/telegram.md)**, **[Discord](../user-guide/messaging/discord.md)**, **[Slack](../user-guide/messaging/slack.md)**, **[WhatsApp](../user-guide/messaging/whatsapp.md)**, **[Signal](../user-guide/messaging/signal.md)**, **[Matrix](../user-guide/messaging/matrix.md)**, **[Mattermost](../user-guide/messaging/mattermost.md)**, **[Email](../user-guide/messaging/email.md)**, **[SMS](../user-guide/messaging/sms.md)**, **[DingTalk](../user-guide/messaging/dingtalk.md)**, **[Feishu/Lark](../user-guide/messaging/feishu.md)**, **[WeCom](../user-guide/messaging/wecom.md)**, **[WeCom Callback](../user-guide/messaging/wecom-callback.md)**, **[Weixin](../user-guide/messaging/weixin.md)**, **[BlueBubbles](../user-guide/messaging/bluebubbles.md)**, **[Buzz](../user-guide/messaging/buzz.md)**, **[QQ Bot](../user-guide/messaging/qqbot.md)**, **[Yuanbao](../user-guide/messaging/yuanbao.md)**, **[Home Assistant](../user-guide/messaging/homeassistant.md)**, **[Microsoft Teams](../user-guide/messaging/teams.md)**, **[Microsoft Teams Meetings](../user-guide/messaging/teams-meetings.md)**, **[Microsoft Graph Webhook](../user-guide/messaging/msgraph-webhook.md)**, **[Google Chat](../user-guide/messaging/google_chat.md)**, **[LINE](../user-guide/messaging/line.md)**, **[ntfy](../user-guide/messaging/ntfy.md)**, **[SimpleX](../user-guide/messaging/simplex.md)**, **[Open WebUI](../user-guide/messaging/open-webui.md)**, **[Webhooks](../user-guide/messaging/webhooks.md)**

See the [Messaging Gateway overview](../user-guide/messaging/index.md) for the platform comparison table and setup guide.

### Quick connect links

The big platforms have a canonical "create your bot/app" URL, and some accept parameters that pre-open the right form. Skip the console-hunting and go straight there:

| Platform | Direct link | What it opens |
|----------|-------------|---------------|
| **Telegram** | [t.me/BotFather](https://t.me/BotFather) | Chat with BotFather — send `/newbot` to mint a bot token |
| **Discord** | [discord.com/developers/applications?new_application=true](https://discord.com/developers/applications?new_application=true) | Developer Portal with the **New Application** dialog pre-opened |
| **Slack** | [api.slack.com/apps?new_app=1](https://api.slack.com/apps?new_app=1) | The **Create New App** dialog — pick *From an app manifest* and paste the manifest `hermes slack manifest --agent-view` generates |
| **LINE** | [developers.line.biz/console](https://developers.line.biz/console/) | LINE Developers Console for creating a Messaging API channel |
| **Feishu/Lark** | [open.feishu.cn/app](https://open.feishu.cn/app) | Feishu open-platform console for creating a custom app |

Each platform's setup page walks through what to do once you're there.

## Collaboration Workspaces

- **[Buzz](./buzz.md)** — Block's Nostr-based human+agent workspace. Three integration paths: Buzz Desktop spawns Hermes as a managed ACP runtime, the `buzz-acp` relay bridge hosts a Hermes identity server-side, or the native gateway platform joins Buzz channels with full Hermes memory/skills/approvals/cron. The overview page compares all three.

## Home Automation

- **[Home Assistant](../user-guide/messaging/homeassistant.md)** — Control smart home devices via four dedicated tools (`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`). The Home Assistant toolset activates automatically when `HASS_TOKEN` is configured.

## Plugins

- **[Plugin System](../user-guide/features/plugins.md)** — Extend Hermes with custom tools, lifecycle hooks, and CLI commands without modifying core code. Plugins are discovered from `~/.hermes/plugins/`, project-local `.hermes/plugins/`, and pip-installed entry points.
- **[Build a Plugin](../developer-guide/plugins/index.md)** — Step-by-step guide for creating Hermes plugins with tools, hooks, and CLI commands.

## Training & Evaluation

- **[Batch Processing](../user-guide/features/batch-processing.md)** — Run the agent across hundreds of prompts in parallel, generating structured ShareGPT-format trajectory data for training data generation or evaluation.
