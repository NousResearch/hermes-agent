---
title: "Integrations"
sidebar_label: "Overview"
sidebar_position: 0
---

# Integrations

Hermes Agent connects to external systems for AI inference, tool servers, IDE workflows, programmatic access, and more. These integrations extend what Hermes can do and where it can run.

## AI Providers & Routing

Hermes supports multiple AI inference providers out of the box. Use `hermes model` to configure interactively, or set them in `config.yaml`.

- **[AI Providers](/docs/user-guide/features/provider-routing)** — OpenRouter, Anthropic, OpenAI, Google, and any OpenAI-compatible endpoint. Hermes auto-detects capabilities like vision, streaming, and tool use per provider.
- **[Provider Routing](/docs/user-guide/features/provider-routing)** — Fine-grained control over which underlying providers handle your OpenRouter requests. Optimize for cost, speed, or quality with sorting, whitelists, blacklists, and explicit priority ordering.
- **[Fallback Providers](/docs/user-guide/features/fallback-providers)** — Automatic failover to backup LLM providers when your primary model encounters errors. Includes primary model fallback and independent auxiliary task fallback for vision, compression, and web extraction.

## Tool Servers (MCP)

- **[MCP Servers](/docs/user-guide/features/mcp)** — Connect Hermes to external tool servers via Model Context Protocol. Access tools from GitHub, databases, file systems, browser stacks, internal APIs, and more without writing native Hermes tools. Supports both stdio and SSE transports, per-server tool filtering, and capability-aware resource/prompt registration.

## Web Search Backends

The `web_search` and `web_extract` tools support four backend providers, configured via `config.yaml` or `hermes tools`:

| Backend | Env Var | Search | Extract | Crawl |
|---------|---------|--------|---------|-------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

Quick setup example:

```yaml
web:
  backend: firecrawl    # firecrawl | parallel | tavily | exa
```

If `web.backend` is not set, the backend is auto-detected from whichever API key is available. Self-hosted Firecrawl is also supported via `FIRECRAWL_API_URL`.

## Browser Automation

Hermes includes full browser automation with multiple backend options for navigating websites, filling forms, and extracting information:

- **Browserbase** — Managed cloud browsers with anti-bot tooling, CAPTCHA solving, and residential proxies
- **Browser Use** — Alternative cloud browser provider
- **Local Chrome via CDP** — Connect to your running Chrome instance using `/browser connect`
- **Local Chromium** — Headless local browser via the `agent-browser` CLI

See [Browser Automation](/docs/user-guide/features/browser) for setup and usage.

## Voice & TTS Providers

Text-to-speech and speech-to-text across all messaging platforms:

| Provider | Quality | Cost | API Key |
||----------|---------|------|---------|
|| **Edge TTS** (default) | Good | Free | None needed |
|| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
|| **OpenAI TTS** | Good | Paid | `VOICE_TOOLS_OPENAI_KEY` |
|| **MiniMax** | Good | Paid | `MINIMAX_API_KEY` |
|| **NeuTTS** | Good | Free | None needed |

Speech-to-text supports three providers: local Whisper (free, runs on-device), Groq (fast cloud), and OpenAI Whisper API. Voice message transcription works across Telegram, Discord, WhatsApp, and other messaging platforms. See [Voice & TTS](/docs/user-guide/features/tts) and [Voice Mode](/docs/user-guide/features/voice-mode) for details.

## IDE & Editor Integration

- **[IDE Integration (ACP)](/docs/user-guide/features/acp)** — Use Hermes Agent inside ACP-compatible editors such as VS Code, Zed, and JetBrains. Hermes runs as an ACP server, rendering chat messages, tool activity, file diffs, and terminal commands inside your editor.

## Programmatic Access

- **[API Server](/docs/user-guide/features/api-server)** — Expose Hermes as an OpenAI-compatible HTTP endpoint. Any frontend that speaks the OpenAI format — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox — can connect and use Hermes as a backend with its full toolset.

## Memory & Personalization

- **[Built-in Memory](/docs/user-guide/features/memory)** — Persistent, curated memory via `MEMORY.md` and `USER.md` files. The agent maintains bounded stores of personal notes and user profile data that survive across sessions.
- **[Memory Providers](/docs/user-guide/features/memory-providers)** — Plug in external memory backends for deeper personalization. Seven providers are supported: Honcho (dialectic reasoning), OpenViking (tiered retrieval), Mem0 (cloud extraction), Hindsight (knowledge graphs), Holographic (local SQLite), RetainDB (hybrid search), and ByteRover (CLI-based).

## Messaging Platforms

Hermes runs as a gateway bot on 15+ messaging platforms, all configured through the same `gateway` subsystem:

- **[Telegram](/docs/user-guide/messaging/telegram)**, **[Discord](/docs/user-guide/messaging/discord)**, **[Slack](/docs/user-guide/messaging/slack)**, **[WhatsApp](/docs/user-guide/messaging/whatsapp)**, **[Signal](/docs/user-guide/messaging/signal)**, **[Matrix](/docs/user-guide/messaging/matrix)**, **[Mattermost](/docs/user-guide/messaging/mattermost)**, **[Email](/docs/user-guide/messaging/email)**, **[SMS](/docs/user-guide/messaging/sms)**, **[DingTalk](/docs/user-guide/messaging/dingtalk)**, **[Feishu/Lark](/docs/user-guide/messaging/feishu)**, **[WeCom](/docs/user-guide/messaging/wecom)**, **[WeCom Callback](/docs/user-guide/messaging/wecom-callback)**, **[Weixin](/docs/user-guide/messaging/weixin)**, **[BlueBubbles](/docs/user-guide/messaging/bluebubbles)**, **[Home Assistant](/docs/user-guide/messaging/homeassistant)**, **[Webhooks](/docs/user-guide/messaging/webhooks)**

See the [Messaging Gateway overview](/docs/user-guide/messaging) for the platform comparison table and setup guide.

## Home Automation

- **[Home Assistant](/docs/user-guide/messaging/homeassistant)** — Control smart home devices via four dedicated tools (`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`). The Home Assistant toolset activates automatically when `HASS_TOKEN` is configured.

## Plugins

- **[Plugin System](/docs/user-guide/features/plugins)** — Extend Hermes with custom tools, lifecycle hooks, and CLI commands without modifying core code. Plugins are discovered from `~/.hermes/plugins/`, project-local `.hermes/plugins/`, and pip-installed entry points.
- **[Build a Plugin](/docs/guides/build-a-hermes-plugin)** — Step-by-step guide for creating Hermes plugins with tools, hooks, and CLI commands.

## Training & Evaluation

- **[RL Training](/docs/user-guide/features/rl-training)** — Generate trajectory data from agent sessions for reinforcement learning and model fine-tuning. Supports Atropos environments with customizable reward functions.
- **[Batch Processing](/docs/user-guide/features/batch-processing)** — Run the agent across hundreds of prompts in parallel, generating structured ShareGPT-format trajectory data for training data generation or evaluation.
