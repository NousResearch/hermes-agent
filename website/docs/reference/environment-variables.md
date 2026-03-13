---
sidebar_position: 2
title: "Environment Variables"
description: "Complete reference of all environment variables used by Hermes Agent"
---

# Environment Variables Reference

All variables go in `~/.hermes/.env`. You can also set them with `hermes config set VAR value`.

## LLM Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (recommended for flexibility) |
| `OPENAI_API_KEY` | API key for custom OpenAI-compatible endpoints (used with `OPENAI_BASE_URL`) |
| `OPENAI_BASE_URL` | Base URL for custom endpoint (VLLM, SGLang, etc.) |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API key ([z.ai](https://z.ai)) |
| `GLM_BASE_URL` | Override z.ai base URL (default: `https://api.z.ai/api/paas/v4`) |
| `KIMI_API_KEY` | Kimi / Moonshot AI API key ([moonshot.ai](https://platform.moonshot.ai)) |
| `KIMI_BASE_URL` | Override Kimi base URL (default: `https://api.moonshot.ai/v1`) |
| `MINIMAX_API_KEY` | MiniMax API key — global endpoint ([minimax.io](https://www.minimax.io)) |
| `MINIMAX_BASE_URL` | Override MiniMax base URL (default: `https://api.minimax.io/v1`) |
| `MINIMAX_CN_API_KEY` | MiniMax API key — China endpoint ([minimaxi.com](https://www.minimaxi.com)) |
| `MINIMAX_CN_BASE_URL` | Override MiniMax China base URL (default: `https://api.minimaxi.com/v1`) |
| `ANTHROPIC_API_KEY` | Anthropic API key or setup-token ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_TOKEN` | Anthropic OAuth/setup token (alternative to `ANTHROPIC_API_KEY`) |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code setup-token (same as `ANTHROPIC_TOKEN`) |
| `HERMES_MODEL` | Preferred model name (checked before `LLM_MODEL`, used by gateway) |
| `LLM_MODEL` | Default model name (fallback when not set in config.yaml) |
| `VOICE_TOOLS_OPENAI_KEY` | OpenAI key for TTS and voice transcription (separate from custom endpoint) |
| `HERMES_HOME` | Override Hermes config directory (default: `~/.hermes`) |

## Provider Auth (OAuth)

| Variable | Description |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | Override provider selection: `auto`, `openrouter`, `nous`, `anthropic`, `zai`, `kimi-coding`, `minimax`, `minimax-cn` (default: `auto`) |
| `HERMES_PORTAL_BASE_URL` | Override Nous Portal URL (for development/testing) |
| `NOUS_INFERENCE_BASE_URL` | Override Nous inference API URL |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | Min agent key TTL before re-mint (default: 1800 = 30min) |
| `HERMES_DUMP_REQUESTS` | Dump API request payloads to log files (`true`/`false`) |

## Tool APIs

| Variable | Description |
|----------|-------------|
| `FIRECRAWL_API_KEY` | Web scraping ([firecrawl.dev](https://firecrawl.dev/)) |
| `FIRECRAWL_API_URL` | Custom Firecrawl API endpoint for self-hosted instances (optional) |
| `BROWSERBASE_API_KEY` | Browser automation ([browserbase.com](https://browserbase.com/)) |
| `BROWSERBASE_PROJECT_ID` | Browserbase project ID |
| `BROWSER_INACTIVITY_TIMEOUT` | Browser session inactivity timeout in seconds |
| `FAL_KEY` | Image generation ([fal.ai](https://fal.ai/)) |
| `ELEVENLABS_API_KEY` | Premium TTS voices ([elevenlabs.io](https://elevenlabs.io/)) |
| `HONCHO_API_KEY` | Cross-session user modeling ([honcho.dev](https://honcho.dev/)) |
| `TINKER_API_KEY` | RL training ([tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/)) |
| `WANDB_API_KEY` | RL training metrics ([wandb.ai](https://wandb.ai/)) |
| `DAYTONA_API_KEY` | Daytona cloud sandboxes ([daytona.io](https://daytona.io/)) |

## Terminal Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona` |
| `TERMINAL_DOCKER_IMAGE` | Docker image (default: `python:3.11`) |
| `TERMINAL_DOCKER_VOLUMES` | Additional Docker volume mounts (comma-separated `host:container` pairs) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity image or `.sif` path |
| `TERMINAL_MODAL_IMAGE` | Modal container image |
| `TERMINAL_DAYTONA_IMAGE` | Daytona sandbox image |
| `TERMINAL_TIMEOUT` | Command timeout in seconds |
| `TERMINAL_LIFETIME_SECONDS` | Max lifetime for terminal sessions in seconds |
| `TERMINAL_CWD` | Working directory for all terminal sessions |
| `SUDO_PASSWORD` | Enable sudo without interactive prompt |

## SSH Backend

| Variable | Description |
|----------|-------------|
| `TERMINAL_SSH_HOST` | Remote server hostname |
| `TERMINAL_SSH_USER` | SSH username |
| `TERMINAL_SSH_PORT` | SSH port (default: 22) |
| `TERMINAL_SSH_KEY` | Path to private key |

## Container Resources (Docker, Singularity, Modal, Daytona)

| Variable | Description |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPU cores (default: 1) |
| `TERMINAL_CONTAINER_MEMORY` | Memory in MB (default: 5120) |
| `TERMINAL_CONTAINER_DISK` | Disk in MB (default: 51200) |
| `TERMINAL_CONTAINER_PERSISTENT` | Persist container filesystem across sessions (default: `true`) |
| `TERMINAL_SANDBOX_DIR` | Host directory for workspaces and overlays (default: `~/.hermes/sandboxes/`) |

## Messaging

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (from @BotFather) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs allowed to use bot |
| `TELEGRAM_HOME_CHANNEL` | Default channel for cron delivery |
| `TELEGRAM_HOME_CHANNEL_NAME` | Display name for home channel |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | Comma-separated user IDs allowed to use bot |
| `DISCORD_HOME_CHANNEL` | Default channel for cron delivery |
| `DISCORD_HOME_CHANNEL_NAME` | Display name for home channel |
| `SLACK_BOT_TOKEN` | Slack bot token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Slack app-level token (`xapp-...`, required for Socket Mode) |
| `SLACK_ALLOWED_USERS` | Comma-separated Slack user IDs |
| `SLACK_HOME_CHANNEL` | Default Slack channel for cron delivery |
| `WHATSAPP_ENABLED` | Enable WhatsApp bridge (`true`/`false`) |
| `WHATSAPP_MODE` | `bot` (separate number) or `self-chat` (message yourself) |
| `WHATSAPP_ALLOWED_USERS` | Comma-separated phone numbers (with country code) |
| `SIGNAL_HTTP_URL` | signal-cli daemon HTTP endpoint (e.g., `http://127.0.0.1:8080`) |
| `SIGNAL_ACCOUNT` | Bot phone number in E.164 format (e.g., `+15551234567`) |
| `SIGNAL_ALLOWED_USERS` | Comma-separated E.164 phone numbers or UUIDs |
| `SIGNAL_GROUP_ALLOWED_USERS` | Comma-separated group IDs, or `*` for all groups (omit to disable groups) |
| `MESSAGING_CWD` | Working directory for terminal in messaging (default: `~`) |
| `GATEWAY_ALLOWED_USERS` | Comma-separated user IDs allowed across all platforms |
| `GATEWAY_ALLOW_ALL_USERS` | Allow all users without allowlist (`true`/`false`, default: `false`) |

## Agent Behavior

| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | Max tool-calling iterations per conversation (default: 60) |
| `HERMES_TOOL_PROGRESS` | Send progress messages when using tools (`true`/`false`) |
| `HERMES_TOOL_PROGRESS_MODE` | `all` (every call, default) or `new` (only when tool changes) |
| `HERMES_HUMAN_DELAY_MODE` | Response pacing: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | Custom delay range minimum (ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | Custom delay range maximum (ms) |
| `HERMES_QUIET` | Suppress non-essential output (`true`/`false`) |
| `HERMES_EXEC_ASK` | Enable execution approval prompts in gateway mode (`true`/`false`) |

## Session Settings

| Variable | Description |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | Reset sessions after N minutes of inactivity (default: 120) |
| `SESSION_RESET_HOUR` | Daily reset hour in 24h format (default: 4 = 4am) |

## Context Compression

| Variable | Description |
|----------|-------------|
| `CONTEXT_COMPRESSION_ENABLED` | Enable auto-compression (default: `true`) |
| `CONTEXT_COMPRESSION_THRESHOLD` | Trigger at this % of limit (default: 0.85) |
| `CONTEXT_COMPRESSION_MODEL` | Model for summaries |

## Provider Routing (config.yaml only)

These go in `~/.hermes/config.yaml` under the `provider_routing` section:

| Key | Description |
|-----|-------------|
| `sort` | Sort providers: `"price"` (default), `"throughput"`, or `"latency"` |
| `only` | List of provider slugs to allow (e.g., `["anthropic", "google"]`) |
| `ignore` | List of provider slugs to skip |
| `order` | List of provider slugs to try in order |
| `require_parameters` | Only use providers supporting all request params (`true`/`false`) |
| `data_collection` | `"allow"` (default) or `"deny"` to exclude data-storing providers |

:::tip
Use `hermes config set` to set environment variables — it automatically saves them to the right file (`.env` for secrets, `config.yaml` for everything else).
:::
