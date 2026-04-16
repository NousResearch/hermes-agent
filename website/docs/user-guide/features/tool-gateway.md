---
title: "Nous Tool Gateway"
description: "Route web search, image generation, text-to-speech, and browser automation through your Nous subscription — no extra API keys needed"
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway

:::tip Get Started
The Tool Gateway is included with paid Nous Portal subscriptions. **[Manage your subscription →](https://portal.nousresearch.com/manage-subscription)**
:::

The **Tool Gateway** lets paid [Nous Portal](https://portal.nousresearch.com) subscribers use web search, image generation, text-to-speech, and browser automation through their existing subscription — no need to sign up for separate API keys from Firecrawl, FAL, OpenAI, or Browser Use.

## What's Included

| Tool | What It Does | Direct Alternative |
|------|--------------|--------------------|
| **Web search & extract** | Search the web and extract page content via Firecrawl | `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `PARALLEL_API_KEY`, `TAVILY_API_KEY` |
| **Image generation** | Generate images via FAL (FLUX 2 Pro + upscaling) | `FAL_KEY` |
| **Text-to-speech** | Convert text to speech via OpenAI TTS | `VOICE_TOOLS_OPENAI_KEY`, `ELEVENLABS_API_KEY` |
| **Browser automation** | Control cloud browsers via Browser Use | `BROWSER_USE_API_KEY`, `BROWSERBASE_API_KEY` |

All four tools bill to your Nous subscription. You can enable any combination — for example, use the gateway for web and image generation while keeping your own ElevenLabs key for TTS.

## Eligibility

The Tool Gateway is available to **paid** [Nous Portal](https://portal.nousresearch.com/manage-subscription) subscribers. Free-tier accounts do not have access — [upgrade your subscription](https://portal.nousresearch.com/manage-subscription) to unlock it.

To check your status:

```bash
hermes status
```

Look for the **Nous Tool Gateway** section. It shows which tools are active via the gateway, which use direct keys, and which aren't configured.

## Enabling the Tool Gateway

### During model setup

When you run `hermes model` and select Nous Portal as your provider, Hermes automatically offers to enable the Tool Gateway:

```
Your Nous subscription includes the Tool Gateway.

  The Tool Gateway gives you access to web search, image generation,
  text-to-speech, and browser automation through your Nous subscription.
  No need to sign up for separate API keys — just pick the tools you want.

  ○ Web search & extract (Firecrawl) — not configured
  ○ Image generation (FAL) — not configured
  ○ Text-to-speech (OpenAI TTS) — not configured
  ○ Browser automation (Browser Use) — not configured

  ● Enable Tool Gateway
  ○ Skip
```

Select **Enable Tool Gateway** and you're done.

If you already have direct API keys for some tools, the prompt adapts — you can enable the gateway for all tools (your existing keys are kept in `.env` but not used at runtime), enable only for unconfigured tools, or skip entirely.

### Via `hermes tools`

You can also enable the gateway tool-by-tool through the interactive tool configuration:

```bash
hermes tools
```

Select a tool category (Web, Browser, Image Generation, or TTS), then choose **Nous Subscription** as the provider. This sets `use_gateway: true` for that tool in your config.

### Manual configuration

Set the `use_gateway` flag directly in `~/.hermes/config.yaml`:

```yaml
web:
  backend: firecrawl
  use_gateway: true

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

## How It Works

When `use_gateway: true` is set for a tool, the runtime routes API calls through the Nous Tool Gateway instead of using direct API keys:

1. **Web tools** — `web_search` and `web_extract` use the gateway's Firecrawl endpoint
2. **Image generation** — `image_generate` uses the gateway's FAL endpoint
3. **TTS** — `text_to_speech` uses the gateway's OpenAI Audio endpoint
4. **Browser** — `browser_navigate` and other browser tools use the gateway's Browser Use endpoint

The gateway authenticates using your Nous Portal credentials (stored in `~/.hermes/auth.json` after `hermes model`).

### Precedence

Each tool checks `use_gateway` first:

- **`use_gateway: true`** → route through the gateway, even if direct API keys exist in `.env`
- **`use_gateway: false`** (or absent) → use direct API keys if available, fall back to gateway only when no direct keys exist

This means you can switch between gateway and direct keys at any time without deleting your `.env` credentials.

## Switching Back to Direct Keys

To stop using the gateway for a specific tool:

```bash
hermes tools    # Select the tool → choose a direct provider
```

Or set `use_gateway: false` in config:

```yaml
web:
  backend: firecrawl
  use_gateway: false  # Now uses FIRECRAWL_API_KEY from .env
```

When you select a non-gateway provider in `hermes tools`, the `use_gateway` flag is automatically set to `false` to prevent contradictory config.

## Checking Status

```bash
hermes status
```

The **Nous Tool Gateway** section shows:

```
◆ Nous Tool Gateway
  Nous Portal   ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
  Modal           ○ available via subscription (optional)
```

Tools marked "active via Nous subscription" are routed through the gateway. Tools with their own keys show which provider is active.

## Advanced: Self-Hosted Gateway

For self-hosted or custom gateway deployments, you can override the gateway endpoints via environment variables in `~/.hermes/.env`:

```bash
TOOL_GATEWAY_DOMAIN=nousresearch.com     # Base domain for gateway routing
TOOL_GATEWAY_SCHEME=https                 # HTTP or HTTPS (default: https)
TOOL_GATEWAY_USER_TOKEN=your-token        # Auth token (normally auto-populated)
FIRECRAWL_GATEWAY_URL=https://...         # Override for the Firecrawl endpoint specifically
```

These env vars are always visible in the configuration regardless of subscription status — they're useful for custom infrastructure setups.

## FAQ

### Do I need to delete my existing API keys?

No. When `use_gateway: true` is set, the runtime skips direct API keys and routes through the gateway. Your keys stay in `.env` untouched. If you later disable the gateway, they'll be used again automatically.

### Can I use the gateway for some tools and direct keys for others?

Yes. The `use_gateway` flag is per-tool. You can mix and match — for example, gateway for web and image generation, your own ElevenLabs key for TTS, and Browserbase for browser automation.

### What if my subscription expires?

Tools that were routed through the gateway will stop working until you [renew your subscription](https://portal.nousresearch.com/manage-subscription) or switch to direct API keys via `hermes tools`.

### Does the gateway work with the messaging gateway?

Yes. The Tool Gateway routes tool API calls regardless of whether you're using the CLI, Telegram, Discord, or any other messaging platform. It operates at the tool runtime level, not the entry point level.

### Is Modal included?

Modal (serverless terminal backend) is available as an optional add-on through the Nous subscription. It's not enabled by the Tool Gateway prompt — configure it separately via `hermes setup terminal` or in `config.yaml`.
