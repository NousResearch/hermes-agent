---
sidebar_position: 16
title: "xAI / Grok capability matrix"
description: "What Hermes Agent supports today for xAI and Grok, and which documented xAI API surfaces are not exposed"
---

# xAI / Grok capability matrix

What Hermes Agent supports today for [xAI](https://docs.x.ai/developers/models), and which documented xAI surfaces it does not call yet. Login itself is covered in [xAI Grok OAuth](./xai-grok-oauth.md).

Last reviewed against xAI docs on September 30, 2026.

## Access

| Surface | Status | How Hermes exposes it |
|---------|--------|------------------------|
| API-key chat (`xai`) | Supported | `XAI_API_KEY`, `codex_responses` transport, `https://api.x.ai/v1`. Aliases: `grok`, `x-ai`, `x.ai`. |
| SuperGrok / X Premium+ OAuth (`xai-oauth`) | Supported | Device-code login (`hermes auth add xai-oauth`). Same Responses transport. Aliases: `xai-grok-oauth`, `grok-oauth`, `x-ai-oauth`. |
| Local proxy | Supported | `hermes_cli/proxy` adapter `xai` forwards the OAuth bearer to `/responses`, `/chat/completions`, `/completions`, `/embeddings`, `/models`. |
| Grok on AWS Bedrock | Partial | Bedrock ids matching `xai.grok` (optional regional prefix) drop `temperature` / `topP`. Static context fallback for `xai.grok-4.6` is 500,000 tokens. |

Direct HTTP tools (images, video, web search) prefer the OAuth bearer, then `XAI_API_KEY`. Two metered tools invert that: **X Search and xAI TTS use an explicit `XAI_API_KEY` when one is set**, and fall back to the OAuth bearer only when no key is configured. STT does the same with the env `XAI_API_KEY` first.

## Chat

| Surface | Status | Notes |
|---------|--------|-------|
| [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning) | Partial | `reasoning.effort` is sent only for models whose id starts with `grok-3-mini`, `grok-4.20-multi-agent`, `grok-4.3`, `grok-4.5`, or `grok-4.6`. Other Grok ids get no `reasoning` field. |
| Effort ladder | Supported | Grok 4.6 (`grok-4.6` or `grok-4.6-*`): `low`, `medium`, `high`, `xhigh`. Older effort-capable Grok: `low`, `medium`, `high`. |
| [Priority Processing](https://docs.x.ai/developers/advanced-api-usage/priority-processing) | Partial | `/fast` sends `service_tier: "priority"` only for the Grok 4.6 family, and only on provider `xai` against `api.x.ai`. Other xAI paths strip `service_tier`. |
| [Prompt caching](https://docs.x.ai/developers/advanced-api-usage/prompt-caching) | Supported | On an xAI Responses request with a session id, Hermes sends `x-grok-conv-id` and body `prompt_cache_key` (via `extra_body`). |

## Tools and media

| Surface | Status | Notes |
|---------|--------|-------|
| [Web search](https://docs.x.ai/developers/tools/web-search) | Supported | `web.backend: xai` calls Responses with native `web_search`. Default model `grok-build-0.1`. `allowed_domains` and `excluded_domains` are mutually exclusive, each capped at 5. On the chat path, an active xAI search backend replaces the client `web_search` tool with `{"type": "web_search"}`. |
| [X Search](https://docs.x.ai/developers/tools/x-search) | Supported | Tool `x_search`. Default model `grok-4.5`. At most 10 handles in `allowed_x_handles` or `excluded_x_handles` (not both). |
| [Image generation](https://docs.x.ai/developers/model-capabilities/images/generation) | Supported | `plugins/image_gen/xai`: `grok-imagine-image` (default), `grok-imagine-image-2.0`, `grok-imagine-image-quality`. Live catalog from `/image-generation-models` when reachable. |
| Image editing | Partial | `POST /v1/images/edits`, up to 3 sources. An explicit model is used only if the catalog lists `image` input; otherwise Hermes falls back to `grok-imagine-image-quality`. |
| [Video generation](https://docs.x.ai/developers/model-capabilities/video/generation) | Supported | Text-to-video defaults to `grok-imagine-video`; image-to-video to `grok-imagine-video-1.5`. Resolutions `480p` and `720p`. Up to 7 reference images (not combined with a single `image_url`). |
| Video edit / extend | Supported | Tools `xai_video_edit` and `xai_video_extend` (`/videos/edits`, `/videos/extensions`). |
| [Text to speech](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech) | Supported | `POST /v1/tts`. Default voice `eve`, language `en`. Streaming uses `wss://api.x.ai/v1/tts` (PCM). |
| [Speech to text](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text) | Supported | `POST /v1/stt`. Optional `language`. `format` defaults on; `diarize` defaults off (`stt.xai`). A prompt is ignored. |
| [May 15, 2026 retirement](https://docs.x.ai/developers/migration/may-15-retirement) | Supported | `hermes doctor` and chat startup warn on retired refs. `hermes migrate xai` previews a rewrite; `--apply` writes it. Mapped ids: `grok-4-0709`, `grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`, `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`, `grok-code-fast-1`, `grok-3` → `grok-4.3`; `grok-imagine-image-pro` → `grok-imagine-image-quality`. |

## Not exposed yet

Hermes does not call these documented xAI surfaces:

| xAI surface | Docs |
|-------------|------|
| Batch API | [batch-api](https://docs.x.ai/developers/advanced-api-usage/batch-api) |
| Deferred chat completions | [deferred-chat-completions](https://docs.x.ai/developers/advanced-api-usage/deferred-chat-completions) |
| Server-side code execution | [code-execution](https://docs.x.ai/developers/tools/code-execution) |
| Collections search | [collections-search](https://docs.x.ai/developers/tools/collections-search) |
| Remote MCP tools | [remote-mcp](https://docs.x.ai/developers/tools/remote-mcp) |
| Files API | [files](https://docs.x.ai/developers/files) |
| Native context compaction (`/v1/responses/compact`) | [context-compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction) |
| Structured outputs on the xAI Responses path | [structured-outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs) |
| Exact billed cost (`usage.cost_in_usd_ticks`) | [cost-tracking](https://docs.x.ai/developers/cost-tracking) |
| Realtime speech-to-speech | [speech-to-speech](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech) |
| US regional endpoint | [regions](https://docs.x.ai/developers/advanced-api-usage/regions) |

When xAI adds, retires or renames a surface, update this page in the same PR.
