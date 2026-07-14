---
title: Image Generation
description: Generate images via FAL.ai — 11 models including FLUX 2, GPT Image (1.5 & 2), Nano Banana Pro, Ideogram, Recraft V4 Pro, Krea 2, and more, selectable via `hermes tools`.
sidebar_label: Image Generation
sidebar_position: 6
---

# Image Generation

Hermes Agent generates images from text prompts via FAL.ai. Eleven models are supported out of the box, each with different speed, quality, and cost tradeoffs. The active model is user-configurable via `hermes tools` and persists in `config.yaml`.

## Supported Models

| Model | Speed | Strengths | Price |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(default)* | `<1s` | Fast, crisp text | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6s | Studio photorealism | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2s | Bilingual EN/CN, 6B params | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8s | Gemini 3 Pro, reasoning depth, text rendering | $0.15/image (1K) |
| `fal-ai/gpt-image-1.5` | ~15s | Prompt adherence | $0.034/image |
| `fal-ai/gpt-image-2` | ~20s | SOTA text rendering + CJK, world-aware photorealism | $0.04–0.06/image |
| `fal-ai/ideogram/v3` | ~5s | Best typography | $0.03–0.09/image |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8s | Design, brand systems, production-ready | $0.25/image |
| `fal-ai/qwen-image` | ~12s | LLM-based, complex text | $0.02/MP |
| `fal-ai/krea/v2/medium/text-to-image` | ~15-25s | Illustration, anime, painting, expressive/artistic styles | $0.030–0.035/image |
| `fal-ai/krea/v2/large/text-to-image` | ~25-60s | Photorealism, raw textured looks (motion blur, grain, film) | $0.060–0.065/image |

Prices are FAL's pricing at time of writing; check [fal.ai](https://fal.ai/) for current numbers.

## Setup

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, you can use image generation through the **[Tool Gateway](tool-gateway.md)** without a FAL API key. Your model selection persists across both paths. New installs can run `hermes setup --portal` to log in and turn on every gateway tool at once; existing installs can pick **Nous Subscription** as the image-gen backend via `hermes tools`.

If the managed gateway returns `HTTP 4xx` for a specific model, that model isn't yet proxied on the portal side — the agent will tell you so, with remediation steps (set `FAL_KEY` for direct access, or pick a different model).
:::

### Get a FAL API Key

1. Sign up at [fal.ai](https://fal.ai/)
2. Generate an API key from your dashboard

### Configure and Pick a Model

Run the tools command:

```bash
hermes tools
```

Navigate to **🎨 Image Generation**, pick your backend (Nous Subscription or FAL.ai), then the picker shows all supported models in a column-aligned table — arrow keys to navigate, Enter to select:

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

Your selection is saved to `config.yaml`:

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # true if using Nous Subscription
```

### GPT-Image Quality

The `fal-ai/gpt-image-1.5` and `fal-ai/gpt-image-2` request quality is pinned to `medium` (~$0.034–$0.06/image at 1024×1024). We don't expose the `low` / `high` tiers as a user-facing option so that Nous Portal billing stays predictable across all users — the cost spread between tiers is 3–22×. If you want a cheaper option, pick Klein 9B or Z-Image Turbo; if you want higher quality, use Nano Banana Pro or Recraft V4 Pro.

## Usage

The agent-facing schema is intentionally minimal — the model picks up whatever you've configured:

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a square portrait of a wise old owl — use the typography model
```

```
Make me a futuristic cityscape, landscape orientation
```

## Image-to-Image / Editing

The same `image_generate` tool also **edits existing images** when the active
model supports it — pass a source image and the backend routes to its editing
endpoint automatically (mirrors how `video_generate` handles image-to-video).
Omit the source image and it's plain text-to-image.

```
Take this photo and make it a rainy Tokyo street at night → <image>
```

```
Blend these two product shots into one hero image → <image1> <image2>
```

Two inputs drive the edit:

- **`image_url`** — the primary source image to edit/transform (public URL or local path).
- **`reference_image_urls`** — additional style/composition references (capped per-model).

### Which backends support editing

| Backend | Image-to-image | Reference cap | How |
|---|---|---|---|
| **FAL.ai** (edit-capable models below) | ✓ | up to 9 | routes to the model's `/edit` endpoint |
| **OpenAI** (`gpt-image-2`) | ✓ | up to 16 | `images.edit()` |
| **xAI** (Grok Imagine) | ✓ | 1 | `/v1/images/edits` (`grok-imagine-image-quality`) |
| **Krea** (`Krea 2`) | ✓ | up to 10 | reference-guided generation (`image_style_references`) |
| **OpenAI (Codex auth)** | ✓ | up to 16 | Codex Responses `image_generation` tool with `input_image` content parts |

FAL models with an editing endpoint: `flux-2/klein/9b`, `flux-2-pro`,
`nano-banana-pro`, `gpt-image-1.5`, `gpt-image-2`, `ideogram/v3`, and
`qwen-image`. Pure text-to-image FAL models (`z-image/turbo`, `recraft`,
`krea/*`) reject image inputs with a clear error pointing you at an
edit-capable model.

The active model's editing capability is surfaced in the tool description at
runtime, so the agent knows whether `image_url` will be honored before it
calls the tool.

## Aspect Ratios

Every model accepts the same three aspect ratios from the agent's perspective. Internally, each model's native size spec is filled in automatically:

| Agent input | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2 maps to 4:3 presets rather than 16:9 because its minimum pixel count is 655,360 — the `landscape_16_9` preset (1024×576 = 589,824) would be rejected.

This translation happens in `_build_fal_payload()` — agent code never has to know about per-model schema differences.

## Automatic Upscaling

Upscaling via FAL's **Clarity Upscaler** is gated per-model:

| Model | Upscale? | Why |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | Backward-compat (was the pre-picker default) |
| All others | ✗ | Fast models would lose their sub-second value prop; hi-res models don't need it |

When upscaling runs, it uses these settings:

| Setting | Value |
|---|---|
| Upscale factor | 2× |
| Creativity | 0.35 |
| Resemblance | 0.6 |
| Guidance scale | 4 |
| Inference steps | 18 |

If upscaling fails (network issue, rate limit), the original image is returned automatically.

## How It Works Internally

1. **Model resolution** — `_resolve_fal_model()` reads `image_gen.model` from `config.yaml`, falls back to the `FAL_IMAGE_MODEL` env var, then to `fal-ai/flux-2/klein/9b`.
2. **Payload building** — `_build_fal_payload()` translates your `aspect_ratio` into the model's native format (preset enum, aspect-ratio enum, or GPT literal), merges the model's default params, applies any caller overrides, then filters to the model's `supports` whitelist so unsupported keys are never sent.
3. **Submission** — `_submit_fal_request()` routes via direct FAL credentials or the managed Nous gateway.
4. **Upscaling** — runs only if the model's metadata has `upscale: True`.
5. **Delivery** — final image URL returned to the agent, which emits a `MEDIA:<url>` tag that platform adapters convert to native media.

## Debugging

Enable debug logging:

```bash
export IMAGE_TOOLS_DEBUG=true
```

Debug logs go to `./logs/image_tools_debug_<session_id>.json` with per-call details (model, parameters, timing, errors).

## Platform Delivery

| Platform | Delivery |
|---|---|
| **CLI** | Image URL printed as markdown `![](url)` — click to open |
| **Telegram** | Photo message with the prompt as caption |
| **Discord** | Embedded in a message |
| **Slack** | URL unfurled by Slack |
| **WhatsApp** | Media message |
| **Others** | URL in plain text |

## Limitations

- **Requires credentials** for the active backend (FAL `FAL_KEY` / Nous Subscription, `OPENAI_API_KEY`, xAI OAuth, `KREA_API_KEY`)
- **Editing is model-dependent** — image-to-image works only on edit-capable models (see the table above); text-to-image-only models reject image inputs with a clear error
- **Temporary URLs** — backends return hosted URLs that expire after hours/days; Hermes materializes them to the local cache so delivery still works after expiry
- **Per-model constraints** — some models don't support `seed`, `num_inference_steps`, etc. The `supports` / `edit_supports` filter silently drops unsupported params; this is expected behavior
