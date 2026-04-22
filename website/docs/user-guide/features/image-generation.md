---
title: Image & Video Generation
description: Generate images with FAL.ai and videos with HappyHorse from Hermes Agent.
sidebar_label: Image & Video Generation
sidebar_position: 6
---

# Image & Video Generation

Hermes Agent currently supports two creative generation flows:

- Image generation via FAL.ai (`image_generate`)
- Video generation via HappyHorse (`video_generate`)

This page is the practical handoff doc for teammates who want to configure the keys, understand the available parameters, and run these tools successfully.

## Quick Setup

Add one or both keys to `~/.hermes/.env`:

```bash
FAL_KEY=YOUR_FAL_KEY
HAPPYHORSE_API_KEY=YOUR_HAPPYHORSE_KEY
```

You can also manage these in the Hermes config UI / Env page. `HAPPYHORSE_API_KEY` now appears there as a tool API key for `video_generate`.

## Tool Availability

- `image_generate` becomes available when `FAL_KEY` is set
- `video_generate` becomes available when `HAPPYHORSE_API_KEY` is set
- Both tools are part of the `image_gen` toolset

## Image Generation

Hermes can generate images from text prompts using FAL.ai's FLUX 2 Pro model with automatic 2x upscaling via the Clarity Upscaler.

### Image setup

1. Sign up at [fal.ai](https://fal.ai/)
2. Generate an API key from your dashboard
3. Add the key to `~/.hermes/.env`
4. Install the client library if needed:

```bash
pip install fal-client
```

### Image usage

Ask Hermes naturally, for example:

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Make me a futuristic cityscape with flying cars and neon lights
```

### `image_generate` parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `prompt` | required | — | Text description of the desired image |
| `aspect_ratio` | `landscape` | `landscape`, `square`, `portrait` | Image aspect ratio |
| `num_inference_steps` | `50` | 1–100 | Number of denoising steps |
| `guidance_scale` | `4.5` | 0.1–20.0 | How closely to follow the prompt |
| `num_images` | `1` | 1–4 | Number of images to generate |
| `output_format` | `png` | `png`, `jpeg` | Output format |
| `seed` | random | any integer | Reproducible generations |

### Image aspect ratios

| Aspect Ratio | Maps To | Best For |
|-------------|---------|----------|
| `landscape` | `landscape_16_9` | Scenes, banners, wallpapers |
| `square` | `square_hd` | Social posts, avatars |
| `portrait` | `portrait_16_9` | Character art, phone wallpapers |

### Image notes

- Generated images are automatically upscaled 2x
- If upscaling fails, Hermes returns the original image as fallback
- URLs are temporary and usually expire after some time

## Video Generation (HappyHorse)

Hermes can generate videos through HappyHorse using the `video_generate` tool.

### Video setup

1. Get a HappyHorse API key from [HappyHorse Docs](https://happyhorse.app/zh/docs)
2. Add it to `~/.hermes/.env`:

```bash
HAPPYHORSE_API_KEY=YOUR_HAPPYHORSE_KEY
```

### What the tool supports

- Text-to-video
- Image-to-video via `image_urls`
- Optional polling until the final video URL is ready
- Multi-shot video requests

### `video_generate` parameters

| Parameter | Default | Range / Values | Description |
|-----------|---------|----------------|-------------|
| `prompt` | required unless using `multi_shots` | — | Main video prompt |
| `mode` | `std` | `std`, `pro` | Quality / cost mode |
| `duration` | `5` | integer `3`–`15` | Video duration in seconds |
| `aspect_ratio` | `16:9` | `16:9`, `9:16`, `1:1` | Output frame ratio |
| `image_urls` | none | list of URLs | Optional source images for image-to-video |
| `sound` | `true` | boolean | Whether to include sound |
| `cfg_scale` | `0.5` | `0`–`1` | Prompt adherence strength |
| `multi_shots` | `false` | boolean | Enables multi-shot generation |
| `multi_prompt` | none | list | Shot definitions used with `multi_shots=true` |
| `happyhorse_elements` | none | list | Optional HappyHorse element definitions |
| `wait_for_completion` | `false` | boolean | Poll until terminal status |
| `poll_interval` | `5` | integer `>= 0` | Polling interval in seconds |
| `timeout` | `300` | integer `>= 1` | Max wait time when polling |

### HappyHorse usage examples

Natural-language request:

```
Generate a 3-second cinematic sunrise video over misty mountains in 16:9.
```

If you are calling the tool directly, the effective request shape is:

```json
{
  "prompt": "A calm cinematic sunrise over misty mountains, soft golden light, gentle camera movement, realistic nature footage",
  "mode": "std",
  "duration": 3,
  "aspect_ratio": "16:9",
  "wait_for_completion": true
}
```

### HappyHorse response behavior

The tool returns a JSON result with fields such as:

- `success`
- `task_id`
- `status`
- `response`
- `status_response` when polling is enabled
- `video_url` when a final result is available

Important live-API note:

- HappyHorse may return terminal success as `SUCCESS`, not only `COMPLETED`
- Hermes now treats both `SUCCESS` and `COMPLETED` as successful terminal states
- Final video URLs are read from `data.response.resultUrls[0]`

### HappyHorse operational notes

- Real API usage consumes HappyHorse credits
- Returned URLs may be temporary
- If `wait_for_completion=false`, you get back the task submission result immediately
- If `wait_for_completion=true`, Hermes polls `/api/status` until success, failure, cancellation, or timeout

## Platform delivery

Generated media is delivered differently depending on platform:

| Platform | Delivery method |
|----------|----------------|
| CLI | URL or media reference in terminal output |
| Telegram | Image as photo, video as media/URL depending on handler |
| Discord | Embedded media or URL |
| Slack / WhatsApp / others | URL or platform-native media handling |

Hermes internally uses `MEDIA:<url-or-path>` conventions where supported by the platform adapter.

## Troubleshooting

### `video_generate` does not appear

Check:

```bash
grep '^HAPPYHORSE_API_KEY' ~/.hermes/.env
```

Then restart Hermes or start a fresh session so tool availability is recomputed.

### Image works but video does not

Check that:

- `HAPPYHORSE_API_KEY` is present
- the session has the `image_gen` toolset available
- the request uses a valid `duration`, `aspect_ratio`, and either `prompt` or `multi_prompt`

### Polling times out

A timeout can still happen if the provider is slow. Retry with:

- a longer `timeout`
- a simpler prompt
- `wait_for_completion=false` followed by later status polling

## Validation performed for this integration

This HappyHorse integration was validated with:

- unit tests for request construction and polling behavior
- tests for successful terminal statuses including live `SUCCESS`
- config metadata tests so the key appears in the Env management UI
- a real HappyHorse API smoke test producing a final `video_url`
