---
title: Image Generation
description: Generate high-quality images using FLUX 2 Pro with automatic upscaling via FAL.ai.
sidebar_label: Image Generation
sidebar_position: 6
---

# Image Generation

Hermes Agent can generate images from text prompts using FAL.ai's **FLUX 2 Pro** model with automatic 2x upscaling via the **Clarity Upscaler** for enhanced quality.

## Setup

### Get a FAL API Key

1. Sign up at [fal.ai](https://fal.ai/)
2. Generate an API key from your dashboard

### Configure the Key

```bash
# Add to ~/.hermes/.env
FAL_KEY=your-fal-api-key-here
```

### Install the Client Library

```bash
pip install fal-client
```

:::info
The image generation tool is automatically available when `FAL_KEY` is set. No additional toolset configuration is needed.
:::

## How It Works

When you ask Hermes to generate an image:

1. **Generation** — Your prompt is sent to the FLUX 2 Pro model (`fal-ai/flux-2-pro`)
2. **Upscaling** — The generated image is automatically upscaled 2x using the Clarity Upscaler (`fal-ai/clarity-upscaler`)
3. **Delivery** — The upscaled image URL is returned

If upscaling fails for any reason, the original image is returned as a fallback.

## Usage

Simply ask Hermes to create an image:

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a portrait of a wise old owl perched on an ancient tree branch
```

```
Make me a futuristic cityscape with flying cars and neon lights
```

## Parameters

The `image_generate_tool` accepts these parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `prompt` | *(required)* | — | Text description of the desired image |
| `aspect_ratio` | `"landscape"` | `landscape`, `square`, `portrait` | Image aspect ratio |
| `num_inference_steps` | `50` | 1–100 | Number of denoising steps (more = higher quality, slower) |
| `guidance_scale` | `4.5` | 0.1–20.0 | How closely to follow the prompt |
| `num_images` | `1` | 1–4 | Number of images to generate |
| `output_format` | `"png"` | `png`, `jpeg` | Image file format |
| `seed` | *(random)* | any integer | Random seed for reproducible results |

## Aspect Ratios

The tool uses simplified aspect ratio names that map to FLUX 2 Pro image sizes:

| Aspect Ratio | Maps To | Best For |
|-------------|---------|----------|
| `landscape` | `landscape_16_9` | Wallpapers, banners, scenes |
| `square` | `square_hd` | Profile pictures, social media posts |
| `portrait` | `portrait_16_9` | Character art, phone wallpapers |

:::tip
You can also use the raw FLUX 2 Pro size presets directly: `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`, `landscape_4_3`, `landscape_16_9`. Custom sizes up to 2048x2048 are also supported.
:::

## Automatic Upscaling

Every generated image is automatically upscaled 2x using FAL.ai's Clarity Upscaler with these settings:

| Setting | Value |
|---------|-------|
| Upscale Factor | 2x |
| Creativity | 0.35 |
| Resemblance | 0.6 |
| Guidance Scale | 4 |
| Inference Steps | 18 |
| Positive Prompt | `"masterpiece, best quality, highres"` + your original prompt |
| Negative Prompt | `"(worst quality, low quality, normal quality:2)"` |

The upscaler enhances detail and resolution while preserving the original composition. If the upscaler fails (network issue, rate limit), the original resolution image is returned automatically.

## Example Prompts

Here are some effective prompts to try:

```
A candid street photo of a woman with a pink bob and bold eyeliner
```

```
Modern architecture building with glass facade, sunset lighting
```

```
Abstract art with vibrant colors and geometric patterns
```

```
Portrait of a wise old owl perched on ancient tree branch
```

```
Futuristic cityscape with flying cars and neon lights
```

## Debugging

Enable debug logging for image generation:

```bash
export IMAGE_TOOLS_DEBUG=true
```

Debug logs are saved to `./logs/image_tools_debug_<session_id>.json` with details about each generation request, parameters, timing, and any errors.

## Safety Settings

The image generation tool runs with safety checks disabled by default (`safety_tolerance: 5`, the most permissive setting). This is configured at the code level and is not user-adjustable.

## Limitations

- **Requires FAL API key** — image generation incurs API costs on your FAL.ai account
- **No image editing** — this is text-to-image only, no inpainting or img2img
- **URL-based delivery** — images are returned as temporary FAL.ai URLs, not saved locally
- **Upscaling adds latency** — the automatic 2x upscale step adds processing time
- **Max 4 images per request** — `num_images` is capped at 4
