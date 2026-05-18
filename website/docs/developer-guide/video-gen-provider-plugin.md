---
sidebar_position: 12
title: "Video Generation Provider Plugins"
description: "How to build a video-generation backend plugin for Hermes Agent"
---

# Building a Video Generation Provider Plugin

Video-gen provider plugins register a backend that services every `video_generate` tool call. Built-in providers (xAI, FAL) ship as plugins. Add a new one, or override a bundled one, by dropping a directory into `plugins/video_gen/<name>/`.

:::tip
Video-gen mirrors [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) almost line-for-line — if you've built an image-gen backend, you already know the shape. The main differences: a `capabilities()` method advertising modalities/aspect-ratios/durations, and a routing convention (pass `image_url` to use image-to-video, omit it to use text-to-video — the provider picks the right endpoint internally).
:::

## The unified surface (one tool, two modalities)

The `video_generate` tool exposes two modalities through one parameter:

- **Text-to-video** — call with `prompt` only. The provider routes to its text-to-video endpoint.
- **Image-to-video** — call with `prompt` + `image_url`. The provider routes to its image-to-video endpoint.

Edit and extend are intentionally out of scope. Most backends don't support them and the inconsistency would force per-backend prose into the agent's tool description.

## How discovery works

Hermes scans for video-gen backends in three places:

1. **Bundled** — `<repo>/plugins/video_gen/<name>/` (auto-loaded with `kind: backend`)
2. **User** — `~/.hermes/plugins/video_gen/<name>/` (opt-in via `plugins.enabled`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` function calls `ctx.register_video_gen_provider(...)`. The active provider is picked by `video_gen.provider` in `config.yaml`; `hermes tools` → Video Generation walks users through selection. Unlike `image_generate`, there is no in-tree legacy backend — every provider is a plugin.

## Directory structure

```
plugins/video_gen/my-backend/
├── __init__.py      # VideoGenProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

## The VideoGenProvider ABC

Subclass `agent.video_gen_provider.VideoGenProvider`. Required: `name` property and `generate()` method.

```python
# plugins/video_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)


class MyVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "my-backend"

    @property
    def display_name(self) -> str:
        return "My Backend"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        # Each entry is a model FAMILY — a name the user picks once.
        # Your provider's generate() routes within the family based on
        # whether image_url was passed.
        return [
            {
                "id": "fast",
                "display": "Fast",
                "speed": "~30s",
                "strengths": "Cheapest tier",
                "price": "$0.05/s",
                "modalities": ["text", "image"],  # advisory
            },
        ]

    def default_model(self) -> Optional[str]:
        return "fast"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["720p", "1080p"],
            "min_duration": 1,
            "max_duration": 10,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "My Backend",
            "badge": "paid",
            "tag": "Short description shown in `hermes tools`",
            "env_vars": [
                {
                    "key": "MY_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://mybackend.example.com/keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,  # always ignore unknown kwargs for forward-compat
    ) -> Dict[str, Any]:
        # ROUTE: image_url presence picks the endpoint.
        if image_url:
            endpoint = "my-backend/image-to-video"
            modality_used = "image"
        else:
            endpoint = "my-backend/text-to-video"
            modality_used = "text"

        # ... call your API ...

        return success_response(
            video="https://your-cdn/output.mp4",
            model=model or "fast",
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio,
            duration=duration or 5,
            provider=self.name,
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(MyVideoGenProvider())
```

## The plugin manifest

```yaml
# plugins/video_gen/my-backend/plugin.yaml
name: my-backend
version: 1.0.0
description: "My video generation backend"
author: Your Name
kind: backend
requires_env:
  - MY_API_KEY
```

## The `video_generate` schema

The tool exposes one schema across every backend. Providers ignore parameters they don't support.

| Parameter | What it does |
|---|---|
| `prompt` | Text instruction (required) |
| `image_url` | When set → image-to-video; when omitted → text-to-video |
| `reference_image_urls` | Style/character refs (provider-dependent) |
| `duration` | Seconds — provider clamps |
| `aspect_ratio` | `"16:9"`, `"9:16"`, `"1:1"`, ... — provider clamps |
| `resolution` | `"480p"` / `"540p"` / `"720p"` / `"1080p"` — provider clamps |
| `negative_prompt` | Content to avoid (Pixverse/Kling only) |
| `audio` | Native audio (Veo3 / Pixverse pricing tier) |
| `seed` | Reproducibility |
| `model` | Override the active model/family |

The provider's `capabilities()` advertises which of these are honored. The agent sees the active backend's capabilities in the tool description, dynamically rebuilt when the user changes backend via `hermes tools`.

## Model families and endpoint routing (the FAL pattern)

When your backend has multiple endpoints per "model" — like FAL, where every family (Veo 3.1, Pixverse v6, Kling O3) has both a `/text-to-video` and an `/image-to-video` URL — represent each **family** as one catalog entry. Your `generate()` picks the right endpoint based on whether `image_url` was passed:

```python
FAMILIES = {
    "veo3.1": {
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        # ... family-specific capability flags ...
    },
}

def generate(self, prompt, *, image_url=None, model=None, **kwargs):
    family_id, family = _resolve_family(model)
    endpoint = family["image_endpoint"] if image_url else family["text_endpoint"]
    # ... build payload from family's declared capability flags, call endpoint ...
```

The user picks `veo3.1` once in `hermes tools`. The agent never thinks about endpoints — it just passes (or doesn't pass) `image_url`.

## Selection precedence

For per-instance model knobs (see `plugins/video_gen/fal/__init__.py`):

1. `model=` keyword from the tool call
2. `<PROVIDER>_VIDEO_MODEL` env var
3. `video_gen.<provider>.model` in `config.yaml`
4. `video_gen.model` in `config.yaml` (when it's one of your IDs)
5. Provider's `default_model()`

## Response shape

`success_response()` and `error_response()` produce the dict shape every backend returns. Use them — don't hand-roll the dict.

Success keys: `success`, `video` (URL or absolute path), `model`, `prompt`, `modality` (`"text"` or `"image"`), `aspect_ratio`, `duration`, `provider`, plus `extra`.

Error keys: `success`, `video` (None), `error`, `error_type`, `model`, `prompt`, `aspect_ratio`, `provider`.

## Where to save artifacts

If your backend returns base64, use `save_b64_video()` to write under `$HERMES_HOME/cache/videos/`. For raw bytes from a follow-up HTTP fetch, use `save_bytes_video()`. Otherwise return the upstream URL directly — the gateway resolves remote URLs on delivery.

## Testing

Drop a smoke test under `tests/plugins/video_gen/test_<name>_plugin.py`. The xAI and FAL tests show the pattern — register, verify catalog, exercise routing both with and without `image_url`, assert clean error responses on missing auth.
