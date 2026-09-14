---
sidebar_position: 11
title: "Image Generation Provider Plugins"
description: "How to build an image-generation backend plugin for Hermes Agent"
---

# Building an Image Generation Provider Plugin

Image-gen provider plugins register a backend that services every `image_generate` tool call — DALL·E, gpt-image, Grok, Flux, Imagen, Stable Diffusion, fal, Replicate, a local ComfyUI rig, anything. Built-in providers (OpenAI, OpenAI-Codex, xAI, FAL, Krea, DeepInfra, OpenRouter, Meta Model API) all ship as plugins. You can add a new one, or override a bundled one, by dropping a directory into `plugins/image_gen/<name>/`.

:::tip
Image-gen is one of several **backend plugins** Hermes supports. The others (with more specialized ABCs) are [Memory Provider Plugins](/developer-guide/memory-provider-plugin), [Context Engine Plugins](/developer-guide/context-engine-plugin), and [Model Provider Plugins](/developer-guide/model-provider-plugin). General tool/hook/CLI plugins live in [Build a Hermes Plugin](/developer-guide/plugins).
:::

## How discovery works

Hermes scans for image-gen backends in three places:

1. **Bundled** — `<repo>/plugins/image_gen/<name>/` (auto-loaded with `kind: backend`, always available)
2. **User** — `~/.hermes/plugins/image_gen/<name>/` (opt-in via `plugins.enabled`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` function calls `ctx.register_image_gen_provider(...)` — that puts it into the registry in `agent/image_gen_registry.py`. The active provider is picked by `image_gen.provider` in `config.yaml`; `hermes tools` walks users through selection.

The `image_generate` tool wrapper asks the registry for the active provider and dispatches there. If no provider is registered, the tool surfaces a helpful error pointing at `hermes tools`.

## OpenAI-compatible Images endpoints

The bundled `openai-compatible` provider targets the Images generations endpoint
of a local service or gateway. Configure behavioral settings in the active
profile's `config.yaml`:

```yaml
image_gen:
  provider: openai-compatible
  model: your-image-model
  openai_compatible:
    base_url: http://localhost:8000/v1
    timeout: 300
```

The backend also appears in `hermes tools` → Image Generation. Set the API root
and model in the configuration above before selecting it. Its credential prompt
can save `OPENAI_COMPATIBLE_IMAGE_API_KEY` into the profile's `.env`; omit the key
for an unauthenticated local service. External secret sources are supported by
Hermes' normal profile-secret loader. Literal `api_key` values in `config.yaml`
are rejected with a migration message.

`base_url` may be a host root (the provider adds `/v1/images/generations`), an API
root such as `/v1` or `/proxy/v1` (it adds `/images/generations`), or the complete
generations endpoint. Trailing slashes are accepted. Redirects are rejected;
configure the destination API root directly. `timeout` must be finite and positive
and is the Requests connect/read timeout, not a total generation deadline.

Model precedence is `image_gen.openai_compatible.model`, then the legacy
`OPENAI_COMPATIBLE_IMAGE_MODEL` setting, then the model forwarded by Hermes from
`image_gen.model`, then `gpt-image-1`. The original
`OPENAI_COMPATIBLE_IMAGE_BASE_URL` remains a fallback when the configured API root
is absent; prefer `config.yaml` for both behavioral settings. Legacy environment
fallbacks honor the active profile scope when multiplexing is enabled.

The provider requests one image. It accepts JSON `data[0].b64_json` or
`data[0].url` and saves the result in the shared image cache. Aspect ratios map
to `1024x1024` (square), `1536x1024` (landscape), and `1024x1536` (portrait).
Image editing and reference inputs return `modality_unsupported`.

For endpoints returning `text/event-stream`, the provider consumes SSE frames
and returns one final image. This does not display live previews or progress.
Supported frames contain a JSON Images response or top-level `b64_json` / `url`:

```text
event: partial_image
data: {"b64_json": "preview-base64"}

event: done
data: {"data": [{"b64_json": "final-image-base64"}]}

data: {"usage": {"total_tokens": 1}}

data: [DONE]

```

Multiple `data:` lines are joined within one event. `done` image data takes
precedence over other image-bearing frames; usage/heartbeat frames do not erase
it. `partial_image` is never treated as a completed image, even at EOF or `[DONE]`.
An `error` event or JSON `error` field fails the request. A missing final image
also fails. The parser tolerates an EOF immediately after the last data line.
These are gateway response conventions; support does not imply compatibility
with the separate Responses/Codex image-generation protocol or every vendor's
stream event schema.

## Directory structure

```
plugins/image_gen/my-backend/
├── __init__.py      # ImageGenProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

A bundled plugin is complete at this point. User plugins at `~/.hermes/plugins/image_gen/<name>/` need to be added to `plugins.enabled` in `config.yaml` (or run `hermes plugins enable <name>`).

## The ImageGenProvider ABC

Subclass `agent.image_gen_provider.ImageGenProvider`. The only required members are the `name` property and the `generate()` method — everything else has sane defaults:

```python
# plugins/image_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)


class MyBackendImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        # Stable id used in image_gen.provider config. Lowercase, no spaces.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to name.title() if omitted.
        return "My Backend"

    def is_available(self) -> bool:
        # Return False if credentials or deps are missing.
        # The tool's availability gate calls this before dispatch.
        if not os.environ.get("MY_BACKEND_API_KEY"):
            return False
        try:
            import my_backend_sdk  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        # Catalog shown in `hermes tools` model picker.
        return [
            {
                "id": "my-model-fast",
                "display": "My Model (Fast)",
                "speed": "~5s",
                "strengths": "Quick iteration",
                "price": "$0.01/image",
            },
            {
                "id": "my-model-hq",
                "display": "My Model (HQ)",
                "speed": "~30s",
                "strengths": "Highest fidelity",
                "price": "$0.04/image",
            },
        ]

    def default_model(self) -> Optional[str]:
        return "my-model-fast"

    def get_setup_schema(self) -> Dict[str, Any]:
        # Metadata for the `hermes tools` picker — keys to prompt for at setup.
        return {
            "name": "My Backend",
            "badge": "paid",        # optional; shown as a short tag in the picker
            "tag": "One-line description shown under the name",
            "env_vars": [
                {
                    "key": "MY_BACKEND_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://my-backend.example.com/api-keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Declare whether this backend supports image-to-image / editing.
        # The tool layer surfaces this in the dynamic schema so the model
        # knows when `image_url` is honored. Default (if you omit this) is
        # text-only: {"modalities": ["text"], "max_reference_images": 0}.
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_input",
                provider=self.name,
                prompt="",
                aspect_ratio=aspect_ratio,
            )

        # Routing: if image_url (or reference_image_urls) is set, the call is
        # an image-to-image / edit request; otherwise text-to-image. Report
        # which path you took via the `modality` field of success_response.
        sources = []
        if image_url:
            sources.append(image_url)
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        modality = "image" if sources else "text"

        # Model selection precedence: env var → config → default. The helper
        # _resolve_model() in the built-in openai plugin is a good reference.
        model_id = kwargs.get("model") or self.default_model() or "my-model-fast"

        try:
            import my_backend_sdk
            client = my_backend_sdk.Client(api_key=os.environ["MY_BACKEND_API_KEY"])
            if modality == "image":
                result = client.edit(
                    prompt=prompt,
                    model=model_id,
                    image_urls=sources,
                )
            else:
                result = client.generate(
                    prompt=prompt,
                    model=model_id,
                    aspect_ratio=aspect_ratio,
                )

            # Two shapes supported:
            #   - URL string: return it as `image`
            #   - base64 data: save under $HERMES_HOME/cache/images/ via save_b64_image()
            if result.get("image_b64"):
                path = save_b64_image(
                    result["image_b64"],
                    prefix=self.name,
                    extension="png",
                )
                image = str(path)
            else:
                image = result["image_url"]

            return success_response(
                image=image,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider=self.name,
                modality=modality,
            )
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_image_gen_provider(MyBackendImageGenProvider())
```

## plugin.yaml

```yaml
name: my-backend
version: 1.0.0
description: My image backend — text-to-image via My Backend SDK
author: Your Name
kind: backend
requires_env:
  - MY_BACKEND_API_KEY
```

`kind: backend` is what routes the plugin to the image-gen registration path. `requires_env` is prompted during `hermes plugins install`.

## ABC reference

Full contract in `agent/image_gen_provider.py`. The methods you'll typically override:

| Member | Required | Default | Purpose |
|---|---|---|---|
| `name` | ✅ | — | Stable id used in `image_gen.provider` config |
| `display_name` | — | `name.title()` | Label shown in `hermes tools` |
| `is_available()` | — | `True` | Gate for missing creds/deps |
| `list_models()` | — | `[]` | Catalog for `hermes tools` model picker |
| `default_model()` | — | first from `list_models()` | Fallback when no model is configured |
| `get_setup_schema()` | — | minimal | Picker metadata + env-var prompts |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | — | The call |

## Response format

`generate()` must return a dict built via `success_response()` or `error_response()`. Both live in `agent/image_gen_provider.py`.

**Success:**
```python
success_response(
    image=<url-or-absolute-path>,
    model=<model-id>,
    prompt=<echoed-prompt>,
    aspect_ratio="landscape" | "square" | "portrait",
    provider=<your-provider-name>,
    extra={...},  # optional backend-specific fields
)
```

**Error:**
```python
error_response(
    error="human-readable message",
    error_type="provider_error" | "invalid_input" | "<exception class name>",
    provider=<your-provider-name>,
    model=<model-id>,
    prompt=<prompt>,
    aspect_ratio=<resolved aspect>,
)
```

The tool wrapper JSON-serializes the dict and hands it to the LLM. Errors are surfaced as the tool result; the LLM decides how to explain them to the user.

## Handling base64 vs URL output

Some backends return image URLs (fal, Replicate); others return base64 payloads (OpenAI gpt-image-2). For the base64 case, use `save_b64_image()` — it writes to `$HERMES_HOME/cache/images/<prefix>_<timestamp>_<uuid>.<ext>` and returns the absolute `Path`. Pass that path (as `str`) as `image=` in `success_response()`. Gateway delivery (Telegram photo bubble, Discord attachment) recognizes both URLs and absolute paths.

## User overrides

Drop a user plugin at `~/.hermes/plugins/image_gen/<name>/` with the same `name` property as a bundled one and enable it via `hermes plugins enable <name>` — the registry is last-writer-wins, so your version replaces the built-in. Useful for pointing an `openai` plugin at a private proxy, or swapping in a custom model catalog.

## Testing

```bash
export HERMES_HOME=/tmp/hermes-imggen-test
mkdir -p $HERMES_HOME/plugins/image_gen/my-backend
# …copy __init__.py + plugin.yaml into that dir…

export MY_BACKEND_API_KEY=your-test-key
hermes plugins enable my-backend

# Pick it as the active provider
echo "image_gen:" >> $HERMES_HOME/config.yaml
echo "  provider: my-backend" >> $HERMES_HOME/config.yaml

# Exercise it
hermes -z "Generate an image of a corgi in a spacesuit"
```

Or interactively: `hermes tools` → "Image Generation" → select `my-backend` → enter API key if prompted.

## Reference implementations

- **`plugins/image_gen/openai/__init__.py`** — gpt-image-2 at low/medium/high tiers as three virtual model IDs sharing one API model with different `quality` params. Good example of tiered models under a single backend + config.yaml precedence chain.
- **`plugins/image_gen/xai/__init__.py`** — Grok Imagine via xAI. Different shape (URL output, simpler catalog).
- **`plugins/image_gen/openai-codex/__init__.py`** — Codex-style Responses API variant reusing the OpenAI SDK with a different routing base URL.

## Distribute via pip

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-imggen = "my_backend_imggen_package"
```

`my_backend_imggen_package` must expose a top-level `register` function. See [Distribute via pip](/developer-guide/plugins#distribute-via-pip) in the general plugin guide for the full setup.

## Related pages

- [Image Generation](/user-guide/features/image-generation) — user-facing feature documentation
- [Plugins overview](/user-guide/features/plugins) — all plugin types at a glance
- [Build a Hermes Plugin](/developer-guide/plugins) — general tools/hooks/slash commands guide
