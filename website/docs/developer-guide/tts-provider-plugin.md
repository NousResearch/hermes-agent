---
sidebar_position: 14
title: "TTS Provider Plugins"
description: "How to build a text-to-speech backend plugin for Hermes Agent"
---

# Building a TTS Provider Plugin

TTS provider plugins register a Python backend that services `text_to_speech` tool calls — and voice replies in the CLI/TUI and gateway. Ten built-in providers (edge, openai, elevenlabs, minimax, xai, mistral, gemini, neutts, kittentts, piper) ship natively inside `tools/tts_tool.py`; the plugin surface exists for engines those built-ins don't cover and a shell command can't reasonably express — Python SDKs without a CLI, streaming synthesis, voice-listing APIs, OAuth-refreshing auth, or a local HTTP service.

This page is the full developer reference. For the short version (minimal plugin, when-to-pick-which table), see [Python plugin providers](/user-guide/features/tts#python-plugin-providers) in the TTS user guide.

:::tip
TTS is one of several **backend plugins** Hermes supports. Its sibling is [Transcription (STT) Provider Plugins](/developer-guide/transcription-provider-plugin); the others are [Image Generation](/developer-guide/image-gen-provider-plugin), [Video Generation](/developer-guide/video-gen-provider-plugin), [Web Search](/developer-guide/web-search-provider-plugin), [Browser](/developer-guide/browser-provider-plugin), [Memory](/developer-guide/memory-provider-plugin), [Context Engine](/developer-guide/context-engine-plugin), and [Model Provider](/developer-guide/model-provider-plugin) plugins. General tool/hook/CLI plugins live in [Build a Hermes Plugin](/developer-guide/plugins).
:::

## Three extension surfaces — resolution order

TTS has three coexisting extension surfaces. Know where yours sits before you build, because name collisions resolve in a fixed order:

1. **Built-in providers** (`BUILTIN_TTS_PROVIDERS` in `tools/tts_tool.py`) — native Python implementations. **Always win.** The registry rejects a plugin that tries to register a built-in name (`edge`, `openai`, `elevenlabs`, `minimax`, `xai`, `mistral`, `gemini`, `neutts`, `kittentts`, `piper`) with a warning, and the dispatcher re-checks defensively.
2. **Command-type providers** — declared under `tts.providers.<name>: type: command` in `config.yaml`. Wire any local CLI into Hermes with shell-template placeholders, zero Python. **Wins over a same-name plugin** — config is more local than a plugin install. See [Custom command providers](/user-guide/features/tts#custom-command-providers).
3. **Plugin-registered providers** (this guide) — a `TTSProvider` subclass registered via `ctx.register_tts_provider()`.

If your engine is a single CLI that reads text and writes an audio file, stop here and use a command-type provider — it's a config entry, not a plugin. Come back when you need Python.

## How discovery works

Hermes scans for TTS backends in three places:

1. **Bundled** — `<repo>/plugins/tts/<name>/` (auto-loaded with `kind: backend`). Reserved for engines maintained in-tree; per [the placement policy](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md), plugins integrating a third-party product don't land here — ship them standalone.
2. **User** — `~/.hermes/plugins/tts/<name>/` (opt-in via `plugins.enabled` / `hermes plugins enable <name>`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` function calls `ctx.register_tts_provider(...)` — that puts it into the registry in `agent/tts_registry.py`. The active provider is picked by `tts.provider` in `config.yaml`; `hermes tools` walks users through selection. Re-registration under the same name is last-writer-wins, so a user plugin can override a pip-installed one.

## Directory structure

```
~/.hermes/plugins/tts/voicebox/
├── __init__.py      # TTSProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

## The TTSProvider ABC — worked example

Subclass `agent.tts_provider.TTSProvider`. The only required members are the `name` property and the `synthesize()` method — everything else has sane defaults.

The running example below is a complete, working provider for [Voicebox](https://docs.voicebox.sh), a local-first voice studio (voice cloning, preset voices, seven TTS engines). It's a good reference precisely because it exercises the parts a shell template can't: an HTTP API on localhost, a voice catalog fetched at runtime, and a fixed output format that differs from the requested one. It's also published as an installable standalone plugin — [`hermes-voicebox`](https://github.com/jamiepine/hermes-voicebox) on [PyPI](https://pypi.org/project/hermes-voicebox/) (`pip install hermes-voicebox`) — if you'd rather read (or just use) the shipping version, which adds a shared HTTP client, an STT sibling, a bundled skill, and an offline test suite.

```python
# ~/.hermes/plugins/tts/voicebox/__init__.py
"""Voicebox TTS provider.

Voicebox's FastAPI backend listens on http://127.0.0.1:17493 while the
desktop app is running (Docker default: http://127.0.0.1:17600). No
auth on loopback. Mapping:

    Hermes ``voice``  ->  Voicebox profile (name or id)
    Hermes ``model``  ->  Voicebox engine (qwen, kokoro, chatterbox, ...)
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.tts_provider import TTSProvider

BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")

ENGINES = [
    {"id": "qwen", "display": "Qwen3-TTS (voice cloning)"},
    {"id": "qwen_custom_voice", "display": "Qwen CustomVoice (presets)"},
    {"id": "kokoro", "display": "Kokoro (fast presets)"},
    {"id": "chatterbox", "display": "Chatterbox Multilingual"},
    {"id": "chatterbox_turbo", "display": "Chatterbox Turbo"},
    {"id": "luxtts", "display": "LuxTTS"},
    {"id": "tada", "display": "HumeAI TADA"},
]


class VoiceboxTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        # Stable id used in tts.provider config. Lowercase, no spaces.
        return "voicebox"

    @property
    def display_name(self) -> str:
        return "Voicebox"

    def is_available(self) -> bool:
        # Voicebox only listens while the app is running, so availability
        # is a live health check, not an env-var check. Must not raise.
        try:
            return requests.get(f"{BASE_URL}/health", timeout=2).ok
        except requests.RequestException:
            return False

    def list_voices(self) -> List[Dict[str, Any]]:
        # Voice catalog fetched from the backend — this is what makes a
        # plugin worth it over a command provider for this engine.
        resp = requests.get(f"{BASE_URL}/profiles", timeout=10)
        resp.raise_for_status()
        return [
            {
                "id": p["id"],
                "display": p["name"],
                "language": p.get("language"),
            }
            for p in resp.json()
        ]

    def list_models(self) -> List[Dict[str, Any]]:
        return ENGINES

    def default_model(self) -> Optional[str]:
        # None = defer to the Voicebox profile's own default engine
        # (see the `"engine": None` note in synthesize()).
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Voicebox",
            "badge": "local",
            "tag": "Local-first voice cloning TTS — desktop app must be running",
            "env_vars": [
                {
                    "key": "VOICEBOX_BASE_URL",
                    "prompt": "Voicebox API base URL (blank = http://127.0.0.1:17493)",
                    "url": "https://docs.voicebox.sh",
                },
            ],
        }

    def _resolve_profile_id(self, voice: Optional[str]) -> str:
        resp = requests.get(f"{BASE_URL}/profiles", timeout=10)
        resp.raise_for_status()
        profiles = resp.json()
        if not profiles:
            raise RuntimeError(
                "No voice profiles exist in Voicebox — create one in the app first."
            )
        if not voice:
            return profiles[0]["id"]
        for p in profiles:
            if p["id"] == voice or p["name"].lower() == voice.strip().lower():
                return p["id"]
        raise RuntimeError(
            f"No Voicebox profile named {voice!r}. "
            f"Available: {', '.join(p['name'] for p in profiles)}"
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "profile_id": self._resolve_profile_id(voice),
            "text": text,
            # Explicit null defers to the profile's default engine —
            # omitting the key would fall back to the API default (qwen).
            "engine": model or None,
        }
        if extra.get("language"):
            payload["language"] = extra["language"]

        # Synchronous endpoint: returns WAV bytes in one call. Local
        # inference can be slow on first generation (model load).
        resp = requests.post(
            f"{BASE_URL}/generate/stream", json=payload, timeout=600
        )
        resp.raise_for_status()

        # Voicebox always emits WAV. The ABC allows substituting the
        # closest format as long as the extension matches the bytes.
        if format != "wav":
            output_path = str(Path(output_path).with_suffix(".wav"))
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return output_path

    @property
    def voice_compatible(self) -> bool:
        # WAV output — the gateway ffmpeg-converts to Opus for voice
        # bubbles on Telegram et al.
        return True


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_tts_provider(VoiceboxTTSProvider())
```

## plugin.yaml

```yaml
name: voicebox
version: 1.0.0
description: Voicebox — local-first voice cloning TTS via the Voicebox desktop app
author: Your Name
kind: backend
```

`kind: backend` is what routes the plugin to the backend registration path. Add `requires_env` if your provider needs API keys prompted during `hermes plugins install` (Voicebox needs none — it's a loopback HTTP call).

## ABC reference

Full contract in `agent/tts_provider.py`. The members you'll typically override:

| Member | Required | Default | Purpose |
|---|---|---|---|
| `name` | ✅ | — | Stable id used in `tts.provider` config; built-in names are rejected |
| `display_name` | — | `name.title()` | Label shown in `hermes tools` |
| `is_available()` | — | `True` | Gate for missing creds/deps/running service; must not raise |
| `list_voices()` | — | `[]` | Voice catalog (`{id, display, language, gender, preview_url}`) |
| `list_models()` | — | `[]` | Model catalog (`{id, display, languages, max_text_length}`) |
| `default_voice()` / `default_model()` | — | first list entry | Fallbacks when nothing is configured |
| `get_setup_schema()` | — | minimal | Picker metadata + env-var prompts |
| `synthesize(text, output_path, *, voice, model, speed, format, **extra)` | ✅ | — | The call |
| `stream(text, *, voice, model, format, **extra)` | — | `NotImplementedError` | Chunked bytes for streaming delivery |
| `voice_compatible` | — | `False` | Opt in to gateway voice-bubble delivery |

## Response contract

`synthesize()` writes the audio bytes to `output_path` and returns the path as a string. **Raise on failure** — the dispatcher converts exceptions into the standard `{success: False, error: …}` JSON envelope the rest of Hermes expects. (Note this is the opposite of the [transcription contract](/developer-guide/transcription-provider-plugin#response-contract), where implementations return an error envelope instead of raising.)

Two contract details worth knowing:

- **Text is pre-truncated.** The dispatcher truncates `text` to the provider's max length before calling you.
- **Format substitution is allowed.** If your engine can't produce the requested `format`, produce the closest equivalent and make sure the returned path's extension matches the actual bytes — the Voicebox example does exactly this (always WAV).

## Streaming (optional)

Override `stream()` to yield audio bytes as they're generated. The default raises `NotImplementedError`, and the dispatcher falls back to `synthesize()` + read-whole-file, so only implement it if your backend genuinely streams. The default `format` for streaming is `opus` because the primary use case is voice-bubble delivery.

## Voice bubbles: `voice_compatible`

When `voice_compatible` is `True`, the gateway's voice-message pipeline delivers your output as a voice bubble (Telegram et al.), running ffmpeg conversion to Opus if needed. When `False` (the default), output is delivered as a regular audio attachment. Opt in if your output is a standard PCM/lossy format ffmpeg can convert.

## Testing

```bash
export HERMES_HOME=/tmp/hermes-tts-test
mkdir -p $HERMES_HOME/plugins/tts/voicebox
# …copy __init__.py + plugin.yaml into that dir…

hermes plugins enable voicebox

# Pick it as the active provider
echo "tts:" >> $HERMES_HOME/config.yaml
echo "  provider: voicebox" >> $HERMES_HOME/config.yaml

# Exercise it
hermes -z "Say hello out loud"
```

Or interactively: `hermes tools` → "Text-to-Speech" → select `voicebox`. For unit tests, mock the HTTP layer — plugin tests must not hit live services (see [Contributing](/developer-guide/contributing)).

## Distribute via pip

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
voicebox-tts = "hermes_voicebox_tts"
```

`hermes_voicebox_tts` must expose a top-level `register` function. See [Distribute via pip](/developer-guide/plugins#distribute-via-pip) in the general plugin guide for the full setup — this is the recommended channel for third-party-product providers, which [ship standalone](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) rather than in-tree.

## Related pages

- [Transcription (STT) Provider Plugins](/developer-guide/transcription-provider-plugin) — the input-side sibling of this guide
- [Text-to-Speech](/user-guide/features/tts) — user-facing feature documentation, command-type providers, config reference
- [Plugins overview](/user-guide/features/plugins) — all plugin types at a glance
- [Build a Hermes Plugin](/developer-guide/plugins) — general tools/hooks/slash commands guide
