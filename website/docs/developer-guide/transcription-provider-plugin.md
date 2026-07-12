---
sidebar_position: 15
title: "Transcription Provider Plugins"
description: "How to build a speech-to-text backend plugin for Hermes Agent"
---

# Building a Transcription (STT) Provider Plugin

Transcription provider plugins register a Python backend that services `transcribe_audio` calls — voice messages on gateway platforms (Telegram, Discord, WhatsApp, Slack, Signal) and push-to-talk dictation in the CLI/TUI all flow through the same dispatcher. Six built-in providers (`local` faster-whisper, `local_command`, `groq`, `openai`, `mistral`, `xai`) ship natively inside `tools/transcription_tools.py`; the plugin surface exists for new engines that need a Python implementation — SDKs without a CLI, local HTTP services, streaming chunks, OAuth-refreshing auth.

This page is the full developer reference. For the short version, see [Python plugin providers (STT)](/user-guide/features/tts#python-plugin-providers-stt) in the voice user guide.

:::tip
Transcription is one of several **backend plugins** Hermes supports. Its sibling is [TTS Provider Plugins](/developer-guide/tts-provider-plugin); the others are [Image Generation](/developer-guide/image-gen-provider-plugin), [Video Generation](/developer-guide/video-gen-provider-plugin), [Web Search](/developer-guide/web-search-provider-plugin), [Browser](/developer-guide/browser-provider-plugin), [Memory](/developer-guide/memory-provider-plugin), [Context Engine](/developer-guide/context-engine-plugin), and [Model Provider](/developer-guide/model-provider-plugin) plugins. General tool/hook/CLI plugins live in [Build a Hermes Plugin](/developer-guide/plugins).
:::

## Extension surfaces — resolution order

STT has three coexisting extension surfaces with a fixed collision order:

1. **Built-in providers** (`BUILTIN_STT_PROVIDERS` in `tools/transcription_tools.py`) — **always win**. The registry rejects a plugin registering a built-in name (`local`, `local_command`, `groq`, `openai`, `mistral`, `xai`) with a warning, and the dispatcher re-checks defensively. The single-env-var shell escape hatch `HERMES_LOCAL_STT_COMMAND` is part of the built-in `local_command` path.
2. **Command-type providers** — declared under `stt.providers.<name>: type: command` in `config.yaml`; any local CLI via shell-template placeholders, zero Python. **Wins over a same-name plugin** — config is more local than a plugin install. See [STT custom command providers](/user-guide/features/tts#stt-custom-command-providers).
3. **Plugin-registered providers** (this guide) — a `TranscriptionProvider` subclass registered via `ctx.register_transcription_provider()`.

If your engine is a single CLI that reads an audio file and prints text, use a command-type provider and skip the plugin entirely.

## How discovery works

Hermes scans for STT backends in three places:

1. **Bundled** — `<repo>/plugins/transcription/<name>/` (auto-loaded with `kind: backend`). Reserved for engines maintained in-tree; per [the placement policy](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md), plugins integrating a third-party product don't land here — ship them standalone.
2. **User** — `~/.hermes/plugins/transcription/<name>/` (opt-in via `plugins.enabled` / `hermes plugins enable <name>`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` function calls `ctx.register_transcription_provider(...)` — that puts it into the registry in `agent/transcription_registry.py`. The active provider is picked by `stt.provider` in `config.yaml`. Re-registration under the same name is last-writer-wins.

## Directory structure

```
~/.hermes/plugins/transcription/voicebox/
├── __init__.py      # TranscriptionProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

## The TranscriptionProvider ABC — worked example

Subclass `agent.transcription_provider.TranscriptionProvider`. The only required members are the `name` property and the `transcribe()` method.

The running example is a complete, working provider for [Voicebox](https://docs.voicebox.sh), a local-first voice studio whose FastAPI backend bundles Whisper (base → turbo) behind a loopback HTTP API. It's a useful reference because it covers the shapes a shell template can't: multipart upload to a local service, a model catalog, and a retryable "model still downloading" condition mapped onto the error envelope. It's also published as an installable standalone plugin — [`hermes-voicebox`](https://github.com/jamiepine/hermes-voicebox) — if you'd rather read (or just use) the shipping version, which adds a shared HTTP client, a TTS sibling, a bundled skill, and an offline test suite.

```python
# ~/.hermes/plugins/transcription/voicebox/__init__.py
"""Voicebox STT provider.

Voicebox's FastAPI backend listens on http://127.0.0.1:17493 while the
desktop app is running (Docker default: http://127.0.0.1:17600). No
auth on loopback. Whisper runs locally inside Voicebox — audio never
leaves the machine.
"""
import os
from typing import Any, Dict, List, Optional

import requests

from agent.transcription_provider import TranscriptionProvider

BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")

WHISPER_MODELS = [
    {"id": "base", "display": "Whisper Base (fastest)"},
    {"id": "small", "display": "Whisper Small"},
    {"id": "medium", "display": "Whisper Medium"},
    {"id": "large", "display": "Whisper Large"},
    {"id": "turbo", "display": "Whisper Large v3 Turbo"},
]


class VoiceboxTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        # Stable id used in stt.provider config. Lowercase, no spaces.
        return "voicebox"

    @property
    def display_name(self) -> str:
        return "Voicebox"

    def is_available(self) -> bool:
        # Voicebox only listens while the app is running, so availability
        # is a live health check. Must not raise.
        try:
            return requests.get(f"{BASE_URL}/health", timeout=2).ok
        except requests.RequestException:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return WHISPER_MODELS

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Voicebox",
            "badge": "local",
            "tag": "Local Whisper via the Voicebox app — audio stays on-device",
            "env_vars": [
                {
                    "key": "VOICEBOX_BASE_URL",
                    "prompt": "Voicebox API base URL (blank = http://127.0.0.1:17493)",
                    "url": "https://docs.voicebox.sh",
                },
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        # Contract: never raise — return the error envelope instead.
        data = {}
        if model:
            data["model"] = model
        if language:
            data["language"] = language
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    f"{BASE_URL}/transcribe",
                    files={"file": f},
                    data=data,
                    timeout=600,
                )
            if resp.status_code == 202:
                # Voicebox is downloading the requested Whisper model in
                # the background; the request must be retried later.
                detail = resp.json().get("detail", {})
                return self._error(
                    f"Voicebox is still downloading "
                    f"{detail.get('model_name', 'the Whisper model')} — "
                    "try again in a minute."
                )
            resp.raise_for_status()
            return {
                "success": True,
                "transcript": resp.json()["text"],
                "provider": self.name,
            }
        except requests.ConnectionError:
            return self._error(
                "Voicebox is not reachable — is the desktop app running?"
            )
        except (requests.RequestException, OSError, KeyError, ValueError) as exc:
            return self._error(str(exc))

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "transcript": "",
            "error": message,
            "provider": self.name,
        }


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_transcription_provider(VoiceboxTranscriptionProvider())
```

## plugin.yaml

```yaml
name: voicebox
version: 1.0.0
description: Voicebox — local Whisper transcription via the Voicebox desktop app
author: Your Name
kind: backend
```

`kind: backend` routes the plugin to the backend registration path. Add `requires_env` if your provider needs API keys prompted during `hermes plugins install`.

## ABC reference

Full contract in `agent/transcription_provider.py`. The members you'll typically override:

| Member | Required | Default | Purpose |
|---|---|---|---|
| `name` | ✅ | — | Stable id used in `stt.provider` config; built-in names are rejected |
| `display_name` | — | `name.title()` | Label shown in `hermes tools` |
| `is_available()` | — | `True` | Gate for missing creds/deps/running service; must not raise |
| `list_models()` | — | `[]` | Model catalog (`{id, display, languages, max_audio_seconds}`) |
| `default_model()` | — | first list entry | Fallback when no model is configured |
| `get_setup_schema()` | — | minimal | Picker metadata + env-var prompts |
| `transcribe(file_path, *, model, language, **extra)` | ✅ | — | The call |

## Response contract

`transcribe()` returns a dict — always, on both paths:

**Success:**
```python
{
    "success": True,
    "transcript": "the transcribed text",
    "provider": "<your provider name>",
}
```

**Failure:**
```python
{
    "success": False,
    "transcript": "",
    "error": "human-readable error message",
    "provider": "<your provider name>",
}
```

Implementations should **not raise** — convert exceptions to the error envelope so the dispatcher delivers a consistent shape to the gateway/CLI caller. (Note this is the opposite of the [TTS contract](/developer-guide/tts-provider-plugin#response-contract), where `synthesize()` raises and the dispatcher wraps.) The dispatcher validates the file's existence and size before calling you, so `file_path` is a readable audio file; language is an optional BCP-47 hint that providers without language support should ignore.

Error messages surface to end users on chat platforms ("couldn't transcribe your voice message: …"), so make them actionable — the Voicebox example distinguishes "app not running" from "model still downloading" from generic failures for exactly this reason.

## Testing

```bash
export HERMES_HOME=/tmp/hermes-stt-test
mkdir -p $HERMES_HOME/plugins/transcription/voicebox
# …copy __init__.py + plugin.yaml into that dir…

hermes plugins enable voicebox

# Pick it as the active provider
echo "stt:" >> $HERMES_HOME/config.yaml
echo "  provider: voicebox" >> $HERMES_HOME/config.yaml

# Exercise it: send a voice message on any connected gateway platform,
# or use push-to-talk in the CLI (hermes → hold the voice hotkey).
```

For unit tests, mock the HTTP layer — plugin tests must not hit live services (see [Contributing](/developer-guide/contributing)).

## Distribute via pip

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
voicebox-stt = "hermes_voicebox_stt"
```

`hermes_voicebox_stt` must expose a top-level `register` function. See [Distribute via pip](/developer-guide/plugins#distribute-via-pip) in the general plugin guide — the recommended channel for third-party-product providers, which [ship standalone](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) rather than in-tree. A single package can register both this provider and a [TTS provider](/developer-guide/tts-provider-plugin) from one `register(ctx)` function.

## Related pages

- [TTS Provider Plugins](/developer-guide/tts-provider-plugin) — the output-side sibling of this guide
- [Voice features](/user-guide/features/tts) — user-facing docs: voice message transcription, command-type providers, config reference
- [Plugins overview](/user-guide/features/plugins) — all plugin types at a glance
- [Build a Hermes Plugin](/developer-guide/plugins) — general tools/hooks/slash commands guide
