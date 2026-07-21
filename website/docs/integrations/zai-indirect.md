---
title: "Z.ai Indirect"
sidebar_label: "Z.ai Indirect"
---

# Z.ai Indirect

`zai-indirect` is a separate, built-in provider for **GLM-5.2**. It uses Z.ai's Claude Code-compatible Anthropic Messages endpoint rather than the OpenAI-compatible Z.ai PaaS endpoint.

| Setting | Value |
|---|---|
| Provider ID | `zai-indirect` |
| Display name | Z.ai Indirect |
| Model | `glm-5.2` |
| API mode | `anthropic_messages` |
| Endpoint | `https://api.z.ai/api/anthropic` |
| Primary credential | `ZAI_INDIRECT_API_KEY` |

## Configure

Store the API key in the active Hermes profile's `.env` file. Do not put credentials in `config.yaml`, source files, shell history, or Git commits.

```dotenv
ZAI_INDIRECT_API_KEY=your-zai-api-key
```

The provider also accepts the existing `ZAI_API_KEY`, `GLM_API_KEY`, and `Z_AI_API_KEY` aliases. `ZAI_INDIRECT_API_KEY` is recommended when you want a credential scoped specifically to this provider.

Then restart the process that owns the model catalogue:

```bash
hermes gateway restart
```

For the desktop app, close and reopen Hermes Desktop after the gateway restart. Open the model picker in the composer and select:

```text
Z.AI INDIRECT
  GLM-5.2
```

The picker only shows explicitly configured providers. If the row is absent, confirm that the key is present in the active profile's `.env`, restart the gateway, and choose **Refresh Models** in the picker.

## Configure directly

You can select the provider from the terminal:

```bash
hermes chat --provider zai-indirect --model glm-5.2
```

Or persist it in `config.yaml`:

```yaml
model:
  provider: zai-indirect
  default: glm-5.2
```

Do not set `model.base_url` or `model.api_mode` for the normal built-in path. The provider profile supplies both values.

## Request routing

All main-agent traffic selected through `zai-indirect` follows this route:

```text
model picker / CLI
  -> runtime provider resolver
  -> provider=zai-indirect
  -> api_mode=anthropic_messages
  -> https://api.z.ai/api/anthropic
  -> agent/anthropic_adapter.py
```

The endpoint-specific request identity is applied only when the exact Z.ai Anthropic endpoint is detected. Native Anthropic OAuth and other third-party Anthropic-compatible providers keep their existing behaviour.

The provider deliberately exposes only `glm-5.2` and disables OpenAI-style `/models` probing, because this route is an Anthropic Messages endpoint rather than an OpenAI model-catalogue endpoint.

## Implementation reference

The implementation is split across these repository files:

- `plugins/model-providers/zai-indirect/__init__.py` — provider identity, endpoint, credential aliases, API mode, and the single-model catalogue.
- `plugins/model-providers/zai-indirect/plugin.yaml` — bundled model-provider manifest.
- `agent/anthropic_adapter.py` — Z.ai endpoint detection and Claude Code-compatible request construction.
- `agent/agent_init.py` — honours plugin-declared `base_url` and `api_mode` for provider-only callers.
- `agent/agent_runtime_helpers.py` and `agent/chat_completion_helpers.py` — preserve the endpoint-specific Anthropic route during provider switches and fallback flows.
- `hermes_cli/providers.py` — resolves a plugin-declared API mode when no hand-maintained provider overlay exists.
- `tests/plugins/model_providers/test_zai_indirect_profile.py` — provider, picker, manifest, and direct-construction regression tests.
- `tests/agent/test_zai_indirect_anthropic.py` — outbound request-identity and payload-sanitisation tests.

## Verification

Run the focused regression suite from the repository root on Windows:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/plugins/model_providers/test_zai_indirect_profile.py \
  tests/agent/test_zai_indirect_anthropic.py \
  tests/agent/test_anthropic_adapter.py \
  tests/plugins/model_providers/test_zai_profile.py -q
```

A no-network routing check can also be performed with a placeholder key:

```bash
.venv/Scripts/python.exe -c "from run_agent import AIAgent; a=AIAgent(provider='zai-indirect', model='glm-5.2', api_key='test-placeholder', quiet_mode=True, enabled_toolsets=[], skip_context_files=True, skip_memory=True); print(a.provider, a.api_mode, a.base_url, type(a._anthropic_client).__name__)"
```

Expected routing fields:

```text
zai-indirect anthropic_messages https://api.z.ai/api/anthropic Anthropic
```
