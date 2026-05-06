---
sidebar_position: 5
title: "Adding Providers"
description: "How to add a new inference provider to Hermes Agent — auth, runtime resolution, CLI flows, adapters, tests, and docs"
---

# Adding Providers

Hermes can already talk to any OpenAI-compatible endpoint through the custom provider path. Do not add a built-in provider unless you want first-class UX for that service:

- provider-specific auth or token refresh
- a curated model catalog
- setup / `hermes model` menu entries
- provider aliases for `provider:model` syntax
- a non-OpenAI API shape that needs an adapter

If the provider is just "another OpenAI-compatible base URL and API key", a named custom provider may be enough.

## The mental model

A built-in provider has to line up across a few layers:

1. `hermes_cli/auth.py` decides how credentials are found.
2. `hermes_cli/runtime_provider.py` turns that into runtime data:
   - `provider`
   - `api_mode`
   - `base_url`
   - `api_key`
   - `source`
3. `run_agent.py` uses `api_mode` to decide how requests are built and sent.
4. `hermes_cli/models.py` and `hermes_cli/main.py` make the provider show up in the CLI. (`hermes_cli/setup.py` delegates to `main.py` automatically — no changes needed there.)
5. `agent/auxiliary_client.py` and `agent/model_metadata.py` keep side tasks and token budgeting working.

The important abstraction is `api_mode`.

- Most providers use `chat_completions`.
- Codex uses `codex_responses`.
- Anthropic uses `anthropic_messages`.
- A new non-OpenAI protocol usually means adding a new adapter and a new `api_mode` branch.

## Choose the implementation path first

### Path A — OpenAI-compatible provider

Use this when the provider accepts standard chat-completions style requests.

Typical work:

- add auth metadata
- add model catalog / aliases
- add runtime resolution
- add CLI menu wiring
- add aux-model defaults
- add tests and user docs

You usually do not need a new adapter or a new `api_mode`.

### Path B — Native provider

Use this when the provider does not behave like OpenAI chat completions.

Examples in-tree today:

- `codex_responses`
- `anthropic_messages`

This path includes everything from Path A plus:

- a provider adapter in `agent/`
- `run_agent.py` branches for request building, dispatch, usage extraction, interrupt handling, and response normalization
- adapter tests

## File checklist

### Required for every built-in provider

1. `hermes_cli/auth.py`
2. `hermes_cli/models.py`
3. `hermes_cli/runtime_provider.py`
4. `hermes_cli/main.py`
5. `agent/auxiliary_client.py`
6. `agent/model_metadata.py`
7. tests
8. user-facing docs under `website/docs/`

:::tip
`hermes_cli/setup.py` does **not** need changes. The setup wizard delegates provider/model selection to `select_provider_and_model()` in `main.py` — any provider added there is automatically available in `hermes setup`.
:::

### Additional for native / non-OpenAI providers

10. `agent/<provider>_adapter.py`
11. `run_agent.py`
12. `pyproject.toml` if a provider SDK is required

## Fast path: Simple API-key providers

If your provider is just an OpenAI-compatible endpoint that authenticates with a single API key, you do not need to touch `auth.py`, `runtime_provider.py`, `main.py`, or any of the other files in the full checklist below.

All you need is:

1. A plugin directory under `plugins/model-providers/<your-provider>/` containing:
   - `__init__.py` — calls `register_provider(profile)` at module-level
   - `plugin.yaml` — manifest (name, kind: model-provider, version, description)
2. That's it. Provider plugins auto-load the first time anything calls `get_provider_profile()` or `list_providers()` — bundled plugins (this repo) and user plugins at `$HERMES_HOME/plugins/model-providers/` both get picked up.

When you add a plugin and it calls `register_provider()`, the following wire up automatically:

1. `PROVIDER_REGISTRY` entry in `auth.py` (credential resolution, env-var lookup)
2. `api_mode` set to `chat_completions`
3. `base_url` sourced from the config or the declared env var
4. `env_vars` checked in priority order for the API key
5. `fallback_models` list registered for the provider
6. `--provider` CLI flag accepts the provider id
7. `hermes model` menu includes the provider
8. `hermes setup` wizard delegates to `main.py` automatically
9. `provider:model` alias syntax works
10. Runtime resolver returns the correct `base_url` and `api_key`
11. `HERMES_INFERENCE_PROVIDER` env-var override accepts the provider id
12. Fallback model activation can switch into the provider cleanly

User plugins at `$HERMES_HOME/plugins/model-providers/<name>/` override bundled plugins of the same name (last-writer-wins in `register_provider()`) — so third parties can monkey-patch or replace any built-in profile without editing the repo.

See `plugins/model-providers/nvidia/` or `plugins/model-providers/gmi/` as a template, and `plugins/model-providers/README.md` for the full contract.

## Full path: OAuth and complex providers

Use the full checklist below when your provider needs any of the following:

- OAuth or token refresh (Nous Portal, Codex, Google Gemini, Qwen Portal, Copilot)
- A non-OpenAI API shape that requires a new adapter (Anthropic Messages, Codex Responses)
- Custom endpoint detection or multi-region probing (z.ai, Kimi)
- A curated static model catalog or live `/models` fetch
- Provider-specific `hermes model` menu entries with bespoke auth flows

## Step 1: Pick one canonical provider id

Choose a single provider id and use it everywhere.

Examples from the repo:

- `openai-codex`
- `kimi-coding`
- `minimax-cn`

That same id should appear in:

- `PROVIDER_REGISTRY` in `hermes_cli/auth.py`
- `_PROVIDER_LABELS` in `hermes_cli/models.py`
- `_PROVIDER_ALIASES` in both `hermes_cli/auth.py` and `hermes_cli/models.py`
- CLI `--provider` choices in `hermes_cli/main.py`
- setup / model selection branches
- auxiliary-model defaults
- tests

If the id differs between those files, the provider will feel half-wired: auth may work while `/model`, setup, or runtime resolution silently misses it.

## Step 2: Add auth metadata in `hermes_cli/auth.py`

For API-key providers, add a `ProviderConfig` entry to `PROVIDER_REGISTRY` with:

- `id`
- `name`
- `auth_type="api_key"`
- `inference_base_url`
- `api_key_env_vars`
- optional `base_url_env_var`

Also add aliases to `_PROVIDER_ALIASES`.

Use the existing providers as templates:

- simple API-key path: Z.AI, MiniMax
- API-key path with endpoint detection: Kimi, Z.AI
- native token resolution: Anthropic
- OAuth / auth-store path: Nous, OpenAI Codex

Questions to answer here:

- What env vars should Hermes check, and in what priority order?
- Does the provider need base-URL overrides?
- Does it need endpoint probing or token refresh?
- What should the auth error say when credentials are missing?

If the provider needs something more than "look up an API key", add a dedicated credential resolver instead of shoving logic into unrelated branches.

## Step 3: Add model catalog and aliases in `hermes_cli/models.py`

Update the provider catalog so the provider works in menus and in `provider:model` syntax.

Typical edits:

- `_PROVIDER_MODELS`
- `_PROVIDER_LABELS`
- `_PROVIDER_ALIASES`
- provider display order inside `list_available_providers()`
- `provider_model_ids()` if the provider supports a live `/models` fetch

If the provider exposes a live model list, prefer that first and keep `_PROVIDER_MODELS` as the static fallback.

This file is also what makes inputs like these work:

```text
anthropic:claude-sonnet-4-6
kimi:model-name
```

If aliases are missing here, the provider may authenticate correctly but still fail in `/model` parsing.

## Step 4: Resolve runtime data in `hermes_cli/runtime_provider.py`

`resolve_runtime_provider()` is the shared path used by CLI, gateway, cron, ACP, and helper clients.

Add a branch that returns a dict with at least:

```python
{
    "provider": "your-provider",
    "api_mode": "chat_completions",  # or your native mode
    "base_url": "https://...",
    "api_key": "...",
    "source": "env|portal|auth-store|explicit",
    "requested_provider": requested_provider,
}
```

If the provider is OpenAI-compatible, `api_mode` should usually stay `chat_completions`.

Be careful with API-key precedence. Hermes already contains logic to avoid leaking an OpenRouter key to unrelated endpoints. A new provider should be equally explicit about which key goes to which base URL.

## Step 5: Wire the CLI in `hermes_cli/main.py`

A provider is not discoverable until it shows up in the interactive `hermes model` flow.

Update these in `hermes_cli/main.py`:

- `provider_labels` dict
- `providers` list in `select_provider_and_model()`
- provider dispatch (`if selected_provider == ...`)
- `--provider` argument choices
- login/logout choices if the provider supports those flows
- a `_model_flow_<provider>()` function, or reuse `_model_flow_api_key_provider()` if it fits

:::tip
`hermes_cli/setup.py` does not need changes — it calls `select_provider_and_model()` from `main.py`, so your new provider appears in both `hermes model` and `hermes setup` automatically.
:::

## Step 6: Keep auxiliary calls working

Two files matter here:

### `agent/auxiliary_client.py`

Add a cheap / fast default aux model to `_API_KEY_PROVIDER_AUX_MODELS` if this is a direct API-key provider.

Auxiliary tasks include things like:

- vision summarization
- web extraction summarization
- context compression summaries
- session-search summaries
- memory flushes

If the provider has no sensible aux default, side tasks may fall back badly or use an expensive main model unexpectedly.

### `agent/model_metadata.py`

Add context lengths for the provider's models so token budgeting, compression thresholds, and limits stay sane.

## Step 7: If the provider is native, add an adapter and `run_agent.py` support

If the provider is not plain chat completions, isolate the provider-specific logic in `agent/<provider>_adapter.py`.

Keep `run_agent.py` focused on orchestration. It should call adapter helpers, not hand-build provider payloads inline all over the file.

A native provider usually needs work in these places:

### New adapter file

Typical responsibilities:

- build the SDK / HTTP client
- resolve tokens
- convert OpenAI-style conversation messages to the provider's request format
- convert tool schemas if needed
- normalize provider responses back into what `run_agent.py` expects
- extract usage and finish-reason data

### `run_agent.py`

Search for `api_mode` and audit every switch point. At minimum, verify:

- `__init__` chooses the new `api_mode`
- client construction works for the provider
- `_build_api_kwargs()` knows how to format requests
- `_interruptible_api_call()` dispatches to the right client call
- interrupt / client rebuild paths work
- response validation accepts the provider's shape
- finish-reason extraction is correct
- token-usage extraction is correct
- fallback-model activation can switch into the new provider cleanly
- summary-generation and memory-flush paths still work

Also search `run_agent.py` for `self.client.`. Any code path that assumes the standard OpenAI client exists can break when a native provider uses a different client object or `self.client = None`.

### Prompt caching and provider-specific request fields

Prompt caching and provider-specific knobs are easy to regress.

Examples already in-tree:

- Anthropic has a native prompt-caching path
- OpenRouter gets provider-routing fields
- not every provider should receive every request-side option

When you add a native provider, double-check that Hermes is only sending fields that provider actually understands.

## Step 8: Tests

At minimum, touch the tests that guard provider wiring.

Common places:

- `tests/test_runtime_provider_resolution.py`
- `tests/test_cli_provider_resolution.py`
- `tests/test_cli_model_command.py`
- `tests/test_setup_model_selection.py`
- `tests/test_provider_parity.py`
- `tests/test_run_agent.py`
- `tests/test_<provider>_adapter.py` for a native provider

For docs-only examples, the exact file set may differ. The point is to cover:

- auth resolution
- CLI menu / provider selection
- runtime provider resolution
- agent execution path
- provider:model parsing
- any adapter-specific message conversion

Run tests with xdist disabled:

```bash
source venv/bin/activate
python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_cli_model_command.py tests/test_setup_model_selection.py -n0 -q
```

For deeper changes, run the full suite before pushing:

```bash
source venv/bin/activate
python -m pytest tests/ -n0 -q
```

## Step 9: Live verification

After tests, run a real smoke test.

```bash
source venv/bin/activate
python -m hermes_cli.main chat -q "Say hello" --provider your-provider --model your-model
```

Also test the interactive flows if you changed menus:

```bash
source venv/bin/activate
python -m hermes_cli.main model
python -m hermes_cli.main setup
```

For native providers, verify at least one tool call too, not just a plain text response.

## Step 10: Update user-facing docs

If the provider is meant to ship as a first-class option, update the user docs too:

- `website/docs/getting-started/quickstart.md`
- `website/docs/user-guide/configuration.md`
- `website/docs/reference/environment-variables.md`

A developer can wire the provider perfectly and still leave users unable to discover the required env vars or setup flow.

## OpenAI-compatible provider checklist

Use this if the provider is standard chat completions.

- [ ] `ProviderConfig` added in `hermes_cli/auth.py`
- [ ] aliases added in `hermes_cli/auth.py` and `hermes_cli/models.py`
- [ ] model catalog added in `hermes_cli/models.py`
- [ ] runtime branch added in `hermes_cli/runtime_provider.py`
- [ ] CLI wiring added in `hermes_cli/main.py` (setup.py inherits automatically)
- [ ] aux model added in `agent/auxiliary_client.py`
- [ ] context lengths added in `agent/model_metadata.py`
- [ ] runtime / CLI tests updated
- [ ] user docs updated

## Native provider checklist

Use this when the provider needs a new protocol path.

- [ ] everything in the OpenAI-compatible checklist
- [ ] adapter added in `agent/<provider>_adapter.py`
- [ ] new `api_mode` supported in `run_agent.py`
- [ ] interrupt / rebuild path works
- [ ] usage and finish-reason extraction works
- [ ] fallback path works
- [ ] adapter tests added
- [ ] live smoke test passes

## Common pitfalls

### 1. Adding the provider to auth but not to model parsing

That makes credentials resolve correctly while `/model` and `provider:model` inputs fail.

### 2. Forgetting that `config["model"]` can be a string or a dict

A lot of provider-selection code has to normalize both forms.

### 3. Assuming a built-in provider is required

If the service is just OpenAI-compatible, a custom provider may already solve the user problem with less maintenance.

### 4. Forgetting auxiliary paths

The main chat path can work while summarization, memory flushes, or vision helpers fail because aux routing was never updated.

### 5. Native-provider branches hiding in `run_agent.py`

Search for `api_mode` and `self.client.`. Do not assume the obvious request path is the only one.

### 6. Sending OpenRouter-only knobs to other providers

Fields like provider routing belong only on the providers that support them.

### 7. Updating `hermes model` but not `hermes setup`

Both flows need to know about the provider.

## Good search targets while implementing

If you are hunting for all the places a provider touches, search these symbols:

- `PROVIDER_REGISTRY`
- `_PROVIDER_ALIASES`
- `_PROVIDER_MODELS`
- `resolve_runtime_provider`
- `_model_flow_`
- `select_provider_and_model`
- `api_mode`
- `_API_KEY_PROVIDER_AUX_MODELS`
- `self.client.`

## Related docs

- [Provider Runtime Resolution](./provider-runtime.md)
- [Architecture](./architecture.md)
- [Contributing](./contributing.md)
