---
sidebar_position: 10
title: "Model Provider Plugins"
description: "How to build a model provider (inference backend) plugin for Hermes Agent"
---

# Building a Model Provider Plugin

Model provider plugins declare an inference backend — an OpenAI-compatible endpoint, an Anthropic Messages server, a Codex-style Responses API, or a Bedrock-native surface — that Hermes can route `AIAgent` calls through. Every built-in provider (OpenRouter, Anthropic, GMI, DeepSeek, Nvidia, …) ships as one of these plugins. Third parties can add their own by dropping a directory under `$HERMES_HOME/plugins/model-providers/` with zero changes to the repo.

:::tip
Model provider plugins are the third kind of **provider plugin**. The others are [Memory Provider Plugins](./memory-provider-plugin.md) (cross-session knowledge) and [Context Engine Plugins](./context-engine-plugin.md) (context compression strategies). All three follow the same "drop a directory, declare a profile, no repo edits" pattern.
:::

## How discovery works

`providers/__init__.py._discover_providers()` runs lazily the first time any code calls `get_provider_profile()` or `list_providers()`. Discovery order:

1. **Bundled plugins** — `<repo>/plugins/model-providers/<name>/` — ship with Hermes
2. **User plugins** — `$HERMES_HOME/plugins/model-providers/<name>/` — drop in a directory; a running process picks it up on its next provider lookup (no restart)
3. **Installed plugins** — `$HERMES_HOME/plugins/<name>/` (where `hermes plugins install owner/repo` clones) — imported only when `plugin.yaml` declares `kind: model-provider`; every other kind there belongs to the general PluginManager
4. **Legacy single-file** — `<repo>/providers/<name>.py` — back-compat for out-of-tree editable installs

Steps 2 and 3 are **per profile home**: one process that serves several profiles (the multiplex gateway a local Desktop attaches to, a `hermes serve` backend) resolves the plugins of whichever profile's `$HERMES_HOME` is bound at lookup time, and a plugin installed in one profile is not visible from another. Install the plugin in every profile that should use it (`hermes -p <profile> plugins install ...`).

**User plugins override bundled plugins of the same name** because `register_provider()` is last-writer-wins. Drop a `$HERMES_HOME/plugins/model-providers/gmi/` directory to replace the built-in GMI profile without touching the repo.

## Directory structure

```
plugins/model-providers/my-provider/
├── __init__.py       # Calls register_provider(profile) at module-level
├── plugin.yaml       # kind: model-provider + metadata (optional but recommended)
└── README.md         # Setup instructions (optional)
```

The only required file is `__init__.py`. `plugin.yaml` is used by `hermes plugins` for introspection and by the general PluginManager to route the plugin to the right loader; without it, the general loader falls back to a source-text heuristic.

## Minimal example — a simple API-key provider

```python
# plugins/model-providers/acme-inference/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

acme = ProviderProfile(
    name="acme-inference",
    aliases=("acme",),
    display_name="Acme Inference",
    description="Acme — OpenAI-compatible direct API",
    signup_url="https://acme.example.com/keys",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=(
        "acme-large-v3",
        "acme-medium-v3",
        "acme-small-fast",
    ),
)

register_provider(acme)
```

```yaml
# plugins/model-providers/acme-inference/plugin.yaml
name: acme-inference
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
author: Your Name
```

That's it. After dropping these two files, the following **auto-wire** with no other edits:

| Integration | Where | What it gets |
|---|---|---|
| Credential resolution | `hermes_cli/auth.py` | `PROVIDER_REGISTRY["acme-inference"]` populated from profile |
| `--provider` CLI flag | `hermes_cli/main.py` | Accepts `acme-inference` |
| `/model --provider`, model picker switch | `hermes_cli/providers.py::resolve_provider_full` | Resolves `acme-inference` and every alias to the profile (switch lands on `name`, so `acme` persists as `acme-inference`); user `providers:` / `custom_providers:` blocks keep precedence. A profile with an empty `base_url` (endpoint minted at runtime) resolves too, on the last rung |
| `hermes model` picker | `hermes_cli/models.py` | Appears in `CANONICAL_PROVIDERS`, model list fetched from `{base_url}/models` |
| `hermes doctor` | `hermes_cli/doctor.py` | Health check for `ACME_API_KEY` + `{base_url}/models` probe |
| `hermes setup` | `hermes_cli/config.py` | `ACME_API_KEY` appears in `OPTIONAL_ENV_VARS` and the setup wizard |
| URL reverse-mapping | `agent/model_metadata.py` | Hostname → provider name for auto-detection |
| Auxiliary model | `agent/auxiliary_client.py` | Uses `default_aux_model` for compression / summarization |
| Runtime resolution | `hermes_cli/runtime_provider.py` | Returns correct `base_url`, `api_key`, `api_mode` |
| Transport | `agent/transports/chat_completions.py` | Profile path generates kwargs via `prepare_messages` / `build_extra_body` / `build_api_kwargs_extras` |

## ProviderProfile fields

Full definition in `providers/base.py`. The most useful ones:

| Field | Type | Purpose |
|---|---|---|
| `name` | str | Canonical id — matches `model.provider` in `config.yaml` and the `--provider` flag |
| `aliases` | `tuple[str, ...]` | Alternative names resolved by `get_provider_profile()` (e.g. `grok` → `xai`) |
| `api_mode` | str | `chat_completions` \| `codex_responses` \| `anthropic_messages` \| `bedrock_converse` |
| `display_name` | str | Human label shown in `hermes model` picker |
| `description` | str | Picker subtitle |
| `signup_url` | str | Shown during first-run setup ("get an API key here") |
| `hidden` | bool | Pre-release: kept off every discovery surface (provider pickers, setup list, dashboard accounts tab) until `listed()` is true — by default once the user signs in by name (`hermes auth add <name>` writes a pool row). Resolution by name is never gated |
| `env_vars` | `tuple[str, ...]` | API-key env vars in priority order; a final `*_BASE_URL` entry is used as the user base-URL override |
| `base_url` | str | Default inference endpoint |
| `models_url` | str | Explicit catalog URL (falls back to `{base_url}/models`) |
| `supports_streaming` | `bool \| None` | `None` (default) preserves transport defaults; `True` opts streaming-capable `external_process` clients into host streaming; `False` disables provider streaming. ACP URLs and session-level streaming disables still take precedence. Only opt in when the client implements the selected API mode's streaming contract, including cancellation and cleanup. |
| `auth_type` | str | `api_key` \| `oauth_device_code` \| `oauth_external` \| `aws_sdk` \| `external_process` \| any plugin-defined value (login is then provider-owned) |
| `auth_handler` | `Callable \| None` | Provider-owned `hermes auth add/status/logout/refresh <name>` — see [Provider-owned auth](#provider-owned-auth-auth_handler-refresh_credential) |
| `refresh_credential` | `Callable \| None` | Provider-owned rotation of a pooled OAuth row — same section |
| `classify_api_error` | `Callable \| None` | Provider-scoped error-classification override — see [Recovery and error classification](#recovery-and-error-classification) |
| `fallback_models` | `tuple[str, ...]` | Curated list shown when live catalog fetch fails — in the `/model` picker AND the first-time `hermes setup` / `hermes model` API-key flow, which resolve the catalog the same way (`fetch_models()` merged curated-first with `fallback_models`; `fallback_models` alone when the fetch returns `None` or raises) |
| `supports_vision` | bool | Declares the provider's API accepts image content inside **tool-result** messages (a provider-wide wire capability). Per-model user-image routing comes from `model_capabilities` / models.dev, not from this flag |
| `model_capabilities` | `dict[str, dict[str, Any]]` | Per-model capability declarations in the `model_overrides` schema — see [Declaring model capabilities](#declaring-model-capabilities) |
| `default_headers` | `dict[str, str]` | Sent on every request (e.g. an `Editor-Version` header an IDE-subscription API requires); also forwarded by the default `fetch_models()` catalog request. A value that is costly to compute at import (a User-Agent carrying the release) can be deferred with `providers.base.LazyHeaders({"User-Agent": lambda: hermes_user_agent("MyAgent")})`, a `dict` whose callable values run on first read (`hermes_user_agent` is in `hermes_cli.version_info`) |
| `fixed_temperature` | Any | `None` = use caller's value; `OMIT_TEMPERATURE` sentinel = don't send temperature at all (Kimi) |
| `default_max_tokens` | `int \| None` | Provider-level max_tokens cap (Nvidia: 16384) |
| `unsupported_response_formats` | `tuple` | `response_format` types the API rejects outright; auxiliary requests omit them instead of paying a guaranteed 400 (DeepSeek: `("json_schema",)`) |
| `default_aux_model` | str | Cheap model for auxiliary tasks (compression, vision, summarization) |

## Declaring model capabilities

Hermes resolves per-model capabilities (`supports_reasoning`, `supports_vision`,
`supports_tools`, `context_window`) from the models.dev catalog, which does not
know an out-of-tree provider's models. Declare them once on the profile:

```python
register_provider(ProviderProfile(
    name="acme",
    auth_type="api_key",
    env_vars=("ACME_API_KEY",),
    base_url="https://api.acme.example/v1",
    fallback_models=("acme-large-high", "acme-small"),
    model_capabilities={
        "acme-large-high": {
            "supports_reasoning": False,   # reasoning tier is fixed by the model id
            "supports_vision": True,
            "supports_tools": True,
            "context_window": 64000,
            "model_family": "acme",
        },
    },
))
```

Keys are exact model IDs. Values use the `model_overrides` schema from
`config.yaml` — the three capability booleans, a positive `context_window`, an
optional `model_family` — and omitted fields stay unknown (not `False`), so a
partial entry patches catalog metadata without erasing it.

One declaration feeds every consumer that reads the catalog through
`agent.models_dev`: the `/model` picker's `reasoning` badge, image routing
(`decide_image_input_mode` goes `native` for a `supports_vision: True` model
even when the profile-wide `supports_vision` is unset), context-window lookup,
and the dashboard's `/api/model/info`. Precedence: explicit user
`model_overrides.<provider>.<model>` → plugin declaration → catalog → fill-gap
`_default`. Models the plugin does not declare keep the catalog/heuristic path.

Not covered: the picker's `fast` badge (a model-name heuristic in
`hermes_cli/models.py::model_supports_fast_mode`), reasoning-effort vocabulary
(`agent/reasoning_effort.py`), and transport request fields. Declarations do not
add models to a picker — use `fallback_models` / `fetch_models` for that. The
registry is discovered once per process: restart Hermes after editing them.

## Overridable hooks

Subclass `ProviderProfile` for non-trivial quirks:

```python
from typing import Any
from providers.base import ProviderProfile

class AcmeProfile(ProviderProfile):
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing. Runs after codex
        sanitization, before developer-role swap. Default: pass-through."""
        # Example: Qwen normalizes plain-text content to a list-of-parts
        # array and injects cache_control; Kimi rewrites tool-call JSON
        return messages

    def build_extra_body(self, *, session_id=None, **context) -> dict:
        """Provider-specific extra_body fields merged into the API call.
        Context includes: session_id, provider_preferences, model, base_url,
        reasoning_config. Default: empty dict."""
        # Example: OpenRouter's provider-preferences block,
        # Gemini's thinking_config translation.
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        """Returns (extra_body_additions, top_level_kwargs). Needed when some
        fields go top-level (Kimi's reasoning_effort, OpenRouter's verbosity for
        adaptive Anthropic models) and some go in extra_body (OpenRouter's
        reasoning dict). Default: ({}, {})."""
        return {}, {}

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0) -> list[str] | None:
        """Live catalog fetch. Default hits {models_url or base_url}/models with
        Bearer auth. Override for: custom auth (Anthropic), no REST endpoint
        (Bedrock → None), or public/unauthenticated catalogs (OpenRouter)."""
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)

    def fetch_account_usage(self, *, api_key=None, base_url=None):
        """Return AccountUsageSnapshot for /usage, or None when unavailable.

        The hook runs only when the provider has no built-in usage fetcher.
        It may raise: core catches failures and keeps /usage empty for that turn.
        """
        return None

    def create_client(self, **client_kwargs):
        """Supply your own client object instead of the shared openai.OpenAI.
        Default returns None (= use the standard client). Override when the
        wire protocol is not OpenAI-over-HTTP — e.g. an ACP subprocess shim.
        client_kwargs is what the core would have passed to openai.OpenAI
        (api_key, base_url, command, args, timeouts, headers…); accept **kwargs
        and pick what you need. A raise is logged and falls back to the
        standard client."""
        return None
```

## ProviderProfile fields and hooks

Every field and hook `providers/base.py` declares, in declaration order. Names ending in `()` are methods
to override in a subclass; the rest are dataclass fields (a few, such as `picker_catalog` and the
device-login callables, take a callable). Endpoint-keyed hooks (`client_headers`, `request_headers`,
`max_tokens_param`) receive the request's `base_url` and return `None` / `{}` for an endpoint that is
not yours; the `responses_*` and `message_carrier_*` fields and the reasoning-replay hooks apply only on
hosts the profile lists in `hostnames`. Hot-path hooks must be pure and cheap (no I/O) unless the
meaning says otherwise.

| Name | Meaning | Default |
|---|---|---|
| `name` | Canonical provider id (`model.provider`, `--provider`) | required |
| `api_mode` | Static wire protocol; `api_mode_for_model()` can vary it per model | `"chat_completions"` |
| `aliases` | Alternative names `get_provider_profile()` resolves | `()` |
| `display_name` | Human label in pickers | `""` |
| `description` | Picker subtitle | `""` |
| `signup_url` | Shown during setup | `""` |
| `hidden` | Pre-release: kept off discovery surfaces until `listed()` is true; resolution by name still works | `False` |
| `env_vars` | API-key env vars in priority order; a final `*_BASE_URL` entry is the base-URL override | `()` |
| `base_url` | Default inference endpoint | `""` |
| `models_url` | Explicit models endpoint (else `{base_url}/models`) | `""` |
| `auth_type` | `api_key` \| `oauth_device_code` \| `oauth_external` \| `external_process` \| `aws_sdk` \| plugin-defined | `"api_key"` |
| `supports_streaming` | `None` keeps transport defaults, `True` opts a process client into streaming, `False` disables it | `None` |
| `supports_health_check` | `False`: `hermes doctor` skips the `/models` probe | `True` |
| `supports_model_listing` | `False`: `fetch_models()` returns `None` without a network call | `True` |
| `auth_handler` | Provider-owned `hermes auth add\|status\|logout\|refresh <name>`; truthy return = handled | `None` |
| `refresh_credential` | Rotates a pooled OAuth row and returns the rotated fields | `None` |
| `classify_api_error` | Error classification for this provider's failures, before the built-in pipeline | `None` |
| `supports_vision` | API accepts images inside tool-result messages | `False` |
| `supports_vision_tool_messages` | API accepts list-type (multipart) tool message content | `True` |
| `supports_prompt_cache_key` | Chat Completions endpoint documents `prompt_cache_key` | `False` |
| `native_reasoning_details_type` | Namespaced `<provider>.native_assistant` reasoning carrier only this profile receives | `None` |
| `process_command` | External-process default binary | `""` |
| `process_args` | External-process default argv tail | `()` |
| `process_command_env_vars` | Env vars overriding the binary, in priority order | `()` |
| `process_args_env_var` | Env var overriding argv (shlex-split) | `""` |
| `fallback_models` | Curated picker list when the live fetch fails | `()` |
| `model_aliases` | Short name → model id; `/model <alias>` resolves here first | `{}` |
| `hostname` | Base hostname for URL → provider reverse mapping (derived from `base_url` when empty) | `""` |
| `default_headers` | Headers on every request and on the default catalog fetch | `{}` |
| `fixed_temperature` | `None` = caller's value; `OMIT_TEMPERATURE` = never send temperature | `None` |
| `default_max_tokens` | Provider-level max_tokens cap | `None` |
| `unsupported_response_formats` | `response_format` types auxiliary requests omit | `()` |
| `default_aux_model` | Cheap model for auxiliary tasks; empty = main model | `""` |
| `model_capabilities` | Per-model metadata in the `model_overrides` schema | `{}` |
| `account_card` | Accounts-tab card overrides `{name, flow, cli_command, docs_url}` (Desktop/dashboard) | `{}` |
| `account_status` | Live status for that card; subprocess- and network-free | `None` (`auth.get_auth_status(name)`) |
| `start_device_login` | In-app device-code sign-in: returns `{user_code, verification_url, expires_in, interval, state}` | `None` |
| `poll_device_login` | One poll of that login: `{status, interval?, credential?, message?}` | `None` |
| `save_device_login` | Persists the approved credential (core calls it in the session's profile scope) | `None` |
| `model_setup_flow` | The provider's own `hermes model` step (credential prompt, catalog, persist) | `None` (generic flow for `auth_type`) |
| `setup_reasoning_efforts` | Effort levels the setup picker offers for a model; `()` = no reasoning control | `None` (generic ladder) |
| `fetch_account_usage()` | `/usage` snapshot when no built-in usage fetcher owns the provider | `None` |
| `resolve_aux_model()` | Live cheap-model id for auxiliary tasks; cached, never raises | `""` (use `default_aux_model`) |
| `get_hostname()` | Provider hostname for URL detection | `hostname`, else from `base_url` |
| `prepare_messages()` | Message preprocessing after codex sanitization, before the developer-role swap | pass-through |
| `build_extra_body()` | Provider-specific `extra_body` fields | `{}` |
| `default_reasoning_config()` | Reasoning config sent when `agent.reasoning_effort` is unset | `None` |
| `build_api_kwargs_extras()` | `(extra_body_additions, top_level_kwargs)` split | `({}, {})` |
| `build_client_kwargs_extras()` | OpenAI-client kwargs defaults (explicit settings win; no I/O) | `{}` |
| `default_vision_model()` | Provider's default vision model id | `None` |
| `get_model_context_length()` | Provider-qualified context bound; explicit user overrides win | `None` |
| `get_usage_cost()` | `CostResult` from canonical usage (estimates are never `actual`) | `None` |
| `get_max_tokens()` | Per-model max_tokens cap when the user set none | `default_max_tokens` |
| `supported_reasoning_efforts()` | Effort vocabulary for a model: `None` unknown, `()` none, tuple = clamp; no I/O | `None` |
| `api_mode_for_model()` | Wire this provider serves a model on; `api_key` given only at resolution time | `None` (static `api_mode`) |
| `client_headers()` | Default headers for a client built for `base_url` (`has_images` for image requests) | `None` (static `default_headers`) |
| `request_headers()` | Per-request `extra_headers` for `base_url`; `user_initiated` on a turn's first call | `{}` |
| `max_tokens_param()` | `"max_completion_tokens"` or `"max_tokens"` for `base_url` | `None` (core heuristic) |
| `listed()` | Whether discovery surfaces may offer the provider | `True` unless `hidden` with no pool row |
| `create_client()` | Provider-specific client object (ACP shim, native SDK) | `None` (standard client) |
| `setup_status()` | External-process login state `{available, logged_in, plan, detail, login_command}` | `None` |
| `discover_models()` | External-process live picker `[{id, label, note}]` without inference | `None` (`fallback_models`) |
| `fetch_models()` | Live model ids | GET `{models_url or base_url}/models` |
| `hostnames` | Hosts (and subdomains) serving this API, so a `custom` endpoint or aux client there is recognised | `()` |
| `metadata_hostnames` | Hosts identifying this catalog for model metadata only (other credentials) | `()` |
| `models_dev_id` | models.dev provider id when it differs from `name` | `""` |
| `live_context_length()` | Context window from the live, credentialed catalog; after the persistent cache | `None` |
| `normalize_model_id()` | Provider-native id for a model (aliases, catalog repair) | `None` (generic normalization) |
| `ambient_credentials` | Credentials are ambient: never auto-detected; aux fallback skips unless configured | `False` |
| `resolve_api_key_secret()` | Provider-owned `(runtime_api_key, source)` | `None` (env/pool ladder) |
| `runtime_base_url()` | Base URL derived from the credential | `None` (env or default) |
| `prepare_pool_entry()` | Turns the selected pool row's secret into the runtime key | the secret as-is |
| `seed_pool_credentials(suppressed_sources, explicitly_configured)` | Pool rows `{source, access_token, base_url, label}` the provider seeds itself, skipping sources in `suppressed_sources` (plain values, so it works under `plugins.isolation: host`); overriding it replaces generic env seeding (return `[]` for none) | not overridden = generic env seeding |
| `remove_credential_source()` | `hermes auth remove`: `(sources_to_suppress, hints)` | `None` (generic removal) |
| `refresh_runtime_credentials()` | Re-mints `(api_key, base_url)` on `unauthorized` \| `stale` \| `auxiliary_unauthorized`; error path only | `None` |
| `is_stale_credential_error()` | A non-auth error really means the credential went stale (pure) | `False` |
| `credential_refresh_notice()` | Status line after a successful runtime refresh | `""` (generic wording) |
| `request_rejection_hint()` | User-facing lines for an endpoint's hard rejection of any route | `()` |
| `external_auth_evidence()` | Subprocess-free `(verified, source)` login evidence for an external CLI | `None` (binary resolves) |
| `missing_credentials_hint()` | One sentence appended to every "No usable credentials found" message (runtime resolution, CLI and Desktop readiness), e.g. the provider's sign-in command | `None` |
| `responses_replay_item_ids` | Replayed Responses `message` items keep their `id` | `True` |
| `responses_prompt_cache_key` | The Responses body carries `prompt_cache_key` | `True` |
| `responses_issuer_kind` | Persisted issuer stamp for encrypted reasoning minted here | `""` (URL-derived) |
| `responses_trusts_completed_status` | `False`: a reasoning-only `completed` response continues | `True` |
| `responses_native_compaction` | `False`: never send `context_management` | `True` |
| `responses_profile_reasoning` | `True`: `reasoning` = `reasoning_payload()` + summary, no `include` | `False` |
| `message_carrier_opaque` | Top-level assistant key of an opaque reasoning blob (latest streamed wins) | `""` |
| `message_carrier_text` | Top-level assistant key of readable reasoning text (streamed parts concatenate) | `""` |
| `replays_tool_call_signatures` | Tool calls keep Gemini `extra_content` signatures | `False` |
| `reasoning_payload()` | Verbatim `reasoning` object for a model (catalog-clamped) | `None` |
| `replay_message_carriers()` | Captured `message_carrier_*` values to put back on a replayed assistant message | `{}` |
| `replays_function_thought_signature()` | Replayed tool calls keep `function.thought_signature` | `False` |
| `picker_catalog` | `(force_refresh) -> list`: the provider's own model-picker catalog, tried before the generic fetch | `None` |
| `static_models` | Curated ids answered as this provider's row of the static catalog (looked up per profile home, never copied into the shared table) | `()` |
| `catalog_credential_files` | Credential files whose mtime joins every cached catalog's fingerprint | `()` |
| `probe_models()` | `(ids or None, probed_url, resolved_base_url)` for a `/models` probe this provider answers | `None` (generic HTTP probe) |
| `invalidate_catalog_cache()` | Drops in-process catalog memos (`/model --refresh`) | no-op |
| `picker_after` | Provider slug this row follows in the provider list | `""` (appended) |
| `picker_group` | `(group_id, label, description)`: folds same-group rows in interactive pickers | `()` |
| `catalog_aliases` | `aliases` also normalize in the model catalog | `False` |
| `aggregator` | Catalog re-exposes many vendors' models | `False` |
| `provider_overlay` | HermesOverlay fields for the models.dev-keyed definition | `{}` |
| `overlay_aliases` | Extra names normalizing to the `provider_overlay` definition | `()` |
| `registry_auth_type` | Auth-registry row `auth_type` when credential resolution differs from the setup flow | `""` (`auth_type`) |
| `base_url_env_var` | Env var overriding `base_url` on the registry row when not in `env_vars` | `""` |
| `registry_after` | Registry id this row follows in priority-ordered scans | `""` (appended) |

### Public helpers for a `model_setup_flow`

A provider's own `hermes model` step calls these instead of re-implementing core's config rules:

| Helper | What it does |
|---|---|
| `hermes_cli.model_setup.persist_model_selection(model, provider, *, base_url=None, api_mode=None)` | Writes the `model:` section (`provider`, then `base_url` / `api_mode` when given), scrubs inline endpoint credentials, saves `config.yaml` and deactivates the previously active OAuth provider. Returns the saved section. |
| `hermes_cli.model_setup.note_setup_failure(failure_class)` | Records why the flow ended without a pick (`"auth"`, `"no_models"`, ...) for the opt-in provider-setup metric; a no-op outside a tracked flow. |
| `agent.secret_scope.secret_source_label(name)` | `" (from Bitwarden)"`-style suffix naming the secret-source plugin that supplied env var `name`; `""` for `.env` / shell values. |

## Account usage

Model-provider plugins can provide account or plan usage to `/usage` by
overriding `ProviderProfile.fetch_account_usage`. Import and return the shared
`agent.account_usage.AccountUsageSnapshot` (with any `AccountUsageWindow`
entries); do not format output in the plugin. Returning `None`, or raising an
exception, leaves `/usage` empty just as it does for providers without usage
data. Built-in usage fetchers always take precedence, so this hook cannot
replace the account-usage behavior for a built-in provider. The hook runs under a shared
10 s deadline (`agent.account_usage.PLUGIN_USAGE_HOOK_DEADLINE_S`) on every surface; overrunning it
renders nothing for that turn rather than stalling `/usage`, so give your own HTTP calls a shorter
timeout.

The bundled `plugins/model-providers/opencode-zen/` profile implements this hook for the
OpenCode Go plan windows; every `/usage` surface (CLI `hermes usage` and `/usage`, the messaging
gateway, the TUI/Desktop usage feed) renders the snapshot through the same core formatter.

```python
from datetime import datetime, timezone

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

def fetch_account_usage(self, *, api_key=None, base_url=None):
    return AccountUsageSnapshot(
        provider=self.name,
        source="my_provider_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="Monthly", used_percent=25),),
    )
```

## External-process (ACP) providers

An agent CLI driven over stdio is not an HTTP endpoint. Set `auth_type="external_process"`, describe how to launch the binary, and supply the client with `create_client`. No core edits are needed — `hermes -m <name>`, `/model`, credential resolution, runtime resolution and the auxiliary client (compression, vision) all key on `auth_type`, not on the provider name. The `copilot-acp` provider in the [`github-copilot` labs plugin](https://github.com/NousResearch/hermes-labs-plugins/tree/main/github-copilot) is a worked out-of-tree example (its client subclasses the public `ACPClient` below).

| Field | Purpose |
|---|---|
| `process_command` | Default binary, e.g. `"kimchi"` |
| `process_args` | Default argv tail, e.g. `("--mode", "acp")` |
| `process_command_env_vars` | Env vars that override the binary, checked in order |
| `process_args_env_var` | Env var that overrides argv (shlex-split) |

The `process_command_env_vars` and `process_args_env_var` names are executable authority: Hermes' generic env writers (dashboard, MCP and connector installs, `save_env_value`) refuse them, so set them in `.env` or the shell.

The client your `create_client` returns receives `command` and `args` in `client_kwargs`. If it is already complete and async-safe, declare `HERMES_SKIP_TRANSPORT_WRAP = True` / `HERMES_SKIP_ASYNC_WRAP = True` as class attributes so the auxiliary client does not re-dispatch it through an HTTP wire adapter.

#### The public ACP client: `agent.acp_client.ACPClient`

For an agent CLI that speaks the [Agent Client Protocol](https://agentclientprotocol.com) on stdio, return an `ACPClient` (or a subclass) from `create_client` instead of writing your own shim. It is an OpenAI-client facade (`client.chat.completions.create(...)`, streaming included) that starts one short-lived ACP session per request, applies the requested model over ACP (`session/set_config_option`, else `session/set_model`), forwards Hermes' tools as a prompt-level bridge, answers `fs/read_text_file` / `fs/write_text_file` inside the session cwd with Hermes' file-safety rules, cancels permission requests, and lists the session's models with `list_models(timeout_seconds=...)`. It already declares both `HERMES_SKIP_*` attributes.

```python
from agent.acp_client import ACPClient

class MyAgentClient(ACPClient):
    default_command = "myagent"            # used only when the caller passes no command/args
    default_args = ("--acp",)
    command_env_vars = ("MYAGENT_ACP_COMMAND",)
    args_env_var = "MYAGENT_ACP_ARGS"
    marker_base_url = "acp://myagent"
    placeholder_model = "myagent-acp"      # "session default": never sent as a model selection
    label = "MyAgent ACP"                  # product name in errors and logs
    auth_required_hint = "Run `myagent login`, then retry."  # optional: added to ACP's auth-required error (-32000)

    def model_entry_disabled(self, entry):  # optional: vendor _meta marking a model unavailable
        return False

    def early_exit_error(self, stderr_text):  # optional: friendlier error for a known crash banner
        return None

class MyAgentProfile(ProviderProfile):
    def create_client(self, **client_kwargs):
        return MyAgentClient(**client_kwargs)
```

Keep the class attributes in step with the profile's `process_*` fields. Override `_handle_server_message(msg, *, process, cwd, text_parts, reasoning_parts, allow_file_requests=True)` to render extra `session/update` kinds (call `super()` for the rest). `hermes_cli.auth.resolve_external_process_provider_credentials(name)` returns the resolved `command`/`args`/`base_url` for a `fetch_models()` probe.

`agent.copilot_acp_client.CopilotACPClient` is a deprecated alias of `ACPClient` with the old GitHub Copilot launch defaults. The module keeps its whole former surface for listed catalog plugins pinned to it (`ACP_MARKER_BASE_URL`, its former private helpers, `subprocess` for `Popen` patching); the first access of any of its names warns once per process. Subclass `ACPClient` instead (set `placeholder_model` to your own placeholder id). The alias stays for at least two minor releases.

### Picker rows for non-api-key plugins

Every registered profile joins `CANONICAL_PROVIDERS` by slug (a plugin re-declaring a built-in slug such as `bedrock` is deduped, never doubled), so external-process and OAuth plugins appear in `hermes model`, `/model` and the Desktop model selector alongside the built-in providers. Visibility is gated by credentials, not by `auth_type`:

| `auth_type` | Row is listed / `authenticated` when | Model list |
|---|---|---|
| `external_process` | the binary resolves (`process_command` or one of `process_command_env_vars` is on `PATH`), or `base_url` is `acp+tcp://…` — the same structural gate `hermes auth status` reports. A resolving binary is also the sign-in evidence (`auth_verified`) the Desktop model selector's explicit-only filter uses, so the row shows there like any signed-in provider does | `fetch_models()` (your subprocess probe), else `fallback_models` |
| `oauth_external` / `oauth_device_code` | the credential pool holds a row for the slug with a live (non-expired) token — `hermes auth status <name>` and `list_available_providers().authenticated` both read the pool; an expired row with `refresh_token` and a `refresh_credential` hook reports `needs_refresh` | `fetch_models()` with the pooled token, else `fallback_models` (declare at least one) |

The catalog cache is keyed on the profile's `process_command_env_vars` / `process_args_env_var` values, so pointing `HERMES_<X>_COMMAND` at a different binary re-discovers models. Executable discovery is not a login check: an unauthenticated CLI still lists, and the subprocess reports the failure at first use.

Selecting the row in `hermes model` (and the setup wizard) runs one generic flow keyed by the profile's `auth_type`: external-process profiles are launch-checked (`resolve_external_process_provider_credentials`), OAuth profiles need a live pool row (otherwise the flow prints `hermes auth add <name>` and stops), then the merged catalog is offered and `config.model` is persisted with the profile's `base_url`/`api_mode`. No `_model_flow_*` entry in core is needed.

#### Optional external-process hooks

External-process profiles may implement `setup_status(**kwargs)` returning `{available, logged_in, plan, detail, login_command}` and `discover_models(**kwargs)` returning `[{id, label, note}]`. The generic flow gates on `logged_in` (running `login_command` inline on a TTY, printing `detail` otherwise) and, when `discover_models()` returns rows, offers them merged with `fallback_models`; `note` renders as a dim per-row annotation (`· usage credits`) and never hides a model. Keep `fetch_models()` returning the same ids so `/model` and the Desktop picker agree with setup. Both hooks must be cheap and must never perform inference; return `None` to fall back to `fallback_models`.

For interruptible non-HTTP requests, implement a class-declared `cancel(self)` method. Hermes calls it from the interrupting thread after marking the request client unusable. It must return promptly and safely stop its own transport, including cancellation racing process startup; it must not close file descriptors owned by the request thread. The request owner still calls `close()` for cleanup. Clients without this method retain the existing socket-shutdown cancellation path.

Declare `model_aliases` (`{"sonnet": "claude-sonnet-5[1m]"}`) for a catalog models.dev does not know: bare `/model <alias>` and `/model <id-prefix>` resolve inside the process provider first, and `validate_requested_model` accepts a declared id without probing `process://`.

Explicit external-process delegation retains the selected provider and its protocol when resolving the child command; an executable override alone does not change an external-process provider into ACP.

Native clients may persist private assistant replay in `reasoning_details` with a namespaced `<provider>.native_assistant` type. Declare the identical string in `ProviderProfile.native_reasoning_details_type` (default `None`). Chat Completions request sanitization forwards that carrier only to its declaring profile, including after fallback or model switching; it removes other private carriers even if their source plugin is no longer installed. Standard reasoning details such as OpenRouter's `reasoning.encrypted` remain unchanged. Filtering is request-only: durable history remains intact for returning to the original provider.

Providers may override `get_model_context_length(model)` with a qualified positive token bound, or return `None` for the existing lookup chain. Explicit configuration and endpoint-scoped overrides take precedence; the provider bound is consulted before generic caches and HTTP probes. Do not confuse a catalog maximum with an account entitlement.

For a nonstandard cost surface, `get_usage_cost(model, usage)` may return an `agent.usage_pricing.CostResult`, or `None` for normal pricing. `usage` is a `CanonicalUsage` whose `raw_usage` retains response metadata when available. Classify native list-price totals as `estimated`, never `actual` or `included`; missing invoice information is not proof of zero charges. The default hooks return `None`, preserving existing providers.

## Hook reference examples

Look at these bundled plugins for idioms:

| Plugin | Why look |
|---|---|
| `plugins/model-providers/openrouter/` | Aggregator with provider preferences, public model catalog |
| `plugins/model-providers/gemini/` | `thinking_config` translation (native + OpenAI-compat nested forms) |
| `plugins/model-providers/kimi-coding/` | `OMIT_TEMPERATURE`, `extra_body.thinking`, top-level `reasoning_effort` |
| `plugins/model-providers/qwen-oauth/` | Message normalization, `cache_control` injection, VL high-res |
| `plugins/model-providers/nous/` | Attribution tags, "omit reasoning when disabled" |
| `plugins/model-providers/custom/` | Ollama `num_ctx` + `think: false` quirks |
| `plugins/model-providers/bedrock/` | `api_mode="bedrock_converse"`, `fetch_models` returns None (no REST endpoint) |

## User overrides — replace a built-in without editing the repo

Say you want to point `gmi` at your private staging endpoint for testing. Create `~/.hermes/plugins/model-providers/gmi/__init__.py`:

```python
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="gmi",
    aliases=("gmi-cloud", "gmicloud"),
    env_vars=("GMI_API_KEY",),
    base_url="https://gmi-staging.internal.example.com/v1",
    auth_type="api_key",
    default_aux_model="google/gemini-3.1-flash-lite-preview",
))
```

In a fresh Hermes process, `get_provider_profile("gmi").base_url` returns the staging URL. No repo patch, no rebuild. Because user plugins are discovered after bundled ones, the user `register_provider()` call wins.

The override also reaches the runtime. Built-in providers have a row in `hermes_cli.auth.PROVIDER_REGISTRY` (the table `resolve_runtime_provider()` reads its endpoint and env vars from); a `$HERMES_HOME` plugin re-registering that name rewrites the row's profile-derived fields, so inference goes to the staging URL, not the bundled one:

| Profile field | Registry row field | When |
|---|---|---|
| `base_url` | `inference_base_url` | profile sets a non-empty `base_url` |
| `env_vars` (non-URL entries) | `api_key_env_vars` | api-key row and profile sets `env_vars` |
| `env_vars` (final `*_BASE_URL` / `*_URL` entry) | `base_url_env_var` | profile declares one; otherwise the built-in env var (e.g. `GMI_BASE_URL`) stays |

Only a **user** plugin (`$HERMES_HOME/plugins/model-providers/` or an installed `kind: model-provider` plugin) triggers this; a bundled profile never rewrites a built-in row, and `kimi-coding`, `kimi-coding-cn` and `zai` keep their bespoke credential resolution. A field the profile leaves empty keeps the built-in value. A `*_BASE_URL` env var still wins over both.

## api_mode selection

Four built-in values are recognized (`chat_completions`, `codex_responses`, `anthropic_messages`, `bedrock_converse`), plus any mode a plugin registers itself. Hermes picks one based on:

1. User explicit override (`config.yaml` `model.api_mode` when set)
2. OpenCode's per-model dispatch (`opencode_model_api_mode` for Zen and Go)
3. URL auto-detection — `/anthropic` suffix → `anthropic_messages`, `api.openai.com` → `codex_responses`, `api.x.ai` → `codex_responses`, `/coding` on Kimi domains → `chat_completions`
4. **Profile `api_mode`** as a fallback when URL detection finds nothing
5. Default `chat_completions`

Set `profile.api_mode` to match the default your provider ships — it acts as a hint. User URL overrides still win.

### Shipping your own wire dialect

A plugin that speaks a protocol none of the built-in transports cover registers one and names it in the profile:

```python
from agent.transports import register_transport
from agent.transports.chat_completions import ChatCompletionsTransport

class MyDialectTransport(ChatCompletionsTransport):
    api_mode = "mydialect"
    # override convert_messages / build_kwargs / normalize_response as needed

register_transport("mydialect", MyDialectTransport)
register_provider(ProviderProfile(name="myprovider", api_mode="mydialect", ...))
```

Every `api_mode` gate (`determine_api_mode`, runtime resolution, agent construction, delegation) accepts a mode iff the transport registry knows it; a profile naming a mode nobody registered still degrades to `chat_completions`.

## Auth types

| `auth_type` | Meaning | Who uses it |
|---|---|---|
| `api_key` | Single env var carries a static API key | Most providers |
| `oauth_device_code` | Device-code OAuth flow | Nous Portal; out-of-tree plugins via `auth_handler` |
| `oauth_external` | User signs in elsewhere, tokens land in `auth.json` | Anthropic OAuth, MiniMax OAuth, Qwen Portal, Nous Portal |
| *plugin-defined* | Any other value: login and refresh are provider-owned (`auth_handler`, `refresh_credential`, the credential hooks below); `registry_auth_type` names the auth-registry row when it differs | e.g. `copilot` in the [`github-copilot` labs plugin](https://github.com/NousResearch/hermes-labs-plugins/tree/main/github-copilot) |
| `aws_sdk` | AWS SDK credential chain (IAM role, profile, env) | `bedrock` plugin only |
| `external_process` | Auth handled by a subprocess the agent spawns (see [External-process providers](#external-process-acp-providers)) | out-of-tree ACP plugins (e.g. `copilot-acp` in the `github-copilot` labs plugin) |

Every profile is mirrored into Hermes' auth registry under the `auth_type` it declares (two exclusions: an `api_key` profile with empty `env_vars`, and the aggregator/user-supplied slugs `openrouter`/`custom` plus the bespoke-refresh built-ins `kimi-coding`/`kimi-coding-cn`/`zai`), so `hermes auth`,
`--provider <name>` and runtime resolution accept it whatever its shape. What differs is who performs the
login: `api_key` profiles get the built-in key prompt / env-var resolution; every other `auth_type` is
**provider-owned** — the plugin ships the two hooks below, and a non-api-key profile without an
`auth_handler` makes `hermes auth add <name>` fail with a clear "ships no auth_handler" error instead of
silently doing nothing.

## Provider-owned auth (`auth_handler`, `refresh_credential`)

`auth_type` describes *what kind* of credential a provider needs; `auth_handler` is how the plugin
**acquires** it — its own device-code / OIDC / IdC flow inside the existing `hermes auth` command family
(model-provider manifests are skipped by the generic command-plugin loader, so `register(ctx)` is not the
way to add commands). `refresh_credential` is how the credential pool **rotates** a pooled token the plugin
stored.

```python
import uuid
from providers import register_provider
from providers.base import ProviderProfile


def example_auth(action: str, args) -> bool:
    """action: "add" | "status" | "logout" | "refresh"; args: parsed CLI namespace."""
    if action == "add":
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        tokens = run_device_code_flow()                      # provider-specific
        load_pool("example-oauth").add_entry(PooledCredential(
            provider="example-oauth", id=uuid.uuid4().hex[:6], label=tokens["account"],
            auth_type=AUTH_TYPE_OAUTH, priority=0, source="manual:example_device",
            access_token=tokens["access_token"], refresh_token=tokens["refresh_token"],
            extra={"tenant": tokens["tenant"]}))             # any extra keys round-trip through auth.json
        print("Signed in to Example.")
        return True
    if action == "status":
        print("example-oauth: " + ("logged in" if load_pool("example-oauth").entries() else "logged out"))
        return True
    return False   # decline → this action stays with the built-in credential-pool handling


def example_refresh(entry):
    """Called by the credential pool with the pooled row; return the rotated values, None, or raise."""
    tokens = post_refresh(entry.refresh_token)          # the raw token-endpoint response is fine as-is
    return {"access_token": tokens["access_token"], "refresh_token": tokens["refresh_token"],
            "expires_at_ms": tokens["expires_at_ms"], "expires_in": tokens["expires_in"]}


register_provider(ProviderProfile(
    name="example-oauth", auth_type="oauth_external", base_url="https://api.example.com/v1",
    auth_handler=example_auth, refresh_credential=example_refresh))
```

| Contract | |
|---|---|
| `auth_handler(action, args)` | `args` is the parsed `hermes auth` namespace for CLI actions; the interactive setup picker passes a minimal namespace carrying only `provider`, so read options with `getattr(args, name, None)`. Truthy = handled (Hermes prints nothing more, exit 0); falsy = fall back to the built-in path **for that action**. An exception becomes `SystemExit("<provider> auth handler failed for `&lt;action&gt;`: …")`. |
| `refresh_credential(entry)` | Receives the `PooledCredential`; returns a mapping of rotated values or `None`. Keys that are `PooledCredential` fields (`access_token`, `refresh_token`, `expires_at_ms`, …) replace the row's fields; every other key (`expires_in`, `token_type`, `scope` — the raw token-endpoint shape) lands in `entry.extra` and round-trips through `auth.json`. Returning `None`/an empty mapping means the plugin could not rotate: the row is benched exactly like a failed refresh request (never reported as refreshed, so a dead bearer is not replayed). Its presence is what makes the provider *refreshable* — `hermes auth refresh <name>` and the main-loop 401 recovery call it through the pool with no core name list involved; the auxiliary client's 401 recovery reaches it only for pooled rows it already treats as recoverable (api-key rows and the built-in OAuth routes). |
| Refresh failures | Raise `hermes_cli.auth_constants.AuthError(..., relogin_required=True)` (or with `code` `invalid_grant` / `invalid_token` / `refresh_token_reused`) when the grant is dead: the row goes **DEAD**, leaves rotation and Hermes logs a WARNING naming `hermes auth add <name>`. Any other exception (network, 429, 5xx) is transient — the row is benched for one cooldown and retried. |
| Concurrency | The hook runs under the shared `auth.json` lock. Before calling it the pool re-reads the row; if another Hermes process (gateway + CLI, two profiles) already rotated the pair, that pair is adopted and your hook is **not** called — safe for single-use refresh tokens. After the hook returns, the rotated row is written through to `auth.json`. |
| No hooks | `api_key` profiles behave exactly as before. Any other `auth_type` without `auth_handler` fails loud on `hermes auth add`. |

`hermes auth add|status|logout|refresh <provider>` consults the handler **first** — before the built-in
credential-pool flow. Registering the same name twice is last-writer-wins, so a user plugin can replace a
bundled provider's flow.

Hermes passes the parsed namespace, not provider-declared flags: ask for provider-specific values
interactively (or read your own config/env). Rows the plugin stores in the pool are its own — extra keys
survive `load → save → load`, and Hermes passes no secrets beyond that pooled row to `refresh_credential`.

### Declarative OAuth (PKCE) for plugins

A provider whose IdP speaks standard OAuth 2.0 Authorization Code + PKCE does not need to write the
hooks above by hand: declare the endpoints in `OAuthPKCEConfig` and let the two factories build them.

```python
from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig, pkce_auth_handler, pkce_refresh_credential
from providers import register_provider
from providers.base import ProviderProfile

cfg = OAuthPKCEConfig(
    client_id="hermes-public-client",                       # public client — no secret, PKCE is the proof
    authorize_url="https://auth.example.com/oauth/authorize",
    token_url="https://auth.example.com/oauth/token",
    scopes=("inference", "offline_access"),
    redirect_port=0,                                        # 0 = OS-assigned; pin it if the IdP allowlists the URI
)

register_provider(ProviderProfile(
    name="example-pkce", auth_type="oauth_external", base_url="https://api.example.com/v1",
    auth_handler=pkce_auth_handler(cfg), refresh_credential=pkce_refresh_credential(cfg)))
```

Hermes then owns the whole lifecycle: `hermes auth add example-pkce [--no-browser]` opens the browser (or
prints the URL, with the SSH-tunnel hint on a remote box), listens on `http://127.0.0.1:<port>/callback`,
checks the CSRF `state`, exchanges the code with S256 PKCE and stores the grant as a pooled `oauth`
credential (`source: manual:loopback_pkce`, `expires_at_ms`, `refresh_token`); `auth status` reports
logged in / expired; `auth logout` removes the rows; `auth refresh` and the 401 recovery paths rotate via
the `refresh_token` grant, re-reading `auth.json` under the auth lock first so a peer's rotation is adopted
instead of spending a single-use refresh token twice.

Security boundary (enforced before any request, on login and refresh alike): both endpoints must be
`https://` (plain `http://` is accepted only for a loopback-literal host — a local development IdP);
the `token_url` host must be the `authorize_url` host or a subdomain of it (or listed in `allowed_hosts`);
the listener binds the literal `127.0.0.1`; tokens, `state` and the PKCE verifier are never logged.
Optional fields: `audience`, `extra_authorize_params`, `extra_token_params`, `redirect_path`,
`timeout_seconds`, `label`.

A confidential client whose secret lives behind your own broker sets `token_request` instead of
`token_url`: Hermes calls it with the grant fields (`grant_type`, `code` / `refresh_token`,
`redirect_uri`, `code_verifier`) and stores whatever token-endpoint JSON it returns, keeping the old
`refresh_token` when the response omits one. Raise `AuthError(..., relogin_required=True)` for a grant
the broker reports dead. Rows rotate ahead of their stored `expires_at_ms`, so opaque (non-JWT) access
tokens refresh on time; the auxiliary client (compression, titles) leases the same pooled row.

## Recovery and error classification

A `kind: model-provider` plugin is loaded by provider discovery, **not** by the generic plugin manager, so
the `transform_api_error_classification` plugin hook is not reachable from it without shipping a second
plugin component. The profile carries the equivalent seam instead:

```python
def classify(error, *, status_code, error_code, message, body, model):
    # A vendor-specific 403 that is a spent plan, not a bad credential.
    if status_code == 403 and error_code == "quota_exhausted":
        return {"reason": "billing", "retryable": False, "should_rotate_credential": True, "should_fallback": True}
    return None  # decline → built-in classification


register_provider(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                                  base_url="https://api.example.com/v1",
                                  refresh_credential=example_refresh, classify_api_error=classify))
```

| Contract | |
|---|---|
| `classify_api_error(error, *, status_code, error_code, message, body, model)` | Consulted by `agent.error_classifier.classify_api_error` for failures of **this provider only**, after any generic `transform_api_error_classification` hooks and before the built-in pipeline. `message` is the lower-cased error text, `body` the parsed JSON body (may be empty). Return `{"reason": <FailoverReason name>}` plus optional `retryable` / `should_compress` / `should_rotate_credential` / `should_fallback` / `error_context` to override (for terminal reasons — billing, auth, model_not_found … — `should_fallback: True` implies `retryable: False` unless you set it, because the fallback chain only runs for non-retryable verdicts; rate-limit reasons keep the built-in retry-then-fallback shape); `None` (or an unknown reason) leaves the built-in verdict. Exceptions are swallowed and logged at DEBUG. The verdict drives the same recovery as for built-ins — e.g. `billing` benches the credential for the billing TTL instead of the transient 403 cooldown. |
| 401 on a plugin credential | Handled by the credential pool, no core edit: the failing pooled row is refreshed through `refresh_credential` once per attempt (capped at two refreshes per row per session), the client is rebuilt with the rotated token and the request retried. A `None`/empty return or an exception benches the row — the request then rotates or falls to the generic "sign in again: `hermes auth add <name>`" copy, never to a built-in provider's guidance. |
| Auxiliary calls | An `oauth_external` / `oauth_device_code` plugin that ships `create_client` serves auxiliary tasks on its pooled row (an expiring row is rotated first); auxiliary 401s take the same pool refresh (`try_refresh_current` → `refresh_credential`). |

Recovery that remains name-keyed in core is behaviour with no safe generic shape (a provider-specific
token store to re-sync, a plan-tier entitlement wall, a single-use refresh-token quarantine). A plugin
that needs one of those owns it inside `refresh_credential` / `classify_api_error`.

## Discovery timing

Provider discovery is **lazy** — triggered by the first `get_provider_profile()` or `list_providers()` call in the process. In practice this happens early at startup (`auth.py` module load extends `PROVIDER_REGISTRY` eagerly). If you need to verify your plugin loaded, run:

```bash
hermes doctor
```

— a successful `auth_type="api_key"` profile appears under the Provider Connectivity section with a `/models` probe.

For programmatic inspection:

```python
from providers import list_providers
for p in list_providers():
    print(p.name, p.base_url, p.api_mode)
```

## Testing your plugin

Point `HERMES_HOME` at a temp directory so you don't pollute your real config:

```bash
export HERMES_HOME=$HOME/.hermes/cache/scratch/hermes-plugin-test
mkdir -p $HERMES_HOME/plugins/model-providers/my-provider
cat > $HERMES_HOME/plugins/model-providers/my-provider/__init__.py <<'EOF'
from providers import register_provider
from providers.base import ProviderProfile
register_provider(ProviderProfile(
    name="my-provider",
    env_vars=("MY_API_KEY",),
    base_url="https://api.my-provider.example.com/v1",
    auth_type="api_key",
))
EOF

export MY_API_KEY=your-test-key
hermes -z "hello" --provider my-provider -m some-model
```

## General PluginManager integration

The general `PluginManager` (the thing `hermes plugins` operates on) **sees** model-provider plugins but does not import them — `providers/__init__.py` owns their lifecycle. The manager records the manifest for introspection and categorizes by `kind: model-provider`. When you drop an unlabeled user plugin into `$HERMES_HOME/plugins/` that happens to call `register_provider` with a `ProviderProfile`, the manager auto-coerces it to `kind: model-provider` via a source-text heuristic — so the plugin still routes correctly even without `plugin.yaml`.

## Distribute via pip

Model providers can ship as a pip package. Expose an entry point in the
`hermes_agent.plugins` group in your `pyproject.toml`:

```toml
[project.entry-points."hermes_agent.plugins"]
acme-inference = "acme_hermes_plugin:register"
```

The target may be either:

- a **callable** (`module:func`) — invoked with no arguments; it should call
  `register_provider(profile)`, or
- a **bare module** (`module`) — imported for its module-level
  `register_provider(...)` side effect, mirroring the directory-plugin
  `__init__.py` contract.

`providers/__init__.py` discovers these entry points itself — the general
`PluginManager` never invokes provider registration for pip packages (its
entry-point path targets `register(ctx)`-style general plugins, gated by
`plugins.enabled`), so the provider registry does its own scan. Two rules
apply:

- **Opt-in required.** The same `plugins.enabled` allow-list (and
  `plugins.disabled` deny-list) from `config.yaml` governs this scan. A pip
  package is never imported just because it is installed — users must add the
  entry-point name to `plugins.enabled`:

  ```yaml
  plugins:
    enabled:
      - acme-inference
  ```

- **Lowest precedence.** Entry-point plugins are discovered **before**
  filesystem plugins: because `register_provider()` is last-writer-wins, a
  bundled or `$HERMES_HOME` profile of the same name always overrides a
  pip-installed one. A pip package can add a genuinely new provider, but
  cannot silently hijack a first-party provider name.

Targets that require arguments (a general plugin's `register(ctx)`) are
skipped by the provider scan — they belong to the `PluginManager`. A broken
entry point is isolated — it is logged at warning level and skipped, and never
blocks discovery of the other providers.

See [Building a Hermes Plugin](./plugins/index.md#distribute-via-pip) for the full entry-points setup.

## Related pages

- [Provider Runtime](./provider-runtime.md) — resolution precedence + where each layer reads the profile
- [Adding Providers](./adding-providers.md) — end-to-end checklist for new inference backends (covers both the fast plugin path and the full CLI/auth integration)
- [Memory Provider Plugins](./memory-provider-plugin.md)
- [Context Engine Plugins](./context-engine-plugin.md)
- [Building a Hermes Plugin](./plugins/index.md) — general plugin authoring
