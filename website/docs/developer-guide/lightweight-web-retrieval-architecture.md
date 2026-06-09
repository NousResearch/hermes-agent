---
sidebar_position: 13
title: "Lightweight-First Web Retrieval Architecture"
description: "Make `web_search` + `web_extract` the default retrieval path and reserve browser tools for explicit or policy-qualified fallback."
---

# Lightweight-First Web Retrieval Architecture

Status: Proposed

## Decision summary

Hermes should treat web access as three planes:

1. Discovery: `web_search`
2. Fetch/extract: `web_extract`
3. Browser: browser tools only for explicit or policy-qualified fallback

This is primarily a routing, policy, and documentation change. It is not a new public tool API. The public retrieval surface remains `web_search` and `web_extract`.

Default policy:
- `web.strategy: lightweight-first`
- `web.browser_fallback: manual`

`manual` is the default because it is deterministic, preserves cost and latency expectations, and avoids silent escalation into browser execution. `auto` remains supported as an opt-in.

## Goals

- Make search-first, extract-second the default retrieval path for personas.
- Keep browser use exceptional, not peer-to-peer with lightweight retrieval.
- Preserve existing tool names and existing `web.backend` configs.
- Support hosted and self-hosted backends through the existing provider model.
- Keep SSRF and internal-network protections mandatory on all non-browser fetch paths.

## Non-goals

- Replacing the browser toolset
- Shipping login automation or CAPTCHA bypass
- Standardizing provider-specific ranking or extraction quality
- Adding a new model-facing retrieval tool
- Shipping a first-party in-repo extract service in this change

## Implementation scope

This proposal changes the following Hermes surfaces:

- `tools/web_tools.py`
  - backend resolution
  - fallback policy evaluation
  - normalized extraction metrics
  - normalized response metadata
  - fallback reason codes
  - telemetry emission hooks
- `tools/website_policy.py`
  - policy deny/block decisions for non-browser fetch paths
- `plugins/web/*/provider.py`
  - provider capability reporting and optional normalized metadata
- `toolsets.py`
  - tool descriptions for `web` and browser toolsets
- `agent/prompt_builder.py`
  - system guidance that teaches lightweight-first behavior
- `hermes_cli/nous_subscription.py`
  - config/status presentation for split search vs extract backends if surfaced in setup/status UX
- `website/docs/user-guide/features/web-search.md`
- `website/docs/developer-guide/web-search-provider-plugin.md`
- tests:
  - `tests/tools/test_web_tools_config.py`
  - `tests/tools/test_web_providers*.py`
  - `tests/plugins/web/test_web_search_provider_plugins.py`
  - `tests/tools/test_website_policy.py`
  - `tests/integration/test_web_tools.py`

Out of scope for this PR/design:
- an in-repo self-hosted extract microservice
- MCP as a second public retrieval API
- browser-tool schema changes

## Normative behavior

### Routing

1. `web_search` resolves backend in this order:
   - `web.search_backend`
   - `web.backend`
   - existing env-based autodetect fallback

2. `web_extract` resolves backend in this order:
   - `web.extract_backend`
   - `web.backend`
   - existing env-based autodetect fallback

3. Browser tools are never selected by a provider adapter.

### Fallback decision owner and precedence

Fallback eligibility is owned by Hermes core runtime in `tools/web_tools.py`.

Precedence:

1. Policy deny in `tools/website_policy.py`
2. Explicit user request for browser behavior
3. Normal `web_search` / `web_extract` execution
4. Core fallback evaluation after `web_extract`
5. Browser execution only if `web.browser_fallback == auto`

Provider adapters may report structured failure or capability signals, but they do not invoke browser tools directly. Persona prompts may encourage lightweight-first usage, but prompts do not decide fallback policy.

Policy-denied targets MUST NOT be overridden by explicit browser requests, provider hints, or automatic browser fallback.

### Browser fallback modes

- `off`: never recommend or invoke browser automatically
- `manual` (default): return structured fallback reason, but do not invoke browser
- `auto`: Hermes core may perform one browser retrieval attempt after a qualifying extract failure

### Normalization point for fallback metrics

Hermes core computes fallback metrics in `tools/web_tools.py` after provider output is normalized into a per-URL extract result. Providers may supply helper metadata, but the final metrics used for fallback thresholds are computed by Hermes core from normalized fields.

For fallback metrics, Hermes core derives these fields per URL:

- `normalized_text`: direct extracted human-readable text intended to represent the fetched page contents
- `normalized_paragraphs`: paragraph splits derived from `normalized_text`
- `content_origin`: one of `direct_extract`, `provider_summary`, `llm_summary`, `mixed`, or `unknown`
- `metrics_eligible`: boolean set by Hermes core

`metrics_eligible` MUST be `true` only when all of the following hold:

- `content_origin == direct_extract`
- `normalized_text` comes from provider-extracted page text, markdownified page text, or equivalent direct extraction output
- Hermes has enough direct text to measure character and paragraph thresholds

`metrics_eligible` MUST be `false` when content is summarized, LLM-compressed, rewritten for brevity, or mixed with direct text in a way that prevents a clean direct-text measurement. Summarized or LLM-compressed content MUST NOT count toward `extract_empty`, `extract_low_signal`, or `js_shell_detected` thresholds.

When `metrics_eligible` is `false`, Hermes core may still recommend or attempt fallback only from deterministic non-threshold signals such as `auth_required`, `provider_unsupported`, `user_requested_browser`, transport failure classifications, or explicit provider flags that map to a normative fallback reason code.

### Measurable fallback triggers

A browser fallback is eligible only for these reason codes:

- `extract_empty`
  - `metrics_eligible == true`
  - extract request succeeded or returned HTML
  - and `normalized_text` is under 200 non-whitespace characters

- `extract_low_signal`
  - `metrics_eligible == true`
  - `normalized_text` is under 500 non-whitespace characters
  - and raw response body size is over 5 KB
  - and no `normalized_paragraphs` entry exceeds 120 characters

- `js_shell_detected`
  - `metrics_eligible == true`
  - HTML response
  - `normalized_text` under 500 non-whitespace characters
  - and page contains at least one JS-shell marker such as `id="root"`, `id="app"`, `__NEXT_DATA__`, `ng-app`, or a large script-heavy body with minimal readable text

- `auth_required`
  - HTTP 401/403
  - or redirect to a login/auth/session path
  - or provider returns explicit auth-required classification

- `provider_unsupported`
  - provider explicitly reports unsupported content type, unsupported page class, or unsupported extraction mode

- `user_requested_browser`
  - user explicitly asks for browser interaction, screenshotting, or JS-rendered verification

Not eligible for browser fallback:
- policy-blocked URLs
- private/internal address targets
- successful direct extract with at least 500 visible characters
- plain documents already returned as usable text
- repeated browser escalation after one failed browser attempt

## Normalized interfaces

The following is a normalized interface target, not a breaking API rewrite.

### Required fields

These fields are normative for Hermes-level tool results:

For `web_search` results:
- `title`
- `url`

For `web_extract` results:
- `url`
- `content` or `error`

For fallback-aware extract results:
- `fallback_reason` when fallback is recommended or attempted

### Optional best-effort fields

These fields are illustrative and may be omitted by older providers:

- `source`
- `id`
- `final_url`
- `content_type`
- `fetch_status`
- `canonical_url`
- `raw_content`
- `provider_metadata`
- `content_origin`
- `body_bytes`
- `normalized_text_chars`

If present, Hermes should pass them through. If absent, Hermes must not fail the request.

### `raw_content` contract

`raw_content` is an optional provider-to-core evidence field for extract results. It means a verbatim or lightly sanitized representation of the fetched response body after redirects and decoding, before Hermes-level summarization.

Rules:

- `raw_content` MUST NOT contain provider-generated summaries, LLM-compressed text, or browser-rendered postprocessing presented as the original body.
- `raw_content` MAY be truncated for body-size limits, memory pressure, or provider quotas.
- `raw_content` MAY be sanitized to remove obviously unsafe binary fragments or transport artifacts.
- `raw_content` MAY be omitted entirely even when the provider had access to the full body.
- When present and truncated, providers SHOULD expose truncation state in `provider_metadata` or a normalized boolean such as `raw_content_truncated`.
- Hermes core MUST treat `raw_content` as optional evidence, not as a guaranteed full-fidelity archive of the response.

### Important rule on summarization

If `content` is LLM-compressed or summarized rather than direct extracted text, Hermes must mark that explicitly in metadata. Hidden summarization is not acceptable for evidence-sensitive use cases.

## Telemetry contract

Telemetry emission for retrieval and fallback is owned by Hermes core in `tools/web_tools.py`. Providers do not emit canonical Hermes retrieval events directly. Providers may supply metadata and classifications that Hermes core copies into event fields.

### Required core event fields

For every `web_search` or `web_extract` completion event, Hermes core MUST emit:

- `event_name`: `web_search_completed`, `web_extract_completed`, `web_fallback_recommended`, `web_fallback_attempted`, or `web_policy_blocked`
- `tool_name`: `web_search` or `web_extract`
- `provider`: resolved backend name
- `url_count`: number of URLs requested or processed
- `duration_ms`: end-to-end Hermes-measured duration for the tool call
- `outcome`: `success`, `partial`, `error`, or `blocked`
- `policy_blocked`: boolean
- `fallback_mode`: `off`, `manual`, or `auto`

For extract-related fallback events, Hermes core MUST additionally emit:

- `fallback_reason`: one normative reason code or `null`
- `fallback_eligible`: boolean
- `browser_attempted`: boolean
- `content_origin`: normalized origin classification or `unknown`
- `metrics_eligible`: boolean

### Optional event fields

Hermes core SHOULD include these when available:

- `search_backend_config_source` or `extract_backend_config_source`
- `content_type`
- `fetch_status`
- `final_url`
- `body_bytes`
- `normalized_text_chars`
- `longest_paragraph_chars`
- `provider_reason`
- `provider_metadata`
- `raw_content_present`
- `raw_content_truncated`
- `policy_reason`

### Provider-supplied inputs to telemetry

Providers may return optional inputs such as:

- `content_origin`
- `provider_reason`
- `body_bytes`
- `raw_content_truncated`
- provider-native extraction diagnostics in `provider_metadata`

Hermes core is responsible for:

- normalizing those inputs
- dropping unknown or unsafe fields
- computing canonical event names and required fields
- emitting the final telemetry event

## Self-hosted design scope

Self-hosted retrieval is supported through the existing provider architecture:

- search: SearXNG or equivalent
- extract: external HTTP extract service behind a Hermes provider adapter

This design does not make Hermes own a first-party in-repo extract service. The service is explicitly out of scope for this change. What is in scope is the adapter contract and policy expectations for any self-hosted extract backend:
- outbound allow/block policy
- SSRF checks
- redirect revalidation
- request timeout and body-size caps
- no credential forwarding from Hermes by default

## Compatibility and migration

- `web_search` and `web_extract` names remain unchanged.
- `web.backend` remains supported as the shared fallback config.
- Existing providers continue to work without new optional metadata.
- Missing optional fields degrade to `null` / absent; they are not fatal.
- Provider plugins may adopt normalized metadata incrementally.
- `auto` fallback should only rely on deterministic signals from Hermes core or explicit provider reason codes. Providers that do not emit optional metadata still remain usable under `manual`.

## Security requirements

These are mandatory for all non-browser fetch paths, including self-hosted backends:

- block RFC1918, loopback, link-local, CGNAT, metadata endpoints, reserved ranges
- allow only HTTP(S)
- fail closed on DNS resolution failure
- revalidate every redirect hop
- enforce policy checks before provider call where possible
- do not forward browser/session credentials into lightweight HTTP fetches
- enforce timeouts and body-size caps in Hermes and, when applicable, in the backend service

## Rollout plan

### Phase 1: core policy and docs
Repo deliverables:
- update tool descriptions in `toolsets.py`
- implement `web.strategy` and `web.browser_fallback` handling in `tools/web_tools.py`
- add fallback reason codes and `manual` default
- add Hermes-core normalization of fallback metrics
- update lightweight-first guidance in `agent/prompt_builder.py`
- update:
  - `website/docs/user-guide/features/web-search.md`
  - `website/docs/developer-guide/web-search-provider-plugin.md`
- add/extend tests in:
  - `tests/tools/test_web_tools_config.py`
  - `tests/tools/test_website_policy.py`

### Phase 2: provider normalization
Repo deliverables:
- document required vs optional normalized fields
- update `plugins/web/*/provider.py` to expose optional metadata where available
- add provider compatibility coverage in:
  - `tests/tools/test_web_providers*.py`
  - `tests/plugins/web/test_web_search_provider_plugins.py`

### Phase 3: telemetry and integration coverage
Repo deliverables:
- emit structured retrieval/fallback events from `tools/web_tools.py`
- record backend, latency, fallback reason, policy-block outcome, and normalization eligibility
- add integration coverage in `tests/integration/test_web_tools.py`

## Recommendation

Adopt this design with:
- `web.strategy: lightweight-first`
- `web.browser_fallback: manual` by default
- browser fallback owned by Hermes core, not providers
- normalized metadata treated as incremental, best-effort compatibility fields
- self-hosted extract service kept out of scope for this PR, with adapter support documented instead

This keeps the change repo-sized, testable, and compatible with current Hermes provider architecture while making browser usage measurably rarer and more intentional.
