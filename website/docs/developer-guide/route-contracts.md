---
sidebar_position: 5
title: "Route Contracts"
description: "Content-safe runtime route proofs for provider/model/auth/cost invariants"
---

# Route Contracts

Hermes route contracts answer a concrete operator question: **what route am I on, and is it allowed before work starts?**

The contract layer does not choose providers. Runtime resolution still lives in `hermes_cli/runtime_provider.py`. Route contracts take the resolved provider/model/auth/runtime shape and produce a small, content-safe proof object that can be surfaced in health, trace, dashboard, and TUI surfaces without leaking credentials.

## Covered surfaces

Tier 2 treats these as separate routes, even when they share implementation plumbing:

| Surface | Source | Why it is separate |
|---|---|---|
| `primary` / `cli` | foreground chat agent | Baseline model/provider/runtime proof |
| `delegation` | `tools/delegate_tool.py` child agents | Children can override provider/model/toolsets and must not silently inherit the wrong API mode |
| `cron` | `cron/scheduler.py` autonomous jobs | Fresh sessions can run unattended, so forbidden fallback must fail before work starts |
| `tui` | `tui_gateway/server.py` | The Ink/Dashboard chat session has its own startup resolver and session info event |
| `gateway` | `gateway/run.py` platform sessions | Messaging platforms can apply per-session `/model`, reasoning, service-tier, and fallback settings |
| `dashboard` | `hermes_cli/web_server.py` `/api/status` | The operator status panel needs route evidence without exposing secrets |

## Proof shape

`hermes_cli.route_contracts.build_agent_route_proof(...)` returns a redacted dictionary with fields including:

- `surface`
- `provider`
- `model`
- `api_mode`
- `runtime`
- `setup_mode` (`hermes_recommended_codex_oauth`, `codex_app_server_opt_in`, `metered_api`, `local_endpoint`, etc.)
- `route_owner` (`hermes`, `hermes_outer_codex_inner`, or `external_cli`)
- `requires_external_cli`
- `base_url_host`
- `base_url_path_hint` (only known-safe API route segments such as `/backend-api/codex` or `/api/v1`; arbitrary proxy paths become `/<redacted-path>`)
- `credential_present`
- `credential_kind`
- `auth_surface`
- `cost_surface`
- `reasoning_effort`
- `service_tier`
- `fallback_chain_count`
- `contract.status`
- `contract.violations[]`

It intentionally does **not** include:

- raw API keys
- OAuth bearer tokens
- query parameters
- connection strings
- raw prompt/user content

## Hermes-recommended Codex baseline

For Carson's hardening plan, the canonical recommended Codex route is:

```text
provider=openai-codex
api_mode=codex_responses
model=gpt-5.5
model.openai_runtime=auto
auth_surface=oauth
cost_surface=subscription
route_owner=hermes
requires_external_cli=false
```

This keeps Hermes as the outer runtime and uses the ChatGPT/Codex OAuth subscription surface without spawning an external Codex CLI app-server. The app-server path still exists, but it is now modeled as `setup_mode = codex_app_server_opt_in` and must be documented as an explicit exception when used.

The point is not to ban all API-backed routes globally. It is to make route intent auditable: if work is supposed to run on the subscription/OAuth path, Hermes should fail or surface `attention` before silently falling back to per-token API billing or an external runtime.

## Seven-tier route impact plan

`hermes_cli.route_contracts.build_route_hardening_plan(route_proof)` maps the active route proof onto the broad B hardening taxonomy:

| Tier | Name | Route-plan responsibility |
|---|---|---|
| 1 | Reliability / source control | Route selection must not mutate git state or hide local checkpoint state. |
| 2 | Route invariants | The active provider/model/auth/runtime/cost contract is either `ok`, `attention`, or `blocked`. |
| 3 | Trace / replay | Persist route-proof metadata in traces and classify blocked route contracts as `route_contract` failures. |
| 4 | Context hygiene | Keep route proof in sidecar/control-plane metadata, not model-visible prompt content. |
| 5 | Skill lifecycle | Native Hermes runtime keeps skill tools available; app-server exceptions must prove skill/MCP bridging before relying on skills. |
| 6 | Autonomous loops | Delegation, cron, TUI, dashboard, and gateway routes must be proved independently. |
| 7 | Security | Forbid Codex OAuth/subscription routes from silently degrading into OpenAI Platform API-key fallback; expose only redacted metadata. |

Dashboard/control-plane surfaces:

- `/api/status` includes both `route_proof` and `route_plan`.
- `/api/harness/route-plan` returns the metadata-only seven-tier plan directly.

`route_plan` is deliberately a plan/check surface, not another resolver. If a route choice creates risk for a tier, the plan should say so as `attention`/`blocked` plus a required action, not silently rewrite runtime behavior.

## Hard contract: Codex app-server auth

`api_mode = codex_app_server` is the Codex subprocess/runtime route. It must not be backed by an OpenAI Platform API key.

Blocked example:

```text
provider=openai-codex
api_mode=codex_app_server
auth_surface=platform_api_key
```

Allowed shape:

```text
provider=openai or openai-codex
api_mode=codex_app_server
auth_surface=oauth
cost_surface=subscription
setup_mode=codex_app_server_opt_in
```

This catches the failure mode where Hermes silently falls back from the intended ChatGPT/OAuth subscription route to per-token OpenAI API billing.

## Cost-surface policy

The proof classifies cost as one of:

- `subscription`
- `local`
- `per_token_api`
- `cloud_metered`
- `none`

By default, Hermes records the cost surface without forbidding API-backed routes globally, because many users intentionally configure paid APIs. Callers that need a stricter environment can pass a policy such as:

```python
verify_agent_route_contract(
    ...,
    policy={"allowed_cost_surfaces": ["subscription", "local"]},
)
```

That blocks `per_token_api` before a model turn starts.

## Runtime integration points

- `AIAgent.__init__` builds `agent._route_proof` after credentials/fallback are resolved and before the first model turn.
- Delegated child agents pass `route_surface="delegation"`.
- Cron agents pass `route_surface="cron"`.
- TUI session info includes `route_proof`.
- Dashboard `/api/status` includes a dashboard route proof and seven-tier route plan.
- Dashboard `/api/harness/route-plan` exposes the same plan as a focused harness endpoint.
- Harness turn traces include `route_proof` in `turn.start` and normalized turn result records.

## Testing

Focused route-contract tests live in:

- `tests/hermes_cli/test_route_contracts.py`
- `tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_get_status`
- `tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_harness_route_plan_endpoint_is_metadata_only_and_seven_tier`
- `tests/agent/test_hermes_harness.py::test_control_plane_harness_exposes_route_plan`

Run them through the hermetic wrapper:

```bash
scripts/run_tests.sh \
  tests/hermes_cli/test_route_contracts.py \
  tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_get_status \
  tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_harness_route_plan_endpoint_is_metadata_only_and_seven_tier \
  tests/agent/test_hermes_harness.py::test_control_plane_harness_exposes_route_plan \
  -q
```
