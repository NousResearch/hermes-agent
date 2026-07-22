---
sidebar_position: 2
title: "Architecture Adherence Checklist"
description: "Review checklist and decision trees for keeping Hermes Agent changes aligned with the documented architecture"
---

# Architecture Adherence Checklist

Use this checklist before opening or reviewing a PR that touches Hermes Agent internals. The goal is not to block useful changes; it is to keep the core narrow, cache-friendly, profile-safe, and easy to reason about as more entry points and integrations are added.

## Subsystem ownership

Map the change to its owning subsystem before writing code.

| Concern | Correct home | Watch for |
|---|---|---|
| Agent orchestration | `run_agent.py`, `agent/` | Duplicated conversation loops in entry points |
| Prompt assembly | `agent/prompt_builder.py`, prompt/context modules | Ad hoc prompt strings in CLI, gateway, cron, or ACP code |
| Provider/model routing | `hermes_cli/runtime_provider.py`, provider/auth modules, adapters | Provider-specific branches in unrelated modules |
| Tool registration and dispatch | `tools/registry.py`, `model_tools.py`, `toolsets.py` | Manual import lists or direct handler calls from surfaces |
| Session persistence | `hermes_state.py`, `gateway/session.py` | Raw writes to session DB/files from tools or entry points |
| Messaging platforms | `gateway/`, `gateway/platforms/`, platform plugins | Gateway behavior leaking into the core agent loop |
| Scheduled work | `cron/` | Long-lived shell workarounds for first-class cron jobs |
| Optional integrations | Plugins, MCP, provider registries, `check_fn` gates | Hard imports that make optional dependencies mandatory |

If a change spans multiple rows, split the implementation and tests by subsystem.

## Dependency direction

Preserve the documented dependency chain:

```text
tools/registry.py
       ↑
tools/*.py
       ↑
model_tools.py
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

Checklist:

- [ ] `tools/registry.py` remains dependency-light.
- [ ] Tool files self-register with `registry.register()`.
- [ ] Tool schemas and dispatch still flow through `model_tools.py`.
- [ ] Entry points do not manually wire individual tool implementation files.
- [ ] New cross-subsystem imports point inward toward stable abstractions, not sideways into implementation details.

## Prompt stability

Prompt stability is a runtime and cost invariant.

- [ ] No mid-conversation system-prompt mutation except explicit reset/model/session actions.
- [ ] Stable prompt content remains in the cacheable prefix where applicable.
- [ ] Volatile values such as memory, timestamps, runtime status, and profile data stay in volatile prompt sections.
- [ ] Toolset changes require a new session/reset rather than silently changing schemas in place.
- [ ] Prompt changes include tests or snapshots that prove the intended stable/volatile behavior.

## Profile isolation

Every profile must get its own config, memory, sessions, skills, cron state, and gateway PID.

- [ ] Code uses `get_hermes_home()` or another profile-aware helper for Hermes-managed paths.
- [ ] No new hardcoded writes to `~/.hermes` or platform-specific home paths.
- [ ] Tests use temporary `HERMES_HOME` or profile fixtures where state is touched.
- [ ] Cron, gateway, CLI, ACP, and tests agree on the same profile-scoped paths.
- [ ] User-facing examples may mention `~/.hermes`, but implementation code should not depend on it unless the module is specifically resolving the default home.

## Optional subsystems

Optional systems should fail closed and degrade clearly.

- [ ] Optional dependency imports are lazy or protected.
- [ ] Tool availability is guarded with `check_fn` and `requires_env` where appropriate.
- [ ] Missing credentials or packages produce actionable setup messages.
- [ ] Plugins and MCP servers use extension points rather than editing core wiring.
- [ ] New integrations do not expand the default model tool schema unless there is a strong reason.

## Decision trees

### Adding a tool

```text
Need a model-callable capability?
  → First check whether an existing tool, CLI command, or skill is enough.
  → If model-callable access is justified, create tools/<name>.py.
  → Register with registry.register(...).
  → Add check_fn/requires_env for optional credentials or services.
  → Expose through toolsets.py intentionally.
  → Test schema exposure, availability gating, dispatch, and JSON-string result shape.
```

### Adding a provider

```text
Need a new model backend?
  → Add or extend auth/credential handling if credentials are new.
  → Route provider/model tuples through runtime provider resolution.
  → Reuse an existing API mode when possible.
  → Add an adapter only when request/response conversion differs.
  → Update model catalog/aliases if needed.
  → Test config, environment, OAuth/API-key fallback, and error messages.
```

### Adding a platform

```text
Need a messaging integration?
  → Implement a gateway platform adapter or platform plugin.
  → Normalize inbound messages to MessageEvent.
  → Let GatewayRunner handle auth, session routing, and agent creation.
  → Implement outbound delivery through the adapter.
  → Test authorization, session key selection, delivery formatting, and failure handling.
```

### Adding persistent state

```text
Need durable conversation or metadata state?
  → Use the session/state abstractions instead of ad hoc files when it is session data.
  → Keep profile boundaries explicit.
  → Add migration/backward-compatibility handling when schema changes.
  → Test read/write/search/lineage behavior and concurrent access where relevant.
```

## Validation checklist

Before marking the change ready:

- [ ] The change is in the owning subsystem.
- [ ] Core agent behavior remains platform-agnostic.
- [ ] Prompt stability and prompt caching assumptions are preserved.
- [ ] Profile-aware paths are used for Hermes-managed state.
- [ ] Optional integrations are gated and do not add hard startup dependencies.
- [ ] New tools self-register and are exposed through toolsets intentionally.
- [ ] Session writes go through the session/state layer.
- [ ] Tests cover every affected execution path: CLI, gateway, cron, ACP, tools, provider resolution, or sessions as applicable.
- [ ] Documentation and PR checklist updates are included when architectural rules or flows change.

## Common review questions

- Could this be a skill, CLI command, plugin, or MCP server instead of a new core model tool?
- Does this duplicate a resolver, registry, adapter, or session abstraction that already exists?
- Would this change behave the same under a named profile, gateway profile, cron run, and test `HERMES_HOME`?
- Does this require a new session because it changes tools or the system prompt?
- Does the failure mode tell the user what to configure or install next?
