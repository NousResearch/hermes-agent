---
sidebar_position: 6
title: "AGENTS.md Reference"
description: "Detailed reference material moved out of AGENTS.md to keep the runtime prompt compact"
---

# AGENTS.md Reference

`AGENTS.md` is runtime prompt surface. It should stay compact and contain only
rules a coding agent needs in most sessions. This page holds longer rationale,
subsystem reference pointers, and examples that agents can load on demand.

## Context budget policy

Hermes loads project context files such as `AGENTS.md` into the coding-agent
system prompt. Oversized context increases first-turn/model-switch latency and
cost, and the runtime truncates files beyond `context_file_max_chars`. If an
invariant sits past the cut line, the agent may silently miss it.

Budget discipline:

- Keep runtime-critical invariants inline in `AGENTS.md`.
- Move rationale, history, detailed subsystem docs, and rarely-used examples
  here or to other developer docs.
- Leave a short summary plus link in `AGENTS.md`.
- Keep hard CI budget below the runtime truncation limit to preserve margin.
- Review each `AGENTS.md` change as either invariant or reference; invariants
  must stay inline.

## Dependency pinning policy

Dependency pinning is a supply-chain invariant, not ordinary docs guidance. The
short rule stays inline in `AGENTS.md`; this reference records the treatment by
source type. See also
[`CONTRIBUTING.md`](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#dependency-pinning-policy-supply-chain-hardening).

| Source type | Required treatment | Rationale |
|---|---|---|
| PyPI package | `>=floor,<next_major` | PyPI versions are immutable once published, but new versions can be pushed into your range. A `<next_major` ceiling prevents silent major-version jumps. |
| Git URL | Full commit SHA | Branches and tags are mutable refs; SHAs are content-addressed. |
| GitHub Actions | Full commit SHA plus version comment | Action tags are mutable refs, so pin `uses:` to a SHA and keep the human-readable version as a comment. |
| CI-only pip installs | `==exact` | Hermetic CI builds can be exact because churn is acceptable there. |

Every new PyPI dependency in a PR must have a `<next_major` upper bound.
Unbounded `>=X.Y.Z` specs are rejected by review and flagged by supply-chain CI.
For pre-1.0 packages, keep a narrow minor-version window rather than allowing
all `<1` releases.

## Contribution rubric

Hermes is expansive at the product edges and conservative at the core waist.
New platforms, providers, desktop/TUI/dashboard features, plugins, skills, and
setup/config UX are welcome when they fit existing extension surfaces. New core
model tools are expensive because their schemas are sent on every API call, so
they are a last resort.

### What we want

- Real bug fixes with reproduction on current `main`, a line-level account of
  where the symptom manifests, and a fix for the whole bug class.
- Edge expansion through adapters, providers, plugins, skills, CLI commands,
  and setup/config UX.
- Declared refactors that extract god-files into focused modules while
  preserving behavior.
- Behavior-contract tests that assert relationships and invariants, not
  snapshots of model catalogs, config versions, or enumeration counts.
- E2E validation for resolution chains, config propagation, security
  boundaries, remote backends, and file/network I/O.
- Cache-safe, role-alternation-safe changes that keep the system prompt stable
  for the life of a conversation.
  Strict role alternation means never writing two same-role messages in a row.

### What we don't want

- Speculative hooks or callbacks with no concrete consumer.
- User-facing `HERMES_*` env vars for non-secret behavior; put behavior in
  `config.yaml` and reserve `.env` for credentials.
- Core tools when terminal/file, CLI+skill, plugin, or MCP can solve it.
- Pagination/offset escape hatches on instructional tools the agent must read
  fully.
- Security mitigations that destroy the feature they secure.
- Outbound telemetry or attribution without opt-in gating.
- Plugins that modify core files instead of using generic plugin surfaces.

### Verify premise and intent

Before accepting or writing a fix, verify the claimed bug against the actual
code and runtime. Some limitations are intentional design (for example,
profile isolation). Some apparently missing pieces are load-bearing omissions.
Use `git log -p -S "<symbol>"` to understand original intent before changing a
restriction. If you cannot point to the exact line where the bug manifests and
show how the fix changes that behavior, the premise is not yet verified.

### Footprint ladder

Choose the least permanent surface that solves the problem:

1. Extend existing code.
2. CLI command + skill.
3. Service-gated tool with `check_fn` so the schema appears only when configured.
4. Plugin.
5. MCP server in the catalog.
6. New core tool only when fundamental, broadly useful, and unreachable through
   the surfaces above.

## Testing

Use `scripts/run_tests.sh` for normal verification because it mirrors CI:
credential variables are unset, `HOME` / `HERMES_HOME` are isolated, timezone is
UTC, locale is C.UTF-8, xdist is enabled, and the in-tree subprocess isolation
plugin prevents per-test state leakage.

Useful commands:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x
scripts/run_tests.sh -v --tb=long
scripts/run_tests.sh --no-isolate tests/foo/
```

Avoid change-detector tests:

```python
# Bad: freezes a changing catalog snapshot
assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]
assert DEFAULT_CONFIG["_config_version"] == 21

# Good: asserts relationships or behavior
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
for model in _PROVIDER_MODELS["huggingface"]:
    assert model.lower() in DEFAULT_CONTEXT_LENGTHS_LOWER
```

## Subsystem reference index

Load code directly for exact definitions; this index points to entry points.

- `run_agent.py` — `AIAgent`, conversation loop, provider dispatch,
  streaming/non-streaming, compression, turn lifecycle.
- `agent/system_prompt.py` and `agent/prompt_builder.py` — stable/context/
  volatile prompt tiers, context file loading, truncation, skills prompt cache.
- `model_tools.py`, `toolsets.py`, `tools/registry.py` — tool discovery,
  schema post-processing, toolset membership, and tool execution.
- `tools/` — built-in tool implementations; auto-discovered through the
  registry. Prefer service-gated tools for optional integrations.
- `cli.py`, `hermes_cli/` — CLI orchestration, setup, config commands,
  curses UI, subcommands.
- `gateway/run.py`, `gateway/session.py`, `gateway/platforms/` — gateway
  runner, sessions, adapters, approval/control message bypass paths.
- `ui-tui/`, `tui_gateway/` — Ink TUI frontend and Python JSON-RPC backend.
- `plugins/` — plugin system; memory providers, model providers, platforms,
  context engines, observability, image generation, kanban, and other edge
  capabilities.
- `skills/`, `optional-skills/` — bundled skills and heavier/niche skills.
- `cron/` and `tools/cronjob_tools.py` — scheduled jobs.
- `agent/subdirectory_hints.py` — progressive discovery of subdirectory
  `AGENTS.md` / `CLAUDE.md` hints as files are read.
- `hermes_constants.py` — profile-aware `get_hermes_home()` and
  `display_hermes_home()`.
- `hermes_state.py` — session SQLite store and FTS search.

## Configuration notes

- Non-secret behavior belongs in `config.yaml`.
- `.env` is for credentials only.
- Use `get_hermes_home()` for code paths and `display_hermes_home()` for
  user-facing paths.
- Module-level constants may call `get_hermes_home()` after profile override has
  run; do not use `Path.home() / ".hermes"` for active-profile state.
- Gateway adapters with unique credentials should use scoped locks in
  `gateway.status` to prevent two profiles from using the same token.

## TypeScript notes

The short TypeScript runtime style stays in `AGENTS.md`; deeper details should
live near the owning frontend package. Keep route roots thin, prefer nanostores
for shared state, colocate action modules, and use table-driven mappings for
ids/routes/views.
