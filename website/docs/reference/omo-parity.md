---
title: Hermes OMO Parity Matrix
description: Honest OMO/OpenCode compatibility status, configuration decisions, non-goals, and verification commands.
---

# Hermes OMO Parity Matrix

This page is an evidence ledger, not a marketing claim. Hermes does **not** claim full literal OMO/OpenCode parity on this branch. It documents verified Hermes-native compatibility surfaces, unmerged wave-backed work, explicit non-goals, and the commands that must pass before any release or PR claims parity.

Final all-up parity claims require the Wave 1-7 implementation branches to be merged into the branch under test and reverified together. Until then, wave-backed items below mean "implemented or prototyped on a separate wave branch, pending integration review."

## Compatibility decisions

### Canonical config format

Hermes canonical configuration is YAML at `~/.hermes/config.yaml`.

- JSONC is **not** a core runtime config format.
- Foreign OMO/OpenCode JSONC may be supported only by explicit import/translation tooling.
- Runtime config loading must not silently accept a second canonical format.

Verification:

```bash
uv run --extra dev python -m pytest -q tests/hermes_cli/test_config_validation.py
```

### Config knobs exposed for Wave 8

`omo_compat` is a conservative compatibility-decision surface. It documents or validates knobs that are otherwise easy to overclaim:

| Key | Default | Risk boundary |
| --- | --- | --- |
| `jsonc_config` | `import-only` | YAML remains canonical; JSONC is translation/import-only. |
| `disable_omo_env` | `false` | Compatibility stance only; does not remove existing Hermes environment behavior. |
| `hashline_edit` | `false` | Hashline-style edit UX is opt-in/config documented; runtime proof lives with file-tool tests. |
| `stale_edit_mode` | `warn` | Backward-compatible default; strict rejection requires explicit `error`. |
| `dynamic_context_pruning.protected_tools` | `[]` | Additive aliases only; built-in protected tools remain code-owned. |
| `named_agents.mode` | `primary` | Valid modes are `primary`, `subagent-only`, `disabled`. |
| `named_agents.color` | `auto` | UX metadata only unless a runtime/UI wave proves consumption. |
| `named_agents.providerOptions` | `{}` | Provider request options must stay non-secret and schema-validated by owning surfaces. |
| `named_agents.permissions` | `{}` | Permission gates are explicit booleans; omitted gates are not implied. |
| `task_scheduler.enabled` | `false` | Avoids claiming Atlas/OMO scheduler parity unless Wave 5 runtime exists and is enabled. |
| `runtime_modes.ralph.max_iterations` | `100` | Documents OMO-style cap; runtime enforcement must be proven by workflow tests. |
| `mcp.builtins` | `configured` | Hermes does not silently enable network MCP defaults; users configure servers. |

Feature map:

- Defaults and validation: `hermes_cli/config.py`
- Example config: `cli-config.yaml.example`
- User docs: `website/docs/user-guide/configuration.md`
- Validation tests: `tests/hermes_cli/test_config_validation.py`

## Rules and Claude compatibility

Hermes implements project context ingestion with a first-match precedence contract:

1. `.hermes.md` / `HERMES.md` walking up to the git root
2. `AGENTS.md` / `agents.md` in the working directory
3. `CLAUDE.md` / `claude.md` in the working directory
4. `.cursorrules` plus `.cursor/rules/*.mdc` in the working directory
5. `SOUL.md` from `HERMES_HOME` as a separate identity/context source

Feature map:

- Prompt assembly: `agent/prompt_builder.py`
- Documentation: `website/docs/user-guide/features/context-files.md`
- Existing proof: `tests/agent/test_prompt_builder.py`, `tests/agent/test_subdirectory_hints.py`

Wave 7 adds separate project-local skill discovery compatibility, including `.claude/skills` in explicit project-rooted precedence order. That is a skill-loader compatibility surface, not a full Claude Code runtime clone.

Explicit non-goals unless future waves add tests:

- `.claude/commands` is not treated as a separate automatic command loader.
- Claude Code private task API behavior is not cloned.
- Cursor `.mdc` files are ingested as context rules, not as a full Cursor runtime.
- `.claude/skills` compatibility does not imply full Claude Code command/task/runtime parity.

## Wave-backed implementation ledger

| Wave | Area | Status | Evidence / honest boundary |
| --- | --- | --- | --- |
| Wave 1 | Code intelligence | Partial, pending branch merge | Bounded code-intel MVP: preview-first rename, conservative Python project-wide rename in provable cases, read-only definition/reference tools, and syntax-aware read chunking metadata. Not full multi-language LSP server parity. Evidence: `tools/code_intel.py`, `tools/tree_sitter_chunking.py`, `tools/lsp_rename_tool.py`, `tests/tools/test_code_intel.py`, `tests/tools/test_tree_sitter_chunking.py`. |
| Wave 2 | Named agents / runtime identity | Implemented pending branch merge | Canonical named-agent contracts, config validation/rendering, runtime activation propagation, and delegation enforcement including disabled-agent blocking. Evidence: `agent/archetypes.py`, `hermes_cli/config.py`, `run_agent.py`, `tools/delegate_tool.py`, `tests/agent/test_named_agents.py`. |
| Wave 3 | Model orchestration policy | Partial, pending branch merge | Central model-policy resolver with tested precedence, fallback-chain construction, ultrawork handling, and sanitized trace output wired into delegation and `/model` resolution. Full universal runtime model-selection unification is not proven until integrated. Evidence: `agent/model_policy.py`, `hermes_cli/model_switch.py`, `tools/delegate_tool.py`, `tests/agent/test_model_policy.py`. |
| Wave 4 | Runtime modes / plan-first / Ralph / Metis / refactor | Implemented pending branch merge | Gap-check continuation, plan-first read-only approval gating, scoped `/refactor` command contracts, exact Ralph completion semantics, and Metis-style gap checks are present on the wave branch. Evidence: `agent/gap_check.py`, `agent/continuation_engine.py`, `hermes_cli/command_templates.py`, `hermes_cli/commands.py`, `tests/agent/test_gap_check.py`, `tests/hermes_cli/test_refactor_command.py`. |
| Wave 5 | Atlas task graph / persistent scheduler | Implemented pending branch merge | Persistent task tool, Atlas-lite scheduler, lifecycle hooks/redaction, execution-supervisor scheduling, and guarded task mutation via the task tool. Evidence: `agent/task_scheduler.py`, `tools/task_tool.py`, `agent/task_store.py`, `tests/agent/test_task_scheduler.py`, `tests/tools/test_task_tool.py`. |
| Wave 6 | PR/workflow guardrails | Partial prototype, pending branch merge | PR workflow state machine, `gh` polling adapter shape, AI-signed comment formatting, remediation-request generation, and explicit refusal of merge actions. Do not claim default-integrated end-to-end PR shepherd behavior yet; observed branch includes duplicated Wave 5 task files. Evidence: `agent/pr_workflow.py`, `tools/pr_workflow_tool.py`, `tests/agent/test_pr_workflow.py`, `tests/tools/test_pr_workflow_tool.py`. |
| Wave 7 | Command/hook/MCP/skill control plane | Implemented pending branch merge | Unified command metadata envelope for CLI/TUI/ACP-facing discovery, hook introspection, read-only MCP control-plane status, and project skill precedence including `.claude/skills`. Evidence: `hermes_cli/commands.py`, `gateway/hooks.py`, `agent/shell_hooks.py`, `agent/skill_commands.py`, `tools/mcp_tool.py`, `tests/cli/test_control_plane_metadata.py`, `tests/cli/test_mcp_control_plane_status.py`. |
| Wave 8 | Config/platform/docs closeout | Implemented on this branch | Conservative `omo_compat` config surface, JSONC stance, parity matrix, and docs/sidebar wiring. Evidence: `hermes_cli/config.py`, `cli-config.yaml.example`, `website/docs/reference/omo-parity.md`, `website/docs/user-guide/configuration.md`, `tests/hermes_cli/test_config_validation.py`. |

## OMO feature classification

| Area | Status | Evidence / boundary |
| --- | --- | --- |
| Hashline/stale edit safety | Implemented differently / stronger where file-tool tests prove it | Hermes already has stale-edit and safety semantics in `tools/file_operations.py`, `tools/file_tools.py`, `tests/tools/test_file_operations.py`, `tests/tools/test_file_tools.py`, `tests/tools/test_file_staleness.py`; Wave 8 only documents the opt-in `omo_compat.hashline_edit` stance. |
| LSP/code intelligence | Partial and wave-backed | Wave 1 proves bounded Python code-intel surfaces; full project-wide/multi-language LSP server parity remains a non-claim. |
| Named agents and specialists | Implemented pending branch merge | Wave 2 provides Hermes-native named-agent contracts, modes, provider options, permissions, runtime snapshots, and delegate enforcement. |
| Category routing | Implemented where route tests pass | Existing Hermes route-category surfaces remain separate from named-agent identity; see `agent/route_categories.py` and delegate tests. |
| Model orchestration | Partial and wave-backed | Wave 3 adds the central resolver and trace behavior; final claim waits for integration through all model-selection chokepoints. |
| Runtime modes / Ralph / Metis / refactor | Implemented pending branch merge | Wave 4 adds the runtime/workflow behavior; final claim waits for merge and closeout tests. |
| Task graph / Atlas scheduler | Implemented pending branch merge | Wave 5 adds scheduler/task-tool behavior; default config remains off unless enabled and verified. |
| PR loop | Partial prototype | Wave 6 proves a PR-workflow substrate but not default-integrated end-to-end PR shepherd parity. No auto-merge. |
| Command/hook/MCP/skill control plane | Implemented pending branch merge | Wave 7 adds the unified metadata/status surfaces; no forced network auth or login side effects. |
| MCP ecosystem | Hermes-native implemented differently | MCP clients/OAuth/server bridge exist, but built-in external server defaults are `configured`, not silent always-on. |
| Skills | Hermes-native implemented / partial Claude compatibility | Hermes skills are first-class; project `.claude/skills` compatibility is wave-backed; `.claude/commands` remains non-goal. |
| CLI/config/platform | Hermes-native implemented differently | Python/Ink/TUI, YAML config, gateway platforms, hooks, and plugins are the canonical Hermes surfaces. |

## Explicit non-goals

Hermes should not clone these literally unless a future product decision reverses this page with tests:

- TypeScript/Bun implementation identity.
- OpenCode binary matrix, AVX/libc loader behavior, or package distribution quirks.
- Exact OpenCode visual/TUI affordances when Hermes has a Python/Ink/TUI equivalent.
- Core JSONC runtime config loading.
- Claude Code private task APIs or unverified `.claude/commands` precedence.
- Silent always-on network MCP server defaults.
- Auto-merge behavior in PR workflows.
- Full OMO parity claims before Waves 1-7 are merged, conflicts resolved, and their verification packets rerun together.

## Integration risks to resolve before release claims

- `tools/delegate_tool.py` is touched by Waves 2, 3, and 4. Merge this file serially and rerun delegate/runtime tests after every merge.
- `run_agent.py` is touched by Waves 2, 4, and 5. Treat it as a shared runtime chokepoint.
- `hermes_cli/config.py` is touched by Waves 2, 3, and 8. Preserve schema validation and readable errors.
- `toolsets.py` is touched by Waves 1 and 5. Verify tool exposure and protected surfaces after integration.
- `uv.lock` drift appears in multiple waves and should not be described as a feature; revert or normalize unless a dependency change is intentional.
- Wave 6 contains duplicated Wave 5 task scheduler/task tool files. During integration, keep Wave 5 as the owner of those files and land only the PR-workflow delta from Wave 6.

## Verification map

Run the family-specific packets after their corresponding branches are merged. Do not use this page as evidence of full parity without fresh command output.

### Wave 1 code-intel

```bash
uv run --extra dev python -m pytest -q \
  tests/tools/test_lsp_rename_tool.py \
  tests/tools/test_code_intel.py \
  tests/tools/test_tree_sitter_chunking.py \
  tests/tools/test_file_operations.py \
  tests/tools/test_file_tools.py \
  tests/test_toolsets.py \
  tests/agent/test_display.py
```

### Wave 2 named agents

```bash
uv run --extra dev python -m pytest -q \
  tests/agent/test_runtime_activation.py \
  tests/run_agent/test_runtime_activation_overlay_parity.py \
  tests/tools/test_delegate.py \
  tests/hermes_cli/test_config_validation.py \
  tests/agent/test_named_agents.py
```

### Wave 3 model policy

```bash
uv run --extra dev python -m pytest -q \
  tests/agent/test_model_policy.py \
  tests/hermes_cli/test_model_resolution_truthing.py \
  tests/hermes_cli/test_model_switch_variant_tags.py \
  tests/hermes_cli/test_model_switch_custom_providers.py \
  tests/run_agent/test_fallback_model.py \
  tests/run_agent/test_provider_fallback.py \
  tests/agent/test_model_metadata.py
```

### Wave 4 workflows

```bash
uv run --extra dev python -m pytest -q \
  tests/agent/test_runtime_activation.py \
  tests/run_agent/test_runtime_activation_overlay_parity.py \
  tests/agent/test_gap_check.py \
  tests/hermes_cli/test_refactor_command.py \
  tests/tools/test_delegate.py
```

### Wave 5 task scheduler

```bash
uv run --extra dev python -m pytest -q \
  tests/tools/test_task_tool.py \
  tests/agent/test_task_scheduler.py \
  tests/tools/test_delegate.py \
  tests/tools/test_delegate_subagent_timeout_diagnostic.py
```

### Wave 6 PR workflow

```bash
uv run --extra dev python -m pytest -q \
  tests/agent/test_pr_workflow.py \
  tests/tools/test_pr_workflow_tool.py \
  tests/agent/test_task_scheduler.py \
  tests/tools/test_task_tool.py
```

### Wave 7 control plane

```bash
uv run --extra dev python -m pytest -q \
  tests/agent/test_skill_commands.py \
  tests/gateway/test_hooks.py \
  tests/gateway/test_unknown_command.py \
  tests/cli/test_quick_commands.py \
  tests/test_tui_gateway_server.py \
  tests/acp/test_server.py \
  tests/cli/test_cli_mcp_config_watch.py \
  tests/acp/test_mcp_e2e.py \
  tests/cli/test_control_plane_metadata.py \
  tests/cli/test_mcp_control_plane_status.py
```

### Wave 8 closeout

```bash
uv run --extra dev python -m py_compile \
  hermes_cli/config.py run_agent.py agent/prompt_builder.py tui_gateway/server.py

uv run --extra dev python -m pytest -q \
  tests/hermes_cli/test_config_validation.py \
  tests/hermes_cli/test_setup.py \
  tests/test_tui_gateway_server.py

# If Node/npm are available:
cd ui-tui && npm test -- --run

git diff --check
git status --short --branch
git diff --stat
```
