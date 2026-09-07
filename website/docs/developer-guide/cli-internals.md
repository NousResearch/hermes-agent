---
sidebar_position: 15
title: "CLI Internals"
description: "How hermes_cli is shaped: slash dispatch, config loaders, the skin engine, the transactional update pipeline, and process-identity rules"
---

# CLI Internals

Companion to `hermes_cli/AGENTS.md` (the rules) — this page holds the longer explanations.

## Update pipeline

The stage-by-stage contract (`plan → snapshot → apply → restart-per-kind → verify → report`) and the
field failure each stage guards are documented in `hermes_cli/AGENTS.md`; user-facing behaviour
(receipts, `--plan`, snapshot modes) is in [Updating](../getting-started/updating.md).

## Process identity: never infer it from argv substrings

The bug class behind ~10 fleet-update issues (#90778, #87594, #78089, #76129, #91964, ...):
classifying a process by `"serve" in cmdline` or similar. `kanban --preserve-cache` contains
"serve"; a flag VALUE can equal a subcommand (`-m dashboard serve`); truncated cmdlines hide the real
subcommand. Rules:

- Use the canonical matchers: `gateway.status.looks_like_gateway_command_line` (gateway run),
  `hermes_cli.update_cmd._hermes_holder_subcommand` (top-level subcommand of any Hermes argv). Never
  hand-roll token scans.
- Flag sets must be DERIVED from the parser (`_holder_value_flags()` introspects
  `build_top_level_parser()`), never hand-written lists — they drift.
- Never blanket-exclude ancestors from process scans: when `/update` runs as the gateway's child, a
  gateway ancestor must stay visible to the pause machinery (#87594). Exclude interactive ancestry,
  carve out gateway-shaped ancestors.
- Match on FULL cmdlines; truncate only at display time (#78089).
- Before adding any new scan heuristic, read #92091 — the gateway control socket replaces scans as
  the primary coordination mechanism; scans are the fallback layer for old/crashed processes.

## Skin engine — what skins customize

| Element | Skin key | Used by |
|---|---|---|
| Banner panel border / title / section headers / dim / body | `colors.banner_border`, `banner_title`, `banner_accent`, `banner_dim`, `banner_text` | `banner.py` |
| Response box border | `colors.response_border` | `cli.py` |
| Spinner faces (waiting / thinking) | `spinner.waiting_faces`, `spinner.thinking_faces` | `display.py` |
| Spinner verbs / wings (optional) | `spinner.thinking_verbs`, `spinner.wings` | `display.py` |
| Tool output prefix / per-tool emojis | `tool_prefix`, `tool_emojis` | `display.py` → `get_tool_emoji()` |
| Agent name / welcome / response label / prompt symbol | `branding.agent_name`, `welcome`, `response_label`, `prompt_symbol` | `banner.py`, `cli.py` |

Built-in skins (`_BUILTIN_SKINS` in `hermes_cli/skin_engine.py`): `default` (classic gold/kawaii),
`ares` (crimson/bronze with custom spinner wings), `mono` (grayscale), `slate` (cool blue). Add a
built-in as a dict entry `{"name", "description", "colors", "spinner", "branding", "tool_prefix"}`.
User skins are `~/.hermes/skins/<name>.yaml` with the same keys, activated with `/skin <name>` or
`display.skin: <name>`; the full YAML template is in the
[Skins & Themes](../user-guide/features/skins.md) user guide.

## Profiles: multi-instance support

Hermes supports profiles — fully isolated instances, each with its own `HERMES_HOME` (config, API
keys, memory, sessions, skills, gateway). `_apply_profile_override()` in `hermes_cli/main.py` sets
`HERMES_HOME` before any module imports, so every `get_hermes_home()` reference scopes to the active
profile. Profile operations are HOME-anchored (`_get_profiles_root()` returns
`Path.home() / ".hermes" / "profiles"`, not `get_hermes_home() / "profiles"`) so
`hermes -p coder profile list` sees all profiles regardless of which one is active — intentional.
Profile-safe coding rules are in the root `AGENTS.md`; multiplex secret-scope rules in
`gateway/AGENTS.md`.
