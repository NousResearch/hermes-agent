---
sidebar_position: 20
title: "Vertical Agent Kit"
description: "CLI scaffolding for constrained vertical agents — turn the four-layer pattern into a generated profile"
---

# Vertical Agent Kit

The [Building Constrained Vertical Agents](/guides/vertical-agents) guide explains the four-layer model — identity, context, workflow, and tool constraints — and *where* each belongs. The **Vertical Agent Kit (VAK)** turns that model into a concrete CLI workflow: pick a blueprint, answer a short questionnaire, and get a ready-to-review scaffold with `SOUL.md`, `USER.template.md`, operations rules, and config patches.

It is a Hermes-native scaffolding tool. It does not introduce a new runtime or wrapper architecture; it composes the primitives that already exist (profiles, `SOUL.md`, memories, skills, `platform_toolsets`) and leaves the generated files for you to refine and activate.

## When to use this

Use the kit when you want to go from "I need a specialist agent" to a shaped draft in a few commands:

- You have a recurring domain job (support triage, research synthesis, etc.).
- You want the agent's voice, context, workflow, and tool surface separated from the start.
- You prefer a generated scaffold you can review instead of writing the files from scratch.

Do not use it when you want a fully autonomous generalist. The whole point is to narrow the agent, not broaden it.

## Install / availability

The kit ships with the Hermes CLI. No separate install is required.

```bash
hermes vertical-agent --help
```

## Quick start

```bash
# See bundled blueprints
hermes vertical-agent list

# Scaffold a support agent
hermes vertical-agent init

# Verify the generated files
hermes vertical-agent verify ./out/my-agent

# Best-effort smoke test
hermes vertical-agent smoke ./out/my-agent
```

The wizard asks for role, objective, users, tone, scope, refusal edges, evidence sources, systems, and decision style. Each question has a blueprint-based default, so you can press Enter to accept it and iterate later.

## Bundled blueprints

Blueprints are starter scaffolds that follow the four-layer model:

| Blueprint | Domain | Default tool posture |
|---|---|---|
| `support` | Support ticket triage and response | Read-first, escalate early, generic execution as fallback |
| `research` | Bounded evidence gathering and synthesis | Search and citation first, no scheduling/image/voice |

Each blueprint generates:

- `SOUL.md` — identity and voice only
- `USER.template.md` — domain and user context, ready to move to `~/.hermes/memories/USER.md`
- `OPERATIONS.md` — scope, refusal edges, tool order, and fallback policy
- `config.patch.yaml` — suggested profile description, required skills, and helper mapping
- `skills.manifest.yaml` — candidate skills to review, not an auto-install list
- `required-skills.md` — short skill reasoning for your own review

## From scaffold to running profile

The kit writes files. You still decide whether and how to activate them. A typical handoff:

1. **Review `SOUL.md`**. Trim anything that sounds like workflow or domain facts.
2. **Move `USER.template.md` to memory**. For the default profile, copy it to `~/.hermes/memories/USER.md`. For a named profile, use `~/.hermes/profiles/<name>/memories/USER.md`.
3. **Turn `OPERATIONS.md` into a skill** (or keep it as an `AGENTS.md`/`.hermes.md` project file if you prefer). A minimal skill lives at `~/.hermes/skills/<vertical>/<name>/SKILL.md` and contains trigger conditions, procedure, evidence requirements, and fallback policy.
4. **Create a dedicated profile**:
   ```bash
   hermes profile create <name> --clone
   hermes profile use <name>
   ```
5. **Prune toolsets** using `platform_toolsets` and `agent.disabled_toolsets` as described in [Building Constrained Vertical Agents](/guides/vertical-agents#tool-pruning-through-native-config).
6. **Run the smoke test** again after activation:
   ```bash
   hermes vertical-agent smoke ./out/<name>
   ```

## Design principles

The kit enforces the same principles as the guide:

- **SOUL.md is for identity only.** Blueprints generate short voice files; you keep them short.
- **USER.md is for user/domain context.** The generated file is a starter, not the runtime file.
- **Skills own workflow.** `OPERATIONS.md` is shaped so it can become a skill or an `AGENTS.md` file.
- **Prune before prompting.** Each blueprint includes a `config.patch.yaml` with a minimal helper-first posture.
- **Helper-first, fallback-last.** Generic execution is deliberately marked as fallback in every blueprint.

## Verifying a scaffold

`hermes vertical-agent verify` checks that the expected files exist and are non-empty. It does not run Hermes; it validates shape only.

`hermes vertical-agent smoke` adds a best-effort probe:

- Confirms `SOUL.md` mentions voice or identity.
- Runs `hermes --version` if the CLI is on `PATH`.
- Falls back to file-only checks if Hermes is not installed locally.

## Extending the kit

Blueprints are plain directories under `hermes_cli/vertical_agent_kit_data/blueprints/`. To add your own:

1. Create a new directory with at least `SOUL.md`, `USER.template.md`, and `OPERATIONS.md`.
2. Use `{{VARIABLE}}` placeholders anywhere in the files (the renderer supports all wizard variables).
3. Re-run `hermes vertical-agent list` to confirm it appears.

For a repository of external, non-bundled blueprints, keep them outside the Hermes tree and reference them from a skill or an `AGENTS.md` file. The kit is intentionally not a marketplace; it ships starting points, not every possible vertical.

## Relationship to the vertical agents guide

- The [guide](/guides/vertical-agents) teaches the *pattern*.
- The kit applies the pattern to a *generated scaffold*.
- The guide is the right place to understand *why* the files are shaped this way.
- This page and the CLI are the right place to *start using* the shape.

If you are reviewing or building on top of the guide, the kit is the concrete mechanism you can point to.
