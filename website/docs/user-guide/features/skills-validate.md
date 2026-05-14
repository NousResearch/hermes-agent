---
sidebar_position: 4
title: "Skills validate"
description: "Static lint pass for installed skills — frontmatter, sections, metadata, line counts. Complementary to the Curator."
---

# Skills Validate

`hermes skills validate` is a **read-only static lint pass** for installed
skills. It checks frontmatter completeness, recommended sections, metadata
fields, and structural limits against a configurable ruleset, then reports
findings. It never modifies skills on disk.

It is **complementary to the [Curator](/docs/user-guide/features/curator)** —
the Curator is a background LLM-driven mutation pass that prunes and
consolidates agent-authored skills; this is a deterministic checker that
runs against *any* installed skill (bundled, hub-installed, or
agent-authored) and produces a report.

## Quick start

Validate every installed skill against the default (lenient) ruleset:

```bash
hermes skills validate
```

Validate a single skill against the strict agentskills.io ruleset:

```bash
hermes skills validate my-skill --ruleset agentskills
```

Validate a SKILL.md file directly:

```bash
hermes skills validate path/to/SKILL.md --ruleset agentskills
```

Use a custom ruleset:

```bash
hermes skills validate --ruleset /path/to/rules.yaml
```

Treat suggestions as blocking for exit-code purposes:

```bash
hermes skills validate --strict
```

## Built-in rulesets

### `hermes` (default — lenient)

Matches the existing `_validate_frontmatter` contract in
`tools/skill_manager_tool.py`. Useful as a regression check: every existing
Hermes skill should pass.

Rules:
- `frontmatter.present` (BLOCKING)
- `frontmatter.parseable` (BLOCKING)
- `frontmatter.name.present` (BLOCKING)
- `frontmatter.description.present` (BLOCKING)
- `frontmatter.description.length` (BLOCKING — max 1024 chars)
- `body.non_empty` (BLOCKING)

### `agentskills` (strict)

Implements the full [agentskills.io](https://agentskills.io) skill spec.
Useful when authoring or accepting skills meant to be portable across
agentskills.io-compatible runtimes.

Additional rules on top of `hermes`:
- `frontmatter.name.kebab_case` (BLOCKING)
- `frontmatter.name.matches_dir` (BLOCKING — name must match directory)
- `frontmatter.license.present` (SUGGEST)
- `frontmatter.allowed_tools.present` (SUGGEST)
- `frontmatter.metadata.author.present` (SUGGEST)
- `frontmatter.metadata.version.present` (SUGGEST)
- `frontmatter.metadata.version.semver` (SUGGEST — must match `M.m` or `M.m.p`)
- `frontmatter.metadata.domain.present` (SUGGEST)
- `frontmatter.metadata.complexity.valid` (SUGGEST — must be one of `basic`,
  `intermediate`, `advanced`)
- `frontmatter.metadata.tags.present` (SUGGEST)
- `section.when_to_use.present` (SUGGEST — recommends `## When to Use`)
- `section.procedure.present` (SUGGEST — recommends `## Procedure`)
- `body.line_count` (SUGGEST — suggested max 500 lines)

## Custom rulesets

A custom ruleset is a YAML file with a top-level `rules` mapping of
`rule_id: severity`. Valid severities are `BLOCKING`, `SUGGEST`, `OK`. Use
`OK` to silence a rule that a parent ruleset would otherwise enable.

```yaml
# rules.yaml — relaxed agentskills variant
rules:
  frontmatter.present: BLOCKING
  frontmatter.parseable: BLOCKING
  frontmatter.name.present: BLOCKING
  frontmatter.name.kebab_case: BLOCKING
  frontmatter.description.present: BLOCKING
  frontmatter.description.length: BLOCKING
  frontmatter.metadata.version.semver: SUGGEST
  body.non_empty: BLOCKING
  body.line_count: OK    # disable line-count check entirely
```

```bash
hermes skills validate --ruleset rules.yaml
```

## Exit code

`hermes skills validate` exits non-zero when any skill has BLOCKING findings.
With `--strict`, SUGGEST findings also count. This makes it usable in CI:

```bash
hermes skills validate --ruleset agentskills --strict
```

## Output

```
Skill Validation — ruleset: agentskills
┌──────────────────┬──────────┬─────────┬──────────┐
│ Skill            │ Blocking │ Suggest │ Status   │
├──────────────────┼──────────┼─────────┼──────────┤
│ my-clean-skill   │        0 │       0 │ OK       │
│ my-thin-skill    │        0 │       4 │ SUGGEST  │
│ broken-skill     │        2 │       1 │ BLOCKING │
└──────────────────┴──────────┴─────────┴──────────┘

broken-skill  ~/.hermes/skills/broken-skill/SKILL.md
  BLOCKING  frontmatter.name.matches_dir   name 'wrong' does not match directory 'broken-skill'
  BLOCKING  frontmatter.description.length description is 1500 characters, max 1024
  SUGGEST   section.when_to_use.present    missing '## When to Use' section (recommended by agentskills.io)

Summary: 1 OK, 1 with suggestions, 1 with blocking findings.
```

## Relationship to the Curator

| Aspect | Curator | Skills validate |
|---|---|---|
| Engine | LLM (forked auxiliary agent) | Deterministic (no LLM) |
| Scope | Agent-authored skills only | Any skill (bundled, hub, agent-authored) |
| Action | Mutates: patch / consolidate / archive | Reports only; never writes |
| Trigger | Background (7d cycle, idle gate) | On-demand CLI |
| Use case | Drift / staleness in self-improvement loop | Spec compliance, CI gate, authoring lint |

The two are designed to be used together: the validator catches structural
problems early (CI, pre-install, post-authoring), while the Curator handles
long-term lifecycle of agent-grown skills.
