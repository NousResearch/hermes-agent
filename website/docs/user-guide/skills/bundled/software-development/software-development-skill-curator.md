---
title: "Skill Curator — Curate and distill session workflows into tested skills"
sidebar_label: "Skill Curator"
description: "Curate and distill session workflows into tested skills"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Skill Curator

Curate and distill session workflows into tested skills.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/skill-curator` |
| Version | `0.1.0` |
| Author | Thamer (taljeri), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Skills`, `Curation`, `Workflow-Synthesis`, `Authoring`, `Automation` |
| Related skills | [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Skill Curator

Distill multi-turn conversation trajectories, complex tool sequences, and successful debugging sessions into standardized, repository-compliant Hermes skills. Inspired by DeepSeek Harness's skill curation engine, this skill turns one-off ad-hoc solutions into permanent, reusable capabilities with tests and documentation.

Zero external dependencies: uses standard library Python and coordinates with `hermes-agent-skill-authoring` standards.

## When to Use

- "Turn what we just did into a skill"
- "Distill this debugging workflow into an in-repo skill"
- "Extract a reusable skill from this session's transcript"
- "Package these commands into a tested skill"

Don't use for:
- One-off shell aliases or single-line commands with no multi-turn logic
- Router/hub skills that only point to other skills

## Prerequisites

- Standard Python 3.9+ environment.
- Access to the session's execution context or transcript file (`transcript.jsonl`).

## How to Run

Run the curation and validation helper through the `terminal` tool:

```bash
# Extract trajectory from a session transcript
python3 skills/software-development/skill-curator/scripts/curate_skill.py extract <path/to/transcript.jsonl> --json

# Validate an existing SKILL.md against repo standards
python3 skills/software-development/skill-curator/scripts/curate_skill.py validate skills/<category>/<name>/SKILL.md

# Scaffold a new skill directory
python3 skills/software-development/skill-curator/scripts/curate_skill.py scaffold <name> --category <category>
```

## Quick Reference

| Task | Command |
|---|---|
| Extract trajectory | `python3 skills/software-development/skill-curator/scripts/curate_skill.py extract <transcript.jsonl> [--json]` |
| Validate SKILL.md | `python3 skills/software-development/skill-curator/scripts/curate_skill.py validate <path/to/SKILL.md>` |
| Scaffold skill structure | `python3 skills/software-development/skill-curator/scripts/curate_skill.py scaffold <name> --category <cat>` |

## Procedure

### 1. Analyze the Session Trajectory
1. Extract user intent, key prompts, and tool sequences using `curate_skill.py extract`.
2. Filter out trial-and-error noise, dead ends, and transient errors, preserving only the optimal, verified execution path.
3. Identify required CLI commands, Hermes tool calls (`terminal`, `read_file`, `write_file`), and parameter defaults.

### 2. Scaffold and Author the Skill
1. Create the skill directory under `skills/<category>/<name>/` using `curate_skill.py scaffold <name> --category <category>`.
2. Draft `SKILL.md` ensuring:
   - Frontmatter starts at byte 0 with `---`.
   - `description` is under 60 characters, one sentence, ends with a period, and contains zero marketing buzzwords.
   - Author format credits the contributor: `Author Name (handle), Hermes Agent`.
   - Explicit sections: `## When to Use`, `## Prerequisites`, `## How to Run`, `## Quick Reference`, `## Procedure`, `## Pitfalls`, `## Verification`.
   - No machine-specific absolute user directories (use repo-relative or profile-aware paths).

### 3. Add Supporting Scripts & Tests
1. If the workflow requires multi-step logic, write a dedicated CLI script under `scripts/`.
2. Add automated unit tests under `tests/skills/test_<name>_skill.py` testing CLI operations and edge cases with mocks/fixtures.

### 4. Validate & Regenerate Docs
1. Verify authoring compliance: `python3 skills/software-development/skill-curator/scripts/curate_skill.py validate skills/<category>/<name>/SKILL.md`.
2. Run authoring tests: `pytest tests/skills/test_authoring_standards.py tests/skills/test_<name>_skill.py`.
3. Update docs via `python3 website/scripts/generate-skill-docs.py` and maintain scoped diffs.

## Pitfalls

- **Over-fitting to one session:** Generalize environment variables, paths, and identifiers rather than hardcoding session-specific values.
- **Description character limit:** Descriptions must strictly remain under 60 characters to fit the system prompt's skill index without truncation.
- **Dangling related skills:** Ensure all `related_skills` exist in the active repository tree.

## Verification

Validate the skill and run authoring tests:
```bash
python3 skills/software-development/skill-curator/scripts/curate_skill.py validate skills/software-development/skill-curator/SKILL.md
pytest tests/skills/test_authoring_standards.py
```
