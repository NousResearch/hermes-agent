---
title: "Hermes Agent Skill Authoring — Guide skill creation, updates, placement, and checks"
sidebar_label: "Hermes Agent Skill Authoring"
description: "Guide skill creation, updates, placement, and checks"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent Skill Authoring

Guide skill creation, updates, placement, and checks.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/hermes-agent-skill-authoring` |
| Version | `2.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `skills`, `authoring`, `maintenance`, `hermes-agent` |
| Related skills | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Agent Skill Authoring Skill

Create and maintain procedural knowledge that changes how Hermes handles a
repeatable class of work. Decide whether to update, create, or skip a skill;
route accepted work to the correct destination; and finish with evidence that
the instructions work. Do not use a skill to justify new core tooling when an
existing Hermes tool, CLI command, plugin, or MCP server already fits.

## When to Use

- The user asks to create, update, refactor, review, or publish a Hermes skill.
- A completed task reveals a reusable procedure, correction, or missing pitfall.
- A repository contribution adds or modernizes `skills/` or `optional-skills/`.
- An existing skill is stale, overlapping, oversized, platform-inaccurate, or
  ineffective in a real run.

Do not create a skill for a one-off answer, generic advice Hermes already
follows, or a workflow already owned by an existing skill. Installing an
already-published skill is a Skills Hub operation, not an authoring task.

## Prerequisites

- Establish the intended audience, persistence, and ownership: personal,
  bundled, official optional, or external/community.
- Resolve and record the exact canonical target path. Disambiguate same-name
  personal, bundled, optional, installed, and external sources before reading
  or writing any of them.
- Classify the source material and intended side effects: secrets, PII,
  private or customer URLs/data, proprietary content, license and attribution
  requirements, and any public, shared, paid, or otherwise external writes.
- Use `skills_list` to survey installed skills, then use `skill_view` to read
  every likely owner completely. Read a linked file before modifying it.
- In a repository checkout, use `search_files` across `skills/` and
  `optional-skills/`, read the repository instructions, and inspect two or
  three peers in the closest category.
- Search the Hub with `terminal` and `hermes skills search <query>` when a
  published skill may already cover the workflow.
- Read [references/authoring-contract.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/software-development/hermes-agent-skill-authoring/references/authoring-contract.md)
  before drafting frontmatter or reviewing a skill for merge.

## How to Run

Use the narrowest write path that preserves the correct source of truth.

| Goal | Preferred path |
| --- | --- |
| List installed candidates | `skills_list` |
| Read a candidate or support file | `skill_view` |
| Create a user-local skill | `skill_manage(action="create")` |
| Make a small installed-skill correction | `skill_manage(action="patch")` |
| Rewrite an installed skill | `skill_manage(action="edit")` after reading it |
| Create repository source | `write_file` under `skills/` or `optional-skills/` |
| Update repository source | `patch` after `read_file` |
| Run validators, helpers, and tests | `terminal` |

`skill_manage(action="create")` always creates under the active profile's
`$HERMES_HOME/skills/`; it does not create repository source. Update actions
resolve an existing installed or external-dir skill in place. In a source
checkout, address the repository path explicitly so an installed copy is not
mistaken for the source file.

## Quick Reference

### Create, update, or skip

| Evidence | Decision |
| --- | --- |
| A skill already owns the same trigger class and outcome | **Update** that skill. |
| No owner exists and the procedure is distinct, repeatable, and verified | **Create** a skill. |
| The task is one-off, generic, speculative, or already covered | **Skip** authoring. |
| Two narrow skills overlap and one can own both workflows coherently | **Consolidate**, then remove the duplicate deliberately. |

Prefer updating an umbrella skill over creating a competing sibling. A new
name is not a new behavioral boundary.

### Choose the destination

| Destination | Use it for |
| --- | --- |
| `$HERMES_HOME/skills/` | Personal, private, profile-specific, or prototype workflows. |
| `skills/<category>/<name>/` | Broad workflows useful to most Hermes users by default. |
| `optional-skills/<category>/<name>/` | Official but niche, paid, platform-specific, or dependency-heavy workflows. |
| External Hub or standalone repository | Community, organization-specific, or third-party integrations outside Hermes core maintenance. |

If instructions plus existing tools can express the workflow, keep it a
skill. Put repeated deterministic logic in `scripts/`. Prefer a plugin, MCP
server, or service-gated tool when the capability needs structured runtime
integration rather than procedural guidance.

## Procedure

1. **Discover ownership.** List installed skills, search both repository
   trees, and search the Hub. Read likely matches and nearby peers rather than
   deciding from names alone. Finish when no plausible owner remains unread.

2. **Choose update, create, consolidate, or skip.** Compare trigger,
   prerequisites, procedure, and output. Update on overlapping intent; create
   only for a distinct reusable behavior; skip no-op knowledge. Finish with a
   one-sentence rationale and a named owner or destination. For consolidation,
   migrate callers, reverse references, `related_skills` metadata, scheduled
   prompts, and generated or hand-written documentation before removing the
   duplicate, and preserve a retrievable backup.

3. **Select the source of truth.** Use the destination table and the user's
   requested scope. An explicit repository request targets repository source;
   a personal memory request targets the active profile. Finish when the write
   path cannot accidentally modify a seeded or installed copy instead.

4. **Set the safety and publication boundary.** Exclude secrets, PII, private
   or customer material, private URLs, and unlicensed proprietary content from
   the skill and its resources. Preview the exact destination and final diff;
   obtain explicit approval before public, shared, paid, costly, or other
   externally mutating actions.

5. **Design progressive disclosure.** Keep decisions and common procedure in
   `SKILL.md`. Move bulky or branch-specific facts to `references/`, repeated
   deterministic work to `scripts/`, reusable text/config to `templates/`, and
   output resources to `assets/`. Create only directories that are needed.

6. **Author the behavioral contract.** Follow the frontmatter and section
   contract in the linked reference. Use imperative instructions, native
   Hermes tool names, checkable completion criteria, and explicit failure
   handling. Delete stale wording that the new guidance replaces.

7. **Apply the smallest coherent change.** Use a targeted patch for a local
   correction and a full edit only when structure or intent changes broadly.
   Read the complete target before either operation. If creation or deletion
   was not explicitly requested, obtain user confirmation first.

8. **Validate at three levels.** Check frontmatter, naming, description, link
   targets, section order, and platform claims. Run every new helper script or
   a representative sample, then run the relevant repository tests through
   its required test wrapper. Finish only when each failure is fixed or
   reported with evidence.

9. **Forward-test behavior.** Point `HERMES_HOME` to a temporary isolated
   directory, then give a fresh session or subagent the skill and a realistic
   request without the intended answer or your diagnosis. Inspect whether it
   chooses the right action, follows the procedure, handles failure, and
   verifies the result. Do not use production credentials or shared state; ask
   before a test would be slow, costly, or externally mutating.

10. **Synchronize derived documentation.** Treat `SKILL.md` as the source. For
    in-repo skills, run `website/scripts/generate-skill-docs.py` normally, then
    run it again with `--check`; do not hand-edit generated catalog pages.
    Review the final diff so every changed file is intentional.

11. **Report the outcome.** State the create/update/skip decision, destination,
    files changed, validation performed, and any remaining limitation. Finish
    when another maintainer can reproduce the evidence without hidden context.

## Pitfalls

- **Duplicate by default:** creating before reading existing skills fragments
  ownership and produces conflicting triggers.
- **Wrong tree:** using `skill_manage(action="create")` for an in-repo request
  writes to the active profile instead of the checkout.
- **Ambiguous target:** editing by name alone can silently change an installed
  or personal copy instead of the intended canonical source.
- **Private material leakage:** examples, URLs, fixtures, and generated docs
  can expose secrets, PII, customer data, or unlicensed proprietary content.
- **Shared-state forward tests:** using the active profile, production
  credentials, or shared services makes tests destructive and irreproducible.
- **Validator/merge-policy confusion:** the runtime accepts legacy descriptions
  up to 1024 characters; new or modernized repository skills must satisfy the
  stricter 60-character hardline in the authoring contract.
- **Tool-name drift:** prose that leads with raw shell file operations teaches
  an interaction surface where Hermes native tools should be named.
- **Sediment:** adding a new rule without removing superseded prose makes the
  skill longer and internally inconsistent.
- **Unverified helpers:** a syntactically plausible script is not evidence that
  its real input/output contract works.
- **Leaky forward tests:** giving the evaluator the expected result tests prompt
  leakage rather than whether the skill generalizes.
- **Generated-file edits:** hand-editing generated catalog pages creates drift
  and will be overwritten by the generator.
- **Implicit deletion intent:** when consolidating, update the umbrella first
  and pass `absorbed_into=<umbrella>` to deletion; use an empty value only for
  deliberate pruning.

## Verification

- The decision is explicitly update, create, consolidate, or skip, with a
  behavioral reason.
- The exact canonical target path and source were recorded, with same-name
  sources disambiguated.
- The destination matches audience, dependency weight, and ownership.
- Sensitivity, licensing, and external-write classifications are complete;
  sensitive or unlicensed material is absent and required approvals are
  recorded.
- The complete target and every modified support file were read first.
- Frontmatter, description, name, modern section order, and platform gating
  satisfy the authoring contract.
- `SKILL.md` contains only always-needed guidance; linked resources resolve
  directly and do not duplicate it.
- New or changed helpers were exercised and relevant tests passed through the
  repository's required wrapper.
- A fresh-context forward test used a temporary `HERMES_HOME`, no production
  credentials, and no shared state for complex or failure-prone guidance.
- Generated documentation came from the source skill; normal generation and
  the following `--check` both passed, and the final diff contains no unrelated
  or stale changes.
