---
sidebar_position: 12
title: "Kanban Task-Scoped Skill Injection"
description: "RFC for resolving required task skills through profile-local roots plus approved read-only task overlays"
---

# Kanban Task-Scoped Skill Injection

Status: RFC / implementation design

Related: [#33245](https://github.com/NousResearch/hermes-agent/issues/33245), [#33640](https://github.com/NousResearch/hermes-agent/pull/33640)

## Problem

Kanban tasks can attach `skills` that must be preloaded into the spawned worker. Today those skills are resolved by the worker profile at CLI startup. If a task names a skill that the worker profile cannot resolve, the child process can fail before the agent loop can read the task, explain the setup problem, or block cleanly.

PR #33640 is the narrow safety fix for that failure class: preflight required `task.skills`, optionally run one local allowlisted profile-skill sync, and block before spawn when a required skill is still missing. That protects the queue from crash loops, but it still uses permanent profile mutation as the only positive availability mechanism. Useful, but not exactly elegant; the gods did give us filesystems so we would not copy the same skill into every profile forever.

This RFC defines the broader model requested in #33245: make task-specific skills available for a single Kanban worker run through approved read-only skill roots, without depending on local Hermes OS overlays or permanently copying skills into profile homes.

## Goals

- Treat `task.skills` as required by default and preflight them before worker spawn.
- Allow a required task skill to resolve from either the target profile's normal skill roots or an approved task-scoped read-only overlay root.
- Keep overlay roots task-scoped: available only to the spawned worker process for that task and run.
- Make overlay roots read-only from the worker's perspective where the backend can enforce it, and non-mutating in Hermes code where the OS/backend cannot.
- Define deterministic resolver precedence and collision behavior.
- Record provenance for every required skill that was injected.
- Leave room for future `optional_skills` that may warn/drop instead of block.
- Avoid any dependency on Hoang-local Hermes OS overlay scripts, profile sync policies, or private skill catalogs.

## Non-goals

- Do not replace `skills.external_dirs`; this feature composes with it.
- Do not expose a user's default/private skill library to every worker automatically.
- Do not prompt-paste arbitrary skill bodies into the Kanban task body.
- Do not silently install, copy, symlink, or mutate skills in the worker profile by default.
- Do not solve orchestrator skill selection in the same implementation. A catalog/capability API can choose task skills later; this RFC covers safe availability and provenance.

## Current behavior and #33640 comparison

The current activation path is:

1. Dispatcher claims a Kanban task.
2. Dispatcher resolves the assignee profile home.
3. Dispatcher builds a child command like `hermes -p <profile> --skills kanban-worker --skills <task-skill> chat -q "work kanban task <id>"`.
4. The child CLI loads the profile's configured skill roots.
5. Missing required skills fail at startup unless preflight blocks first.

PR #33640 improves step 3 by checking `task.skills` against the worker's effective profile-local roots before `Popen`. It also adds a deliberately local, allowlisted profile-skill sync escape hatch. That escape hatch must remain a local policy mechanism, not the upstream long-term product model.

Task-scoped injection changes the positive path:

1. Dispatcher resolves profile-local roots.
2. Dispatcher resolves approved task overlay roots for this task.
3. Dispatcher resolves each required skill against the combined effective roots.
4. Dispatcher records which root supplied each skill.
5. Dispatcher spawns the worker with only the necessary overlay roots exported for that run.
6. Worker loads the requested skills normally through the standard skill resolver.

If overlays are disabled or no overlay resolves a required skill, #33640-style fail-closed blocking remains the fallback.

## Configuration model

Task overlays should be opt-in and allowlisted in config. Suggested shape:

```yaml
kanban:
  task_skill_overlays:
    enabled: false
    allowed_roots:
      - path: ~/.hermes/shared-skills
        name: shared
        read_only: true
        allowed_profiles: [reviewer, backend, frontend, pm]
      - path: ~/.hermes/skill-bundles/review
        name: review-bundle
        read_only: true
        allowed_profiles: [reviewer, code-reviewer]
    collision_policy: profile_wins
    allow_task_roots: false
```

Rules:

- `enabled` defaults to `false`.
- `allowed_roots` is the only upstream source of overlay roots by default.
- `allow_task_roots` defaults to `false`; if later enabled, explicit task root paths must still be inside an allowlisted parent root.
- Paths are expanded with `~` and environment variables, normalized with `resolve()`, and rejected if missing or not directories.
- A root's `allowed_profiles` limits which assignees can receive that root.
- `read_only` defaults to `true`; non-read-only roots should be rejected unless a future explicit unsafe mode exists.

This keeps Hermes upstream generic. Local operators may point `allowed_roots` at their own shared skill bundle, but the dispatcher does not know or care about Hermes OS overlay machinery.

## Task model

Current `Task.skills` remains the required skill list.

Future schema extension:

```python
@dataclass
class Task:
    skills: Optional[list[str]] = None          # required
    optional_skills: Optional[list[str]] = None # best-effort, future
```

Semantics:

- `skills`: required. Missing after profile + overlay resolution blocks before spawn.
- `optional_skills`: best-effort. Missing optional skills produce an audit/provenance warning and are dropped from the child command.
- Both fields use skill identifiers, not arbitrary prose.
- Plugin-qualified skills such as `plugin:skill` stay in the plugin namespace and are not resolved through filesystem overlay roots.

## Resolver order and collision behavior

Resolution should be deterministic and evidence-producing. Suggested order:

1. Plugin-qualified skills (`plugin:skill`) resolve through plugin skill machinery.
2. Profile-local skills under the target profile's `skills/` root.
3. Profile-configured `skills.external_dirs`.
4. Task-scoped overlay roots from `kanban.task_skill_overlays.allowed_roots` that are allowed for the assignee profile.

Default collision policy: `profile_wins`.

- If the same skill name exists in profile-local roots and overlay roots, the profile-local copy is used.
- If the same skill name exists in multiple overlay roots, block as ambiguous unless the config explicitly orders roots and sets `collision_policy: first_match`.
- If a relative path skill id such as `software-development/systematic-debugging` resolves to multiple distinct `SKILL.md` files, apply the same policy.
- Frontmatter `name` matches count as aliases but must not hide a directory/path collision. If two `SKILL.md` files advertise the same frontmatter name under effective roots, block unless `profile_wins` picks a profile-local copy over overlays.

The preflight output should not just return `True` or `False`; it should return a resolution record:

```python
@dataclass(frozen=True)
class SkillResolution:
    requested: str
    resolved_name: str
    source_kind: Literal["plugin", "profile", "external_dir", "task_overlay"]
    root: Optional[Path]
    skill_path: Optional[Path]
    read_only: bool
```

That record drives spawn env, audit events, and block messages.

## Read-only enforcement

The dispatcher must avoid mutating overlay roots. Enforcement should be layered:

- Hermes code: never call `skill_manage`, sync, copy, delete, or curator operations against task overlay roots during dispatch.
- Resolver: overlay roots are read-only sources; writes target only the active profile's normal local skill root unless the user explicitly uses an existing `skills.external_dirs` mutation path.
- Local backend: if possible, pass overlay roots through env only and do not bind writable mounts specially.
- Docker/remote backends: mount task overlay roots read-only when the backend supports mount options.
- Fallback: if the backend cannot hard-mount read-only, still treat roots as non-mutating in Hermes and expose provenance so operators can audit misuse.

The important invariant is that a Kanban task can borrow procedure, not acquire ownership of the library.

## Worker spawn contract

The child process needs to see overlay roots only for this task. Suggested env var:

```bash
HERMES_TASK_SKILL_OVERLAY_DIRS=/abs/shared-skills:/abs/review-bundle
```

Then the skill resolver should include those roots after normal profile roots. This is separate from `HERMES_SKILLS_EXTERNAL_DIRS` because external dirs are profile/runtime configuration, while task overlay dirs are per-dispatch provenance-controlled injection.

The dispatcher should also pass the exact requested skills as normal `--skills` flags after preflight succeeds:

```bash
hermes -p reviewer --skills kanban-worker --skills release-gate-checklist chat -q "work kanban task t_..."
```

No prompt-body paste. No permanent profile sync. No little filesystem goblin secretly copying folders at 3am.

## Required skill preflight

Preflight algorithm:

1. Normalize and dedupe `task.skills` while preserving order.
2. Ignore `kanban-worker` for missing-skill purposes because it is dispatcher-added opportunistically and the lifecycle guidance is already in the system prompt.
3. Build effective roots for the assignee:
   - profile-local root;
   - profile `skills.external_dirs`;
   - allowed task overlay roots.
4. Resolve each required skill into exactly one `SkillResolution`.
5. If a required skill is missing, block before spawn with a deterministic setup reason.
6. If a required skill is ambiguous, block before spawn with the candidate roots in the reason/audit event.
7. If all required skills resolve, spawn the worker with task overlay roots exported and `--skills` flags preserved.

Suggested block reason:

```text
missing forced skill(s) for profile 'reviewer': release-gate-checklist. Configure kanban.task_skill_overlays.allowed_roots or install/sync the skill into the target profile before unblocking this task.
```

Suggested ambiguity reason:

```text
ambiguous forced skill for profile 'reviewer': release-gate-checklist resolves in multiple task overlay roots: shared, review-bundle. Set collision_policy or remove the duplicate before unblocking.
```

## Provenance and audit events

For every spawned task with required or optional skills, record an event before spawn:

```json
{
  "kind": "skill_resolved",
  "task_id": "t_...",
  "profile": "reviewer",
  "required": [
    {
      "requested": "release-gate-checklist",
      "resolved_name": "release-gate-checklist",
      "source_kind": "task_overlay",
      "root_name": "review-bundle",
      "root": "/home/user/.hermes/skill-bundles/review",
      "skill_path": "/home/user/.hermes/skill-bundles/review/devops/release-gate-checklist",
      "read_only": true
    }
  ],
  "optional_missing": []
}
```

This should be stored in the Kanban event log, not only stderr. Operators need durable evidence of which external procedure a worker saw.

## Future optional skills

`optional_skills` should use the same resolver and provenance path but different failure semantics:

- Missing optional skill: record `optional_missing`, do not pass `--skills`, continue spawn.
- Ambiguous optional skill: record ambiguity warning, do not pass `--skills`, continue spawn unless a future strict mode says otherwise.
- Present optional skill: pass `--skills <name>` and record provenance.

Required and optional lists should be separately visible in `kanban_show` so workers and reviewers can distinguish "must-have procedure" from "nice-to-have context".

## Implementation sketch

Likely code touch points:

| Area | File | Change |
|---|---|---|
| Config defaults | `hermes_cli/config.py` | Add `kanban.task_skill_overlays` defaults. |
| Task schema | `hermes_cli/kanban_db.py` | Add future `optional_skills` field and persistence helpers when implemented. |
| Resolver | `agent/skill_utils.py` or a small new module | Add root-aware skill resolution returning `SkillResolution` records. |
| Dispatcher | `hermes_cli/kanban_db.py` | Replace boolean `_skill_available` preflight with resolution records and overlay env injection. |
| Prompt builder / skill loading | `agent/prompt_builder.py` and `tools/skills_tool.py` | Include `HERMES_TASK_SKILL_OVERLAY_DIRS` in read-only scan roots after normal roots. |
| Remote/backend support | `tools/credential_files.py` / terminal backend plumbing | Ensure read-only overlay roots can be exposed to workers without copying or writable mounts where supported. |
| Tests | `tests/hermes_cli/test_kanban_core_functionality.py`, `tests/agent/test_external_skills.py`, `tests/tools/test_skills_tool.py` | Cover required preflight, overlay resolution, collisions, provenance, optional-skill warnings. |

## Test matrix

Minimum tests for the implementation PR:

1. Required skill in profile-local root spawns without overlay env.
2. Required skill missing locally but present in an allowed overlay root spawns with `HERMES_TASK_SKILL_OVERLAY_DIRS`.
3. Required skill missing everywhere blocks before spawn.
4. Required skill in two overlay roots blocks as ambiguous by default.
5. Required skill in profile-local root and overlay root uses profile-local under `profile_wins`.
6. Plugin-qualified skill is not resolved through filesystem roots and is passed through.
7. Relative path skill id resolves deterministically.
8. Frontmatter `name` aliases are detected, including duplicate-name ambiguity.
9. Overlay root not allowlisted for the assignee profile is ignored.
10. Nonexistent or non-directory overlay root is ignored or config-warned, never treated as a match.
11. `optional_skills` missing records provenance warning and does not block once that field exists.
12. Kanban event log includes provenance for profile, external-dir, and overlay-resolved skills.
13. Worker cannot mutate an overlay root through dispatcher/sync paths.
14. Existing #33640 missing-required blocking behavior remains unchanged when overlays are disabled.

## Rollout plan

1. Land #33640 or equivalent preflight first to remove crash loops.
2. Add resolver records without overlays and keep behavior equivalent.
3. Add disabled-by-default `kanban.task_skill_overlays` config.
4. Enable overlay roots in preflight and child env only when configured.
5. Add provenance events.
6. Add optional-skills persistence and warning semantics as a separate follow-up.
7. Add catalog/capability selection as a separate feature, likely sharing the same resolution/provenance schema.

## Open questions

- Should `HERMES_TASK_SKILL_OVERLAY_DIRS` be colon-separated, JSON, or both for Windows path safety?
- Should collision policy be global, per-root, or per-profile?
- Should optional skills be persisted in the same `tasks` table JSON payload as `skills`, or split into first-class columns during a schema migration?
- How should remote backends expose read-only roots when the root lives on the dispatcher host but the worker runs in a container or SSH target?
- Should `skill_manage` explicitly reject writes to task overlay roots even if a user passes a path/name that resolves there?

## Acceptance checklist

- Required skill preflight: specified.
- Read-only roots: specified, including layered enforcement and backend caveat.
- Resolver precedence/collision behavior: specified.
- Provenance/audit: specified with event shape.
- Future `optional_skills`: specified.
- No local Hermes OS overlay dependency: explicit non-goal and config model.
