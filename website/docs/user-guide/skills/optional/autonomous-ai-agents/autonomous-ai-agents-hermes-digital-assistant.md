---
title: "Hermes Digital Assistant — Upgrade Hermes into a persistent personal assistant"
sidebar_label: "Hermes Digital Assistant"
description: "Upgrade Hermes into a persistent personal assistant"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Digital Assistant

Upgrade Hermes into a persistent personal assistant.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/hermes-digital-assistant` |
| Path | `optional-skills/autonomous-ai-agents/hermes-digital-assistant` |
| Version | `0.1.0` |
| Author | Daniel Steele (keeltrace), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `hermes`, `agents`, `personal-assistant`, `automation` |
| Related skills | [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Digital Assistant Skill

Turn the current Hermes installation into a persistent personal digital assistant that implements the supplied method. Treat `references/DIGITAL_ASSISTANT.md` as the behavioral specification and adapt it to the Hermes version actually present. Do not merely follow the method for the current turn: persist the upgrade through supported Hermes configuration and extension seams so the behavior remains after this skill unloads.

## When to Use

- Use when the user asks to install, enable, repair, upgrade, audit, or verify Hermes Digital Assistant behavior.
- Use when the user says to make stock Hermes behave like the method in `references/DIGITAL_ASSISTANT.md`.
- Use when a previous HDA installation exists and needs migration to a newer Hermes version.
- Do not use for ordinary personal-assistant tasks after HDA is already working; the installed behavior should handle those without this skill.
- Do not use as an excuse to rewrite Hermes core when configuration, profile state, hooks, plugins, or existing tools can satisfy the requirement.

## Prerequisites

- A Hermes installation that can inspect and modify its own profile or installation with the tools available in the current session.
- Permission from the user to upgrade this Hermes installation. Treat that as authorization for reversible, backed-up Hermes-local configuration and extension changes required by this skill.
- Do not infer authorization for purchases, external messages, deletion of user data, credential changes, remote repository pushes, or unrelated service changes.

Before changing anything, load all of these references with `skill_view`:

- `references/DIGITAL_ASSISTANT.md` — authoritative behavioral specification.
- `references/UPGRADE_PLAYBOOK.md` — adaptive implementation order and preservation rules.
- `references/ACCEPTANCE_TEST.md` — evidence required before reporting completion.
- `references/ENGINEERING_DISCIPLINE.md` — build and review discipline.

## How to Run

The canonical invocation is conversational: the user loads this skill and asks Hermes to upgrade itself, repair an HDA installation, or verify that HDA is still correctly installed.

Use Hermes-native tools to do the work:

- `read_file` and `search_files` to inspect source, profile state, configuration, docs, and existing extensions.
- `terminal` only for supported Hermes CLI operations, tests, version control inspection, or commands with no native tool equivalent.
- `write_file` and `patch` for narrow, reviewable local changes.
- `session_search` or memory tools when validating retrieval behavior.
- `delegate_task` for an independent falsification pass when available.

Never tell the user to manually perform steps that Hermes can safely perform itself.

## Quick Reference

- **Spec authority:** `references/DIGITAL_ASSISTANT.md`.
- **Implementation order:** native config → existing profile/context state → existing Hermes hooks/plugins/tools → generated user-local extension → core source change as a last resort.
- **State authority:** inspect the running installation; never assume paths, versions, commands, or capabilities from this skill.
- **Preservation:** back up or checkpoint anything changed; preserve unrelated local work.
- **Activation:** do not restart or interrupt active work unnecessarily. If activation requires a restart that would interrupt a live task, checkpoint first and obtain explicit permission unless the user already requested immediate activation.
- **Completion:** evidence decides. Passing prose is not a passing installation.

## Procedure

### 1. Establish current reality

Inspect the Hermes instance before designing the upgrade. Record at minimum:

- Hermes version or exact source revision when available.
- Active profile and resolved Hermes home/config location.
- Whether the running process is source-based, packaged, containerized, remote, or otherwise managed.
- Existing memory/context facilities and session-history search.
- Existing system-prompt or context-extension seams.
- Plugin/hook APIs and human-approval mechanisms.
- Busy-input behavior and whether follow-ups queue, steer, or interrupt.
- Available task/Kanban/cron/state facilities for open loops and conditional triggers.
- Active messaging platforms and shared-chat admission/response controls.
- Current local modifications, user plugins, context files, and configuration that must be preserved.

**Completion criterion:** every implementation decision in the gap analysis can cite an observed capability or an explicitly documented absence.

### 2. Build a requirement-to-mechanism map

Read every pillar, vein, the pre-send gate, implementation notes, and build order in the specification. For each requirement, classify current state as:

- `native` — Hermes already satisfies it; configure or verify only.
- `partial` — Hermes provides the mechanism but needs configuration or a thin extension.
- `missing` — a new local extension or, only if unavoidable, a core change is required.
- `unverified` — evidence is insufficient; inspect further before implementing.

Prefer one deep mechanism that satisfies several requirements over many prompt fragments or duplicated subsystems.

**Completion criterion:** all five pillars, all eight veins, the pre-send gate, and the three implementation-note behaviors have a mapped mechanism and verification path.

### 3. Apply the build order from the specification

Implement in this order unless current Hermes already supplies a later dependency:

1. Pinned rules block plus durable context storage.
2. Retrieval loop.
3. Verification rubric.
4. Ask-vs-act gate with action-level reversible/irreversible enforcement.
5. Durable day board/open-loop state with conditional triggers.
6. Pre-send/attention gate.

Also ensure the implementation-note behaviors:

- rules survive new sessions and compression,
- shared chats do not trigger replies merely because messages exist,
- new incoming user messages queue as follow-ups instead of implicitly cancelling active work.

**Completion criterion:** every changed mechanism is persistent across a fresh Hermes session and does not depend on this skill remaining loaded.

### 4. Use the narrowest stable Hermes seam

For every change, prefer in order:

1. Existing Hermes configuration.
2. Existing profile/context/memory files or structured state.
3. Existing hooks, plugin APIs, task/Kanban/cron facilities, and approval gates.
4. A generated user-local Hermes extension owned by the local installation.
5. A narrow source modification only when the current Hermes version exposes no stable extension seam.

Do not create a new external repository, network service, package, database server, or maintenance dependency merely to implement HDA. Local files or a local SQLite database are acceptable when Hermes has no native structured store.

**Completion criterion:** each non-native component has a written reason why a narrower existing mechanism was insufficient.

### 5. Preserve the installation

Before a write:

- capture the current config/state needed to roll back,
- preserve unrelated uncommitted source changes,
- avoid destructive resets, cleans, or blanket overwrites,
- keep user memories, credentials, sessions, and existing extensions intact unless the specification explicitly requires migration,
- make repeated installation or repair idempotent where practical.

If the live installation already contains a prototype or older HDA behavior, reconcile it instead of layering duplicate mechanisms.

**Completion criterion:** the pre-upgrade state has a recovery path and no unrelated local work disappears from the diff or state inventory.

### 6. Verify each vertical slice before continuing

After each pillar or cross-cutting gate is implemented, run its smallest objective test before adding the next layer. Use the done-tests in `references/DIGITAL_ASSISTANT.md` and the consolidated scenarios in `references/ACCEPTANCE_TEST.md`.

Do not substitute source inspection for behavior tests when behavior can be exercised. Do not substitute a successful command exit for verifying the intended state.

**Completion criterion:** every implemented slice has positive evidence and any regression is repaired before the next slice begins.

### 7. Run the full acceptance suite

Run `references/ACCEPTANCE_TEST.md` against the upgraded Hermes. Use disposable/synthetic facts, messages, loops, approvals, and stale values where the test would otherwise touch real people, money, bookings, credentials, or destructive actions.

A test that cannot safely be exercised must be marked `UNVERIFIED` with the exact blocker; never infer a pass.

**Completion criterion:** all material acceptance cases are `PASS`, or the installation is reported `PARTIAL` with failed/unverified cases named.

### 8. Attack the completion claim independently

Use `delegate_task` for a reviewer when available. Give it the specification, requirement map, changed-state evidence, and test results, and ask it to falsify the claim that HDA is complete. The reviewer should search for:

- a requirement satisfied only by prompt text when a persistent gate/state mechanism was required,
- behavior lost after session reset or compression,
- stale facts treated as current,
- preferences treated as grants,
- irreversible actions without exact approval,
- open loops that vanish or nag,
- shared-chat noise,
- incoming messages that still kill active work,
- claimed verification that checked only process success rather than resulting state.

If delegation is unavailable, perform a fresh review pass separated from implementation and explicitly try to disprove each material claim.

**Completion criterion:** every credible finding is repaired or recorded as a blocker before final status.

### 9. Activate carefully

If changes require a reload/restart, verify that no important work will be lost. Checkpoint active work first. Restart only the minimum Hermes component needed, then re-run the persistence and smoke checks against the restarted process.

**Completion criterion:** the running Hermes process is confirmed to be using the upgraded state, not merely files written on disk.

### 10. Report evidence, not confidence

Finish with exactly one status:

- `DONE` — all material scope is implemented and sufficiently verified.
- `PARTIAL` — useful progress exists but material behavior is missing or unverified.
- `BLOCKED` — no safe implementation path remains without an external dependency or user decision.
- `FAILED` — the attempted upgrade did not achieve the goal.

Report:

- mechanisms installed or configured,
- exact verification performed,
- preservation/rollback state,
- failed or unverified acceptance cases,
- whether a restart occurred and whether the running process was reverified,
- one next action only when something remains.

## Pitfalls

- **Prompt-only installation:** loading the method into the current conversation is not a persistent upgrade.
- **Frozen implementation recipe:** do not assume a particular Hermes version, file path, hook name, or source layout when the running installation can be inspected.
- **Duplicate plumbing:** do not build a second memory, task board, approval system, or scheduler when Hermes already has a suitable one.
- **Preference becomes permission:** only an explicit user grant authorizes the exact action shape it covers.
- **Core-first patching:** source edits are the last resort, not the default.
- **Self-certification:** the builder's own explanation is not verification; attack the result with observable tests.
- **Restart equals success:** after restart, confirm the live process loaded the new behavior.
- **Over-notifying:** HDA itself must not create a stream of status chatter while installing the system that is supposed to reduce status chatter.

## Verification

An HDA upgrade is complete only when:

1. The full requirement map is accounted for.
2. The method's pillar and vein tests pass or are explicitly marked unverified.
3. Pinned rules and relevant context survive a fresh session and compression/rebuild path.
4. Stale volatile values are rechecked rather than repeated from memory.
5. Load-bearing ambiguity and irreversible actions stop at the approval boundary.
6. Open loops survive restart, carry next moves/dependencies/triggers, and do not nag without change.
7. Shared-chat and outgoing-message gates suppress noise.
8. New user input does not implicitly cancel active work.
9. Independent review finds no unresolved material contradiction.
10. The currently running Hermes instance, not merely its files, is verified to be using the upgrade.
