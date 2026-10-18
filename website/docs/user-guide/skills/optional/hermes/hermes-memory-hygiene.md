---
title: "Memory Hygiene — Memory hygiene across MEMORY.md, Mnemosyne, LCM, skills"
sidebar_label: "Memory Hygiene"
description: "Memory hygiene across MEMORY.md, Mnemosyne, LCM, skills"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Memory Hygiene

Memory hygiene across MEMORY.md, Mnemosyne, LCM, skills.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/hermes/memory-hygiene` |
| Path | `optional-skills/hermes/memory-hygiene` |
| Version | `0.1.0` |
| Author | Ben Barclay (benbarclay), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `memory`, `mnemosyne`, `lcm`, `maintenance`, `hygiene`, `skills-audit` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Memory Hygiene Skill

Routine maintenance across four memory layers: user-profile MEMORY.md, Mnemosyne private bank, LCM compression database, and the skill library. Keeps each layer healthy, bounded, and audit-ready.

## When to Use

- Scheduled hygiene pass (weekly or on-demand)
- MEMORY.md over budget (≥1200 bytes)
- Mnemosyne warnings on vec_working or recall health
- LCM fragmentation or DB size growth
- Skills library audit overdue
- Session cleanup needed

Don't use for:
- One-off memory queries — use mnemosyne_recall or session_search directly
- Live incident response — this is maintenance, not triage

## Prerequisites

- Hermes Agent with mnemosyne_* tools available
- lcm_doctor and lcm_status on PATH (from lcm package)
- Git for skill library operations

## How to Run

Load the skill and execute its procedure. No standalone CLI.

## Quick Reference

| Action | Tool |
|---|---|
| MEMORY.md prune | `read_file` + `memory` (patch) |
| Background review queue check | `/memory pending` via `terminal` |
| Mnemosyne stats | `mnemosyne_stats` |
| Mnemosyne diagnose | `mnemosyne_diagnose` |
| Mnemosyne recall diagnostics | `mnemosyne_recall_diagnostics` |
| Mnemosyne sleep (consolidate) | `mnemosyne_sleep` |
| LCM doctor | `terminal(command="lcm_doctor")` |
| LCM status | `terminal(command="lcm_status")` |
| Skills list | `skills_list` |
| Session search | `session_search` |

## Procedure

### 1. MEMORY.md prune
- Read `~/.hermes/memories/MEMORY.md`
- Classify each entry: keep only if machine-specific, non-derivable, failure-repeating
- Route displaced content: preferences → Mnemosyne canonical; procedures → skills; progress → session history
- Compress survivors to `symptom/trap → consequence → exact fix` format
- Verify size ≤ 1200 bytes and readability

### 1b. Background-review staged writes (the approval gate)
Hermes runs a background self-improvement review after turns (a forked review agent). A fail-closed guard (`tools/memory_tool.py`, `_background_delete_gate`) forbids that unattended fork from applying destructive memory ops: `add` applies automatically, but any `replace`/`remove` — single or inside a batch — is staged into `~/.hermes/pending/memory/` for human approval instead of being applied or dropped.

- Inspect the gate state and the queue: `/memory` (gate + list), `/memory pending` (all records). Apply/reject one or many: `/memory approve <id>` / `/memory reject <id>`, or `... all`. `approval on|off` toggles the gate; `memory.write_approval` defaults to `false` (off) but the background-review gate is independent of that setting and always stages.
- A silent backlog is the failure mode. The review's own summary is never published back, so staged records accumulate unreviewed and nothing surfaces them — check the directory mtime/count during any hygiene pass, not only when the user asks.
- NEVER `approve all` blind. A memory `replace` overwrites the WHOLE matched entry: a lossy consolidation silently drops a clause from a fact. The approver is the last person who can notice — review each destructive record, or bulk-reject the stale ones (reject only discards a proposal; it never touches live memory).
- Skill patches from the same review are NOT gated the same way — they apply directly. Audit them separately (`/skills pending` should be empty); skills are not version-controlled, so a bad patch is fixed by re-patching, not reverting.
- Records staged before entry pinning carry no `matched_entry` and are refused at apply time ("predates entry pinning") — reject and recreate them rather than forcing them.

### 2. Mnemosyne health check
```bash
mnemosyne_stats
mnemosyne_diagnose
mnemosyne_recall_diagnostics
```
- If `repair_vec_working` needed, run `mnemosyne_diagnose` with `repair_vec_working=true`
- Verify repair succeeded by re-running `mnemosyne_diagnose` and confirming zero warnings
- Consolidate old working memories: `mnemosyne_sleep` with `all_sessions=false`, `force=false`
- Verify sleep completed without error
- Store durable lessons as canonical, not working

### 3. LCM DB maintenance
```bash
lcm_doctor
lcm_status
```
- If fragmentation warnings: run lifecycle repair
- If DB size large and compression advancing: consider threshold full sweep
- Never delete `lcm.db`; backup before sweep

### 4. Skills audit
- List all skills: `skills_list`
- Flag: pinned skills, stale skills, duplicate coverage, missing triggers, oversized SKILL.md (>150 lines)
- Cross-reference with actual usage in recent sessions
- Propose prune/merge/update actions
- Verify audit produced at least one actionable recommendation or explicit "no action needed"

### 5. Session cleanup
- List recent sessions: `session_search` with `limit=20` or `terminal(command="lcm_recent", args=["period=7d"])`
- Archive or prune stale sessions per `session-librarian` pattern
- Remove orphaned externalized payload refs if any
- Verify cleanup produced zero orphaned refs or documented exceptions

### 6. Weekly review (optional, if triggered)
- Use `weekly-review-planning` structure adapted to memory/projects
- Review stalled work, commitments, next actions
- Update STATUS.md for active projects
- Verify STATUS.md updated or weekly review explicitly skipped with reason

## Hard-gate
Do NOT declare hygiene complete until:
- MEMORY.md ≤ 1200 bytes verified
- Mnemosyne stats healthy, vec_working repaired if needed
- LCM DB fragmentation addressed or explicitly deferred
- Skills audit complete with actionable recommendations
- Session cleanup complete with zero orphaned refs or documented exceptions
- Weekly review completed or skipped with reason

## DoD
- [ ] MEMORY.md ≤ 1200 bytes, entries pass admission test
- [ ] Mnemosyne stats healthy, vec_working repaired if needed
- [ ] LCM DB fragmentation addressed or explicitly deferred
- [ ] Skills audit complete with prune/merge/update actions listed
- [ ] Session cleanup done or explicitly skipped with reason
- [ ] If weekly review: stalled items flagged, next actions named

## Pitfalls
- Don't conflate `MEMORY.md` with Mnemosyne — they serve different purposes
- Don't delete `lcm.db` — it's runtime state, not cache
- Don't auto-prune without checking pinned status and active usage
- Don't run LCM full sweep without backup
- Skills >150 lines need splitting; don't let them drift
- Char-budget memory batches are all-or-nothing — pass `nil`, then measure, then add. When a user-profile memory `replace/add` batch keeps failing by a shrinking margin (1477/1375, then 1458, then 1415, then 1391 — the edits are converging but each retry restarts from zero), stop fighting the arithmetic. The batch semantics are atomic: any failing op voids all others, so a batch of PURE `remove`/`replace-shrink` ops (which only free space) is always appliable on its own. Fixed sequence: (1) one batch of nothing but shrinks/removes; (2) read the reported usage; (3) one add/replace that fits the reported room with margin to spare. Rule of thumb: if the replacement text is longer than the text it replaces, the net is an add — budget it as one; do not assume "replace" is neutral just because the entry still exists.
- Per-profile memory slots do NOT cross profiles. Each Hermes profile (default, hr-server, teammate profiles) has its own `memories/MEMORY.md` and `memories/USER.md`. A correction written to default's MEMORY.md is invisible to hr-server. When work done in one profile must be visible to another, write a timestamped handoff file (e.g. `BetterAsk-Handoff_YYYY-MM-DD-TOPIC.md`) to the target profile's root, then add a one-line pointer in the target profile's MEMORY.md referencing the handoff path. Mnemosyne `remember` calls also do NOT cross profiles — Mnemosyne banks are isolated by profile. Treat cross-profile state as a file-copy problem, not a memory-save problem.
- A staged background-review write is not a pending decision you may ignore. It is the only surfacing path for the review's destructive consolidations, so an unread `~/.hermes/pending/memory/` queue means the review's findings are silently dropped. Treat a growing queue as a hygiene finding, and review destructive records one by one (a `replace` overwrites the whole entry).
- Mnemosyne `remember` is not the same write as user-profile `memory add`. The Mnemosyne private bank has no hard char budget and returns a memory_id on success; the user-profile memory slot is char-capped (e.g. 1,375 chars) and rejects writes after the cap with an error message. When both are available, write the canonical fact to Mnemosyne AND a short pointer to the user-profile slot — but if the user-profile slot is over budget, Mnemosyne alone is the durable carrier. Never claim a fact is "saved" if you only verified Mnemosyne and not the file system, or vice versa.

## Red flags
- "I'll just run lcm_doctor and call it done" — misses Mnemosyne and skills audit; this skill covers all four layers.
- "Skills audit is optional, we're just pruning memory" — the skill is hygiene, not just MEMORY.md.
- "I'll delete old sessions without checking Mnemosyne recall" — session cleanup can lose durable context; verify before pruning.
- "lcm.db is just a cache" — it's runtime state; deletion loses compression history.

## Verification
```bash
python3 -c "from pathlib import Path; p=Path.home()/'.hermes/memories/MEMORY.md'; s=p.read_text(); assert p.stat().st_size <= 1200; print(f'ok: {p.stat().st_size} bytes')"
mnemosyne_stats
lcm_doctor
```
