---
title: Autonomous Loop Contracts
---

# Autonomous Loop Contracts

Tier 6 makes recurring automation visible and quiet. It does not create recurring jobs.

Autonomous loops are useful only when they are self-contained, narrow, and boring when nothing changes. The audit surface inventories cron jobs and persisted `/goal` state, then reports metadata-only issue codes.

## Runtime surfaces

- Python: `agent.autonomous_loops.audit_autonomous_loops()`
- Harness facade: `HermesHarness().control_plane.autonomous_loops()`
- Dashboard/API: `GET /api/harness/autonomous-loops`

The response declares:

- `content_policy: metadata_only`
- `mode: audit_only_no_create`

That mode is a contract: the audit never creates, runs, pauses, resumes, updates, or deletes cron jobs/goals.

## What the audit inventories

| Surface | Evidence returned | Raw data excluded |
| --- | --- | --- |
| Cron jobs | counts by active/recurring/agent/no-agent/schedule/delivery class | job names, prompts, job IDs, script paths, explicit delivery IDs |
| Script-only watchdogs | count of `no_agent=True` jobs and missing-script issues | script contents and script path strings |
| Chained/profile/workdir jobs | counts only | upstream job IDs, profile names, workdir paths |
| `/goal` loops | goal row counts, status counts, turns/subgoal totals | goal text, subgoal text, session IDs |
| Guidance | static template/contract IDs | no runtime private content |

## Issue codes

| Issue code | Meaning |
| --- | --- |
| `loop_missing_silence_condition` | Active recurring loop lacks an obvious silence/no-change condition. |
| `loop_broad_delivery` | Active recurring loop delivers to `all`; fanout should be intentional and rare. |
| `loop_side_effect_policy_missing` | Active recurring loop has side-effect-capable toolsets without an explicit approval/read-only/dry-run policy. |
| `loop_tool_scope_unbounded` | Active recurring agent job has no per-job `enabled_toolsets`, so it inherits broad cron defaults. |
| `loop_no_agent_missing_script` | Script-only watchdog mode is enabled but no script is configured. |

## Prompt templates / contracts

### Agent cron job

A recurring LLM-driven cron prompt should include:

1. A self-contained task description.
2. The exact source to inspect.
3. The threshold for reporting.
4. A silence condition, usually: `If there is nothing new/actionable, respond exactly [SILENT].`
5. Toolset scope via `enabled_toolsets` whenever possible.
6. Approval constraints for side effects: read-only by default, ask before external/irreversible actions.

### Script-only watchdog

Use `no_agent=True` when the script itself can produce the final message.

Rules:

1. `script` is required.
2. Non-empty stdout is delivered verbatim.
3. Empty stdout is silent.
4. Non-zero exit alerts the user.
5. The script should stay quiet when there is nothing to report.

### Side-effect rule

Recurring jobs should not perform external, destructive, purchasing, publishing, or credential-affecting actions without explicit approval. Prefer read-only checks, dry-runs, or single-shot user-approved jobs for risky work.

## Verification

Focused Tier 6 gate:

```bash
scripts/run_tests.sh tests/agent/test_autonomous_loops.py \
  tests/agent/test_hermes_harness.py::test_control_plane_harness_exposes_autonomous_loops \
  tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_harness_trace_replay_endpoints_are_content_safe -q
```
