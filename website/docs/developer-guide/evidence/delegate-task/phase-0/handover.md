Phase 0 handover to Phase 1

Phase 0 status: GO; independently audited
Frozen rollback point: 98a4aa453c1c576798a4461d5b2902d805eef71d
Canonical repository: /home/kensei/repos/KenseiAgent

Phase 1 is authorised only in the isolated worktree after the recorded Phase 0 GO.

After GO, create exactly one isolated implementation worktree from the frozen baseline. Example command (do not run during Phase 0):

  git -C /home/kensei/repos/KenseiAgent worktree add -b delegate-task-phase-1-profile-and-units /home/kensei/repos/KenseiAgent-worktrees/delegate-task-phase-1 98a4aa453c1c576798a4461d5b2902d805eef71d

Before implementation in that worktree:

1. Confirm the worktree is absolute, isolated and clean.
2. Confirm `HEAD` is the frozen rollback SHA.
3. Confirm imports resolve from that worktree's `.venv` and unset inherited `PYTHONPATH` for pytest.
4. Copy the Phase 0 source map, ADR and test contract into the handover record; do not modify the canonical live checkout for implementation.
5. Witness RED tests before any production source edit.
6. Use one reviewed phase commit (or explicitly enumerated task commits plus a phase SHA).
7. Run the independent audit against the implementation worktree, not against a moving branch.

Phase 1 RED scope

- Profile argument/schema and target-profile child construction.
- Target profile model, toolset, fallback, provider routing and credential scope.
- Parent capability ceiling and no writable child long-term memory.
- Ungrouped detached tasks returning independently.
- Same-group tasks returning together.
- Grouping as delivery coordination, not execution ordering.
- Fast task not waiting behind an unrelated slow sibling.
- Whole-call task indexes and task-first notices.
- One delegate call consuming one async capacity slot.
- Distinct `active_count` versus `active_task_count` semantics.
- Child-level partial persistence across a crash.
- Failed-child notices separated from the final unit result.
- Synthesis/verification result preservation through unit completion.
- Per-model `provider_routing.models.<model-id>` partial overlays.
- Spelling variants, `openrouter/` prefixes and dot/dash variants.
- `/model` and fallback model changes re-resolving routing.
- Delegated worker target-profile routing not inheriting parent routing.
- OpenRouter-only payloads and direct-provider/Portal exclusion.
- Endpoint-pin precedence and separation from `fallback_providers`.

Phase 1 must not:

- cherry-pick upstream commits blindly;
- alter active gateways or restart services;
- push or deploy;
- rewrite aggregation without a witnessed failing seam;
- weaken profile, secret, terminal, recursion, authority, memory or redaction invariants;
- introduce OS sandboxing or writable child memory.

Required Phase 1 evidence directory:

  docs/evidence/delegate-task/phase-1/

Required Phase 1 audit inputs:

- base and phase SHAs;
- changed file/symbol manifest;
- RED and GREEN raw logs;
- Simplify Swarm output and post-simplification test log;
- remediation diff and rerun log;
- security/authority/profile-boundary review;
- rollback SHA;
- explicit GO / REVISE / STOP verdict.

Current baseline facts to carry forward:

- 69 tests collect successfully in the Phase 0 focused set.
- Baseline execution: 61 passed, 1 skipped, 7 expected profile-contract failures.
- Current upstream reference: e9bccc90a5e314ddf6dd4bd3772d50cb40027275.
- Headline upstream anchors are absent from KenseiAgent HEAD.
- No production source was changed in Phase 0.
