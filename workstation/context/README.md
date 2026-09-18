# Hermes Workstation — Coding-Agent Context

This directory is the operational entry point for coding agents working on the downstream Hermes Workstation surface. It does **not** replace the repository-wide rules in [`../../AGENTS.md`](../../AGENTS.md) or duplicate the architecture documents in `workstation/`.

## Required reading order

Before changing Workstation code, read these documents in order:

1. [`../../AGENTS.md`](../../AGENTS.md) — repository-wide engineering rules and invariants.
2. [`CURRENT_STATE.md`](CURRENT_STATE.md) — what works, what is partial, what is not built, and the latest validation state.
3. [`ADAPTIVE_EXECUTION_COMPILATION.md`](ADAPTIVE_EXECUTION_COMPILATION.md) — **active execution-policy correction** discovered by 2026-09-18 native-browser dogfood; preserves causal safety while making novel stateful work executable.
4. [`PROGRESSIVE_OPERATIONAL_COMPILATION.md`](PROGRESSIVE_OPERATIONAL_COMPILATION.md) — **active target architecture** for Operational Capability, Operational Kernel, automatic deterministic reuse and the new role of `work_execute`; defines what should be compiled after AEPC defines when compilation is safe.
5. [`UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md`](UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md) — active reliability hardening lane for ownership, transcript resync, Kanban provenance/liveness, worker-exit truth and bounded browser recovery. Read H-065/H-066 before repeating its audit.
6. [`CANONICAL_EXECUTION_RELIABILITY_GATE.md`](CANONICAL_EXECUTION_RELIABILITY_GATE.md) — implemented causal reliability invariants that all active lanes must preserve.
7. [`FORENSIC_RELIABILITY_SYNTHESIS_2026-09-17.md`](FORENSIC_RELIABILITY_SYNTHESIS_2026-09-17.md) — integrated evidence map across the three 2026-09-17 investigations.
8. [`MAINLINE_CONSOLIDATION.md`](MAINLINE_CONSOLIDATION.md) — the pre-1.5 gate result, complete PR/branch disposition ledger, and recurring post-milestone review rule.
9. [`DECISIONS.md`](DECISIONS.md) — settled downstream architecture decisions.
10. [`CONSTRAINTS.md`](CONSTRAINTS.md) — non-negotiable boundaries and security/maintenance constraints.
11. [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — Workstation runtime architecture and distribution model.
12. [`../UPSTREAM.md`](../UPSTREAM.md) and [`../UPSTREAM_DELTA.md`](../UPSTREAM_DELTA.md) — upstream base, synchronization model, tracked downstream delta, and planned selective adaptations.
13. [`../SOURCE_MATRIX.md`](../SOURCE_MATRIX.md) — ownership and use of internal/external components.
14. [`../ROADMAP.md`](../ROADMAP.md) — sequencing and intentionally deferred work. Immediate architecture work is AEPC-E002 plus Progressive Operational Compilation; feature expansion remains subordinate to reliability.
15. [`TESTING.md`](TESTING.md) — validation ladder and evidence required before a Workstation change is considered stable.
16. [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — reproduced or observed problems whose causes must not be guessed.
17. [`engineering-journal/CURRENT.md`](engineering-journal/CURRENT.md) — active hypothesis/experiment ledger and anti-repeat memory. H-067 is the current Capability abstraction finding.
18. [`../PATCH_MANIFEST.md`](../PATCH_MANIFEST.md) when touching an upstream integration point, rebase/migration tooling, or the downstream patch surface.

After this read-order, inspect the **current `main` implementation and its tests** for the subsystem you intend to change. Documentation is intent and state; code on `main` is the source of truth for implementation details. Historical findings that current `main` has already fixed become regression evidence, while unverified claims must be reproduced before architecture changes. In particular, do not recreate a native Browser harness merely because #114964 describes an upstream keepalive bug: H004/H013 already exist downstream and must be reused/extended.

## Working rule

A coding agent must be able to answer these questions before editing Workstation code:

- Which existing Hermes subsystem owns this state or capability?
- Is the proposed change extending an existing path or creating a duplicate source of truth?
- Is the capability process-scoped, session-scoped, BrowserTask-scoped, TaskRun-scoped, operation-scoped, or profile-scoped?
- What canonical `task_id` / `run_id` / `operation_id` lineage applies to a mutation or completion claim?
- What is the upstream delta created by the change?
- Which behavior contract will prove the change works end to end?
- What security boundary changes, if any?

If any answer is unclear, inspect the implementation and tests before writing code. Do not infer missing behavior from filenames, old plans or previous conversation state.

## Continuous engineering journal

For an active Workstation investigation, the agent must keep `engineering-journal/CURRENT.md` synchronized with evidence as work proceeds.

Before executing a new experiment, register the hypothesis/experiment and its confirming/refuting evidence. After execution, record the exact observed result and classification before moving to the next hypothesis. Do not repeat a failed approach unless a material input changed and that change is recorded.

This journal is deliberately operational and may change frequently. Settled product truth must still be promoted into the canonical context documents rather than living only in the journal.

## Maintenance

Update these context documents when the corresponding fact changes. Keep them concise and refer to the canonical detailed document instead of copying it. `CURRENT_STATE.md`, `KNOWN_ISSUES.md`, and `engineering-journal/CURRENT.md` are expected to evolve most frequently; architectural decisions should change only when a deliberate replacement decision is recorded. After each major milestone, perform and record the proportional Mainline Consolidation Review defined in `MAINLINE_CONSOLIDATION.md` before branching the next milestone.
