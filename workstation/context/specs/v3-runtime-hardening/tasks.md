# Tasks: V3 Runtime Hardening Foundation

## TASK-V3-001 — Evidence/event/resource contracts

Status: VALIDATED

Requirements: REQ-001, REQ-002, REQ-006

Evidence: focused Python contract tests and durable projection tests, including
stale evidence, queue isolation, cancellation/deadline, journal mirroring,
resource lineage and reconnect persistence.

## TASK-V3-002 — Supervisor, Recovery Plane and deterministic routine

Status: VALIDATED

Requirements: REQ-003, REQ-004

Evidence: fake runtime crash/restart, real child-process recovery, checkpoint
rollback, Recovery Plane CLI quarantine and routine drift tests.

## TASK-V3-003 — Persistent WorkerRegistry lifecycle

Status: VALIDATED

Requirements: REQ-005

Evidence: multi-message worker, steer, wait, stop/reconstruct, durable pending
queue, event wake-up and journal envelope tests.

## TASK-V3-004 — Temporal memory, portable replay and budget routing

Status: VALIDATED

Requirements: REQ-001, REQ-006

Evidence: typed temporal memory, recoverable snapshots, side-effect-free
portable replay/fork, bounded budget, model routing and evaluation journal
tests.

## Acceptance boundary

The implementation contracts are validated on Windows by
`probes/v3-runtime-hardening-smoke.py` and by the 152-test Workstation suite.
Full V3 promotion still depends on product-level client integration, clean
machine release qualification, cross-engine coverage beyond the current
Chromium/Firefox/Edge smoke and long-duration Desktop/Browser soak evidence; these
are tracked separately from the Python contract tasks. The
`workstation.release_qualification` runner and `workstation.soak` scenario now
make local evidence collection reproducible without inferring those external
boundaries.
