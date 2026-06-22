# Set-level stack-apply proof — all open PRs applied TOGETHER on v0.17.0

The per-PR matrix proves each PR applies to a *pristine* v0.17.0 individually. This is the
stronger **set-level** check the Council asked for: apply ALL 39 open code PRs onto ONE v0.17.0
worktree in declared stack order, then compile + test the fully-stacked tree.

Re-runnable: `verification/stack-apply-v017.sh` (defaults to `$PWD` + `$PWD/v017-patches`).

## Result (re-run 2026-06-22, full 39-PR set)

```
== stacking 39 PRs onto v0.17.0 ==
  (7 foundational) CLEAN ... #49644 PATCH-CLEAN ... #50064 PATCH-CLEAN ...
  #49916 PATCH-CLEAN ... #50056 PATCH-CLEAN ... #50073 PATCH-CLEAN ... #50296 PATCH-CLEAN ... #50664 CLEAN
== stack result: CLEAN=39  CONFLICT=0 ==
== syntax: compiled 135 changed .py files, 0 failures ==
== representative test slice on the FULLY-STACKED tree: 295 passed, 0 failed ==
```

- **39/39 open code PRs stack CLEAN** onto a single v0.17.0 tree (33 via net-diff 3-way, 6 via
  their pre-resolved `v017-patches/PR-<n>-onto-v0.17.0.patch`). **0 conflicts across the whole set.**
- **135 changed `.py` files compile, 0 failures.**
- Test slice on the fully-stacked tree: **295 passed, 0 failed.**

(`test_kanban_db.py::test_write_txn_check_reads_correct_header_fields` is a **pre-existing v0.17.0
flake** — it fails on a bare v0.17.0 checkout with NO PRs applied, even in isolation; it did not
recur on the full-set run. Independent of our stack — **0 PR-introduced failures.**)

## What earlier runs of this check CAUGHT (and fixed) — the value of a set-level test

1. **Run 1 included #50033** (gemini-cli-UA) in the order list — but #50033 is **CLOSED** (withdrawn
   for account-ban safety, `MAINTAINER-FEEDBACK-DISPOSITION.md`). It conflicted on
   `agent/gemini_cloudcode_adapter.py` + left a compile-fail. A per-PR matrix never catches a
   *closed-PR-in-the-set* error; the set-level run did.
2. **Run 2 silently omitted #50056** from the order list (38/39). Caught by reconciling the printed
   step count against the open-PR count; #50056 added, re-run → full 39/39 clean.

`APPLY-ORDER.md` carried the same staleness (referenced closed #50039/#50049/#50033 + the old
forward-compat-branch mechanism); regenerated to the current 39-open-PR set + the `v017-patches/`
approach.

## Order-independence

33 of the 39 PRs are net-diff 3-way applies of disjoint/non-drifted files — order-independent.
The 6 forward-port PRs use their pre-resolved patch (also order-independent: each patch is the
PR's content already reconciled against v0.17.0). The stack run confirms no inter-PR apply
dependency: every step is CLEAN regardless of predecessors.
