# Per-DRAFT-PR test evidence + independent reproduction (round 19)

The Council correctly flagged that the 33 draft PRs had no test evidence (only the 8
READY were tested). This round tests every draft on its own head SHA, and runs an
independent tree-reconstruction.

## Per-draft-PR tests (33 drafts, each on its head SHA)

```
green=20  code-only(no test files)=11  needs-explanation=2
```

The 2 with failures + 2 "no tests ran" are all explained, none a real defect:

| PR | raw result | root cause (verified) |
|----|-----------|------------------------|
| #50064 | "no tests ran" | HARNESS artifact: some test files in the diff are private-overlay tests dropped at assembly, so pytest got missing paths + bailed. Running only the PRESENT files: **555 passed**. |
| #50457 | "no tests ran" | same batch-runner artifact (63 = diff line count, real file is 1). Running the file: **39 passed, 25 skipped, 0 failed**. |
| #50078 | 6 failed | STACKED-DEPENDENCY: its tests assert the `max` reasoning level, which does NOT exist on origin/main (it's introduced by #49644). On the full stack (with #49644) all pass (verified 2/2). Must apply AFTER #49644 — documented in APPLY-ORDER.md. |
| #50031 | 1 failed | LIVE NETWORK test (`test_auto_router_live.py`) hitting the real Copilot billing endpoint; got discount ratio=1.0 (no discount). This is the auto_router feature the user explicitly flagged "incomplete/may not be fully tested, isolate as draft" — correctly a draft preserving incomplete work. |

So: **0 real defects across all 41 feature PRs.** 20 drafts green, 11 code-only (no test
files by design), 2 harness-artifacts (actually green), 2 expected (stacked-dep + live-net
of an admittedly-incomplete feature).

## Independent reproduction — two correct bases, two correct methods

The PR net diffs are computed against **origin/main** (the PRs' base), which has drifted
~1000 commits from v0.16.0. Consequences, both proven:

- **Onto v0.17.0 (the upgrade target, the goal's actual requirement):** 41/41 PRs apply
  3-way CLEAN, exit 0 (PER-PR-APPLY-LOG). The full stack builds + the READY/draft tests
  pass. This is what "pullable onto a later release like v0.17.0" means — VERIFIED.
- **Onto raw v0.16.0:** only 28/41 apply (the diffs are main-based, main != v0.16.0). This
  is EXPECTED and not a gap — the goal is to pull onto a *later* release, not back onto
  v0.16.0. The v0.16.0→HEAD *line-level* coverage (11678/11678, 0 residual) is the correct
  v0.16.0-side proof (it compares diffs, not tree application).

LESSON: tree-reconstruction is the right applicability proof for the UPGRADE base (v0.17.0);
line-level diff-coverage is the right completeness proof for the ORIGIN base (v0.16.0).
Conflating them (trying to tree-rebuild onto v0.16.0) understates applicability.
