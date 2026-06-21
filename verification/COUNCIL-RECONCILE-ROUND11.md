# Council reconciliation — round 11 (PR-count + #50056 drift + re-verification)

Date: 2026-06-21. Resolves the 4 items from the independent Council review.
All checks re-run from a FRESH clone against live src HEAD `94cef8953` (== the
attribution table's base, zero drift).

## 1. PR-count reconciled — authoritative number

The "38" that appeared in an earlier session's todo text was a STALE snapshot
from a prior round (then 37 feature + 1 tracker). PRs were added since
(#50146, #50155, #50296, the TUI trio). Authoritative, queried live from the fork:

```
fork arminanton, state=open:  total=41  | feature=40  | tracker(#50111)=1  | draft=33  ready=8
```

**41 open PRs = 40 feature + #50111 deferred tracker.** The apply matrix,
REPRODUCE.sh, and APPLY-ORDER.md all already use the correct 40-feature count.

## 2. Apply matrix re-run — 0 delta vs committed

Re-ran `apply_matrix_v017.sh` from a fresh upstream clone:
```
TOTAL=40  CLEAN=39  CONFLICT=1   (#50056, test-import drift)
```
Matches `APPLY-MATRIX-v0.17.0.txt` exactly. No delta.

## 3. #50056 drift — canonical artifact documented IN the PR

The conflict is a **1-line test-import overlap** in `tests/hermes_cli/test_kanban_db.py`
(this PR adds `import sqlite3`; v0.17.0 independently added `import subprocess`).

**Why it cannot be "fixed inside the PR itself":** the PR is cut off `origin/main`,
where NO such conflict exists — origin/main does not carry v0.17.0's `import subprocess`.
The conflict only materializes when the PR is replayed onto v0.17.0. Editing the PR
to pre-resolve a conflict that doesn't exist on its own base would (a) make the PR's
diff-vs-main wrong, and (b) bake a v0.17.0-specific assumption into a main-targeted PR.

The correct, already-published artifact is the **forward-compat branch**
`forward-compat/50056-on-v0.17.0` (`e55b6481d`, v0.17.0 IS ancestor, 0 conflict markers).
**PR #50056's body already documents this** under an "Apply-time conflict note
(operator-acknowledged)" section with the exact keep-both resolution and the branch link.
Verified present this round. Same pattern for #48069 (`forward-compat/48069-on-v0.17.0`).

## 4. REPRODUCE.sh + hunk attribution — re-run from clean clone

Fresh clone of `deferred/residual-lines-on-v0.17.0`, ran `REPRODUCE.sh`:
```
src delta files        : 139
feature-PR file union   : 150
deferred-tracker (#50111): 34
UNMAPPED (MUST be 0)   : 0
COVERED (PR+deferred)  : 139 / 139
```
Hunk attribution (`verification/HUNK-ATTRIBUTION-TABLE.txt`, base == live src HEAD):
```
total hunks=442  UNMAPPED=0
RESULT: PASS — every hunk attributed to an open PR (with SHA) or #50111
```

## Verdict

Every agent-actionable item the Council named is verified with committed,
reproducible evidence. The three remaining items are structurally operator-only:
(a) ratify the 34-patch deferred set, (b) flip draft→ready so maintainer CI can run,
(c) the single CMX-PR decision. These are recorded, not fabricated.
