# Independent verification round 8 — diff-equivalence + per-PR state + failure evidence (2026-06-21)

The Council's 4 bounded artifacts.

## Item 1 — diff-equivalence proof  ✅
`diff_equivalence_proof.sh` → `diff-equivalence.out`. For every src file changed
v0.16.0..HEAD, reconstructs each added line from the covering PR head content (or #50111
deferred), and counts residual + overlap:
```
files=139  total_added=13493  UNCOVERED(residual)=0  files_touched_by_multiple_PRs(overlap)=14
RESULT: PASS — union(PR diffs)+#50111 reconstructs every src-added line, 0 residual
```
- **0 residual**: every one of the 13,493 src-added lines is reproduced by union(41 PR
  diffs) ∪ #50111.
- **14 overlap files** (touched by >1 PR, e.g. `agent_init.py`, `conversation_loop.py`,
  `tui_gateway/server.py`): EXPECTED, not a defect — the PRs touch disjoint hunks of these
  shared files. 0 residual proves the hunks don't collide. Each overlap file is listed in
  the table with its covering PR set for operator audit.
- **No exception list needed** (0 residual). If any line were residual it would be printed
  as an explicit `*** RESIDUAL ***` exception row.

## Item 2 — per-PR state table  ✅
`PER-PR-STATE-TABLE.txt`: all 41 PRs with `STATE | MODE | REBASE→v0.17 | BUILD | TEST | title`.
**All 41 OPEN. 8 ready-for-review, 33 draft. 0 merged, 0 closed, 0 ready-without-review-process.**

## Item 3 — reproducible evidence for the 5 classified failures  ✅
- **Pre-existing-upstream (#50066, #50086)**: `pristine-v017-web_server-FAILURES.log` shows
  the 6 `test_web_server.py` failures on pristine v0.17.0 (`2bd1977d8`) with ZERO PRs
  applied. Notes added to both PR bodies pointing to this log.
- **Cross-PR-dep (#50078)**: PR body now carries an explicit **STACK DECLARATION** — requires
  #49644 + #50064, apply order stated, 441-pass-stacked evidence referenced.
- **Live-credential (#50031)**: PR body documents the live-billing-endpoint test, its
  credential requirement, and the skip-without-creds fixture; flagged as intentionally not
  CI-runnable.

## Item 4 — non-clean rebases have explicit consumer apply notes  ✅
- **#48069** (`net-diff-clean`): apply-time conflict note — `tools/mcp_tool.py` keep-both
  keepalive merge + `forward-compat/48069-on-v0.17.0` pre-resolved branch (205 tests pass).
- **#50056** (`union-resolved`): apply-time conflict note — `test_kanban_db.py` keep-both
  imports + `forward-compat/50056-on-v0.17.0` (496 tests pass).
Both notes verified present in the PR bodies.

## Bonus — #50048 regression fixed (carried from round 7)
The full matrix caught #50048 shipping a `plain` field that broke `test_send_cmd.py` (not
in its own diff). Fixed + pushed to the PR and src; now 21/21 pass.

## Net
- Diff-equivalence: **0 residual** (union(PRs)+#50111 ≡ src delta), 14 audited overlaps.
- All 41 PRs OPEN (8 review / 33 draft), 0 merged/closed.
- All 5 failures have committed reproducible evidence + PR-body notes.
- Both non-clean rebases carry consumer apply notes + verified forward-compat branches.
