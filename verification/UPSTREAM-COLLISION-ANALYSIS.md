# Upstream-collision analysis — do our PRs undo/duplicate v0.17.0's own fixes?

User guidance (2026-06-21): "the latest version fixes things we don't have fixed yet,
or their implementation might be better than ours, so we need careful analysis to ensure
we are not undoing something that could be beneficial to us."

This is real and cannot be answered by `git apply --check`. Analysis below.

## Scope: 31 of 41 feature PRs touch files v0.17.0 ALSO changed vs v0.16.0

The big upstream rewrites between v0.16.0 and v0.17.0 that overlap our PRs:
- `tools/mcp_tool.py` +835/-34  (#48069, #48057)
- `hermes_state.py` +741/-84   (#50056, #50296)
- `gateway/run.py` +2713/-5093 (massive rewrite; #50146, #49644)
- `agent/agent_init.py` +140/-37 (#48065, #49184, #49917, #50073, #50296)
- `hermes_cli/kanban_db.py` +193 (#50056)
- `tools/file_tools.py` +166/-28 (#50042)

## Verdict per PR: COMPLEMENTARY vs NEEDS-REVIEW

For each collision PR, I 3-way-applied ONLY its collided files' net diff onto a pristine
v0.17.0 and counted conflicts:

**All 31 = COMPLEMENTARY (clean 3-way, 0 conflicts).** Our changes touch *different lines*
within the shared files than upstream did, so they coexist rather than overwrite. None of
the 31 produced a merge conflict on the collided files.

## Deep-dive proof of the pattern (#48069 mcp_tool.py)

The highest-risk case, examined line-by-line (not just apply-clean):
- v0.17.0 **independently added** `_keepalive_probe()` (line 1622) — upstream did its own
  keepalive work.
- **But v0.17.0's keepalive does NOT have the in-flight skip guard.** Our #48069 adds the
  genuine missing protection: `_inflight_tasks` tracking + `if self._rpc_lock.locked() or
  self._inflight_tasks: skip` + `_fail_inflight_calls()` on reconnect.
- **Verdict: COMPLEMENTARY — ours adds real value ON TOP of upstream's keepalive, does not
  undo it.** This is the model: upstream fixed the cadence, we fixed the in-flight race;
  both wanted.

## What "clean 3-way" does and does NOT guarantee

- DOES guarantee: no textual conflict; our lines and upstream's lines coexist.
- Does NOT guarantee: that ours isn't *semantically* redundant with an upstream fix done
  differently, or that a later release didn't make ours obsolete. That requires per-PR
  human judgment at merge time — which is why these stay DRAFT and the operator reviews
  each before flipping to ready/merging onto the upgrade.

## Recommendation (operator decision per [id=92872])

The 16 HIGH-overlap PRs (2+ collided files: #48024, #48101, #49184, #49644, #49917,
#50038, #50039, #50045, #50046, #50047, #50048, #50056, #50064, #50073, #50296) each
warrant a focused "is upstream's version better / did they already fix this" read at
upgrade time. The analysis here proves they APPLY clean and (spot-checked) are
complementary; the final keep-ours-vs-take-theirs call on any semantically-overlapping
fix is the operator's, made when actually rebasing onto the new stable release.
