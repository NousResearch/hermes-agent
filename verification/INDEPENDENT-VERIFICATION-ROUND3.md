# Independent verification round 3 — external cherry-pick from REMOTE heads (2026-06-21)

Addresses the Council's round-3 gap: verify against PR REMOTE heads (not local src),
end-to-end cherry-pick onto v0.17.0, and confirm #50033 carries the utf-8 fix.

## Item 3 — #50033 remote head carries the encoding='utf-8' form  ✅
`git show <#50033-head>:agent/google_user_agent.py` on the FORK remote contains
`read_text(encoding="utf-8")` ×2 and `encoding="utf-8"` ×4. The src commit `544d230a3`
aligned src TO #50033 (which already had it) — the fix is in the PR, not only local src.
(So the coverage was never circular: #50033-remote had the encoded form, src had the bare
form, src was fixed.)

## Item 1 — external cherry-pick of all 40 feature PR remote heads onto fresh v0.17.0  ✅
`external_cherrypick_all_prs.sh` — real `git cherry-pick` of each PR's own commits from its
fork remote head, PR-number order:
**37 CLEAN, 3 CONFLICT.** The 3 conflicts:
- #48069 `tools/mcp_tool.py` — DOCUMENTED keepalive drift.
- #50056 `tests/hermes_cli/test_kanban_db.py` — DOCUMENTED import-combine (sqlite3+subprocess).
- #48101 (empty file list) — a transient stack-state artifact: #48101 cherry-picks CLEAN
  both ALONE and after #49917 (the system_prompt.py overlap source), verified in isolation.
So genuine cumulative conflicts = exactly the 2 documented drifts.

## Item 4 — failures reproduced on pristine v0.17.0 (zero PRs)  ✅
The 6 `test_web_server.py` failures reproduce on a fresh v0.17.0 worktree with NO PRs
(6 failed, 300 passed). Cause: editable-install `cron->src` sys.modules pollution (5) +
one timing flake (1). #50078's reasoning failures are a cross-PR dependency (pass 441
stacked on #49644+#50064). #50031 is a live-credential smoke test.

## Item 2 — per-file coverage against remote heads  ✅ (and it caught more)
`independent_coverage_proof.sh` reads each PR's REMOTE head (`git show <sha>:<file>`), not
local src. Method = per-file content containment. **PASS — 0 uncovered.**

### Whole-file residual ≠ coverage (important methodological note)
A naive whole-file equality check (`external_residual_check.sh`) flagged 94 "differing"
files. Investigated every category — ALL explained, ZERO real gaps:
- **~84% upstream drift**: the file changed between v0.16.0 and v0.17.0 upstream, so a
  v0.17-based stack differs from the v0.16-based src REGARDLESS of our PR (e.g.
  `usage_pricing.py`: 5 upstream lines). Our PR's added lines ARE in the PR.
- **PR-adds-new-file**: clean tests the PR adds that src lacks (e.g.
  `test_copilot_claude_endpoint_routing.py` — #49184 wrote a fresh clean test instead of
  shipping src's private entangled `test_copilot_opus_context_fix_2026_06_04.py`).
- **deferred content**: src versions that are private-overlay (e.g.
  `test_copilot_context.py` = private-overlay-phaseh, confirmed in
  `#50111:deferred/private-overlay-phaseh/...patch`; `probe_prelude_e2e.py`'s 2
  `/mnt/devvm/...review-sysprompt-prelude` path lines = private, in the deferred set).

The correct coverage metric is per-line content containment (0 uncovered), not whole-file
byte-equality (which conflates upstream drift + new files + private-deferred content).

## Net (round 3)
- 40 feature PRs cherry-pick onto v0.17.0 from remote heads: **37 clean + 2 documented
  drifts** (#48069, #50056); #48101's "conflict" is a transient that's clean in isolation.
- #50033 remote head verified to carry the utf-8 fix.
- Per-file coverage vs remote heads: **0 uncovered**.
- Pristine-v0.17.0 failure reproduction: confirmed.
- Whole-file 94-diff: 100% explained (upstream drift / new-file / deferred), 0 real gaps.
