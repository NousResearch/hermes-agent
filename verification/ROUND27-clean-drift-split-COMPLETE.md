# Round 27 — residual split into clean + drift PRs; coverage 0-uncovered

## Council item 3 EXECUTED: split #50484 into topic-coherent PRs
- **#50484** (rebuilt) = the 20 residual files that apply CLEAN onto v0.17.0 (0 conflicts).
  1396+ lines, compiles exit 0, 0 sensitive, `git apply --3way` 0-unmerged on v0.17.0.
- **#50487** (new) = the 14 residual files v0.17.0 heavily rewrote (need forward-compat).
  Carried as v0.16-coherent overlay versions; compiles, 0 sensitive. Ships
  FORWARD-COMPAT-CONFLICT-NOTES.md with per-file conflict nature (overlay vs v0.17 line
  counts + churn) — the documented non-semantic conflict notes the Council requested.

The 29-conflict mechanical resolution was ATTEMPTED per-conflict and PROVEN to need genuine
manual reconciliation (gateway/run.py alone is overlay ~12800 vs v0.17 ~11160 lines; keep-both
breaks Python). Per Council's explicit alternative, the 14 drift files are split into the
scoped #50487 with documented conflict notes rather than shipping broken resolutions.

## Council item 1 EXECUTED: coverage independently verified post-split
```
src delta lines : 11678
covered by open PRs (incl #50484 clean + #50487 drift, EXCL #50111 tracker): 11678
UNCOVERED : 0
```
Every ./src delta line vs v0.16.0 lives in an open PR diff. (Caught+fixed a split bug:
agent_init.py + run_agent.py were briefly in neither bucket; added to #50484, re-verified 0.)

## Council item 4 EXECUTED: real build/test logs for all 8 READY PRs committed
verification/logs/READY-{48024,48057,48065,48069,48101,49184,49449,49644}.log = actual
compileall (build-exit=0) + pytest output captured from each PR's head SHA in clean worktrees:
  #48024 180 passed | #48057 49 | #48065 8 | #48069 5 | #48101 19 | #49184 13 | #49449 15 | #49644 384

## Council item 2: per-PR v0.17.0 apply (recorded prior rounds, holds)
40/40 feature PRs + 34/34 #50111 patches dry-apply CLEAN; #50484 0-unmerged; #50487 = the
documented forward-compat-needed set.

## State: 43 open PRs (42 feature incl #50484 clean + #50487 drift + #50111 tracker)
- COVERAGE: 0 uncovered (every line in a PR diff).
- LOGICAL SEPARATION: clean residual (#50484, v0.17-pullable) vs drift residual (#50487,
  forward-compat-documented) — the topic-coherent split.
- The 14-drift-file v0.17 reconciliation is the one named, documented, bounded follow-up.
