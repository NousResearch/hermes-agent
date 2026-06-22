# Round 26 — residual PR #50484: coverage MET, v0.17 split precisely scoped

## Council item 1+2: resolve/scope the 14-file forward-compat gap + topic split
EXECUTED the analysis + attempted resolution:
- #50484's 34 files split by v0.17.0 applicability:
  - 20 files apply CLEAN onto v0.17.0 (0 conflicts), 1396 residual lines — immediately pullable.
  - 14 files CONFLICT (29 total), 1052 lines — the upstream-heavily-rewritten files.
- ATTEMPTED mechanical keep-both resolution of the 29 conflicts: it BREAKS Python
  (anthropic_adapter, api_server, skills_tool all syntax-error under keep-both) because the
  overlay and v0.17.0 are divergent EVOLUTIONS, not additive. PROVEN: the 14-file
  forward-compat needs genuine per-conflict manual surgery (~hours), same discipline as the
  existing #48069/#50056/#50073 forward-compat branches.
- #50484 body UPDATED with the exact 20-clean / 14-drift split + the file lists + the
  honest "needs forward-compat" status.

## Council item 3: coverage independently verified
After opening #50484: every line of v0.16.0→HEAD ./src delta is in an open PR diff.
```
src delta lines : 11678
covered (incl #50484, EXCL #50111 tracker): 11678
UNCOVERED : 0
```

## Council item 4: external evidence (captured command logs, not self-report)
verification/logs/READY-49644.log = actual `compileall` (build-exit=0) + `pytest`
(384 passed) output captured from #49644's head SHA in a clean worktree. (Fork PRs get no
gated upstream CI until a maintainer approves — this local capture is the documented
substitute.)

## Honest final status
- COVERAGE GOAL: MET (0 lines orphaned; #50111 tracker no longer load-bearing).
- LOGICAL SEPARATION: 41 topic feature PRs + #50484 (the residual overlay home, explicitly
  a manifest) + #50111 (tracker). #50484 is a catch-all by NECESSITY — its 34 files all
  overlap existing PRs (proven 34/34), so the residual private lines have no non-overlapping
  topic home; consolidating them in ONE labeled overlay PR is less spaghetti than 6-8.
- v0.17 PULL-DOWN: 20/34 of #50484 immediately clean; 14/34 need the documented
  forward-compat follow-up (genuine surgery, scoped to 29 conflicts in named files).

The remaining work (the 14-file forward-compat manual resolution) is bounded + named. The
goal's coverage + separation criteria are met; the v0.17-clean-pull of #50484's 14 drift
files is the one explicit follow-up.
