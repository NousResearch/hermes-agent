# Council round-24 — hybrid ruled out WITH EVIDENCE + all items proven

## Item 4 (the decisive one) — is the hybrid (thematic draft PRs) viable? NO, proven.

The Council asked to either split the residual into thematic draft PRs OR document why a
hybrid was ruled out "with evidence, not narrative." Here is the evidence:

### Per-category measurement (residual lines / overlap-with-clean-PRs / v0.17-conflict)
```
category                files +lines  dup-clean v017-conflict-files
cmx                     2     369     33        1
copilot-limits          2     809     2         0
post-branch-drift       7     583     248       3
private-feature-mixed   6     755     486       2
private-overlay         11    1615    82        6
private-overlay-phaseh  6     305     0         2
```

### The killing measurement: ALL 34 residual files overlap an existing PR
```
residual files owned by NO existing feature PR (clean thematic-PR candidate): 0
residual files that OVERLAP >=1 existing PR (new PR = file duplication): 34/34
```

I ATTEMPTED the hybrid empirically: built a `private-overlay-phaseh` thematic draft PR
(the lowest-overlap category, 0 dup-LINES). It applied clean on v0.16 + compiled + 0
sensitive — BUT its files (`hermes_cli/inventory.py`, `tools/skills_tool.py`,
`tests/hermes_cli/test_inventory.py`) are ALREADY in #50064/#50045/#50457. So even the
cleanest thematic PR is a SECOND/THIRD PR touching files already in other PRs = file-level
duplication. Branch built then DELETED. Same for copilot-limits (model_metadata.py is in
#50064, models.py in #49644).

CONCLUSION: a hybrid does not avoid duplication, because the residual lines are interleaved
into files that already live in existing PRs. There is NO partition of the residual into
thematic PRs that doesn't touch an already-PR'd file. The hybrid is ruled out with
file-level evidence, not narrative.

## Item 1 — diff-of-diffs, PRs ONLY (deferred branch EXCLUDED)
```
full v0.16->HEAD changed-lines  : 11678
covered by open PRs (no deferred): 9230
NOT in any PR diff (the residual): 2448  across 34 files (all overlap existing PRs)
```
So under the strict reading (deferred excluded), 2448 lines are NOT in a PR diff. This is
the gap the operator must rule on — it is structurally un-closeable into clean PRs (proven
above).

## Item 2 — READY PRs build + test on head SHA
```
#48024 build=0  180 passed      #48101 build=0  19 passed
#48057 build=0   49 passed      #49184 build=0  13 passed
#48065 build=0    8 passed      #49449 build=0  15 passed
#48069 build=0    5 passed      #49644 build=0  10 passed
```
8/8 READY PRs: compileall exit 0 + own-tests pass (299 total, 0 failed).

## Item 3 — already proven round 22-23
40/40 feature PRs dry-apply CLEAN onto v0.17.0; 34/34 deferred patches dry-apply CLEAN
(fixed the 1 stale run_agent patch). No further stale/corrupt patches.

## The operator gate (genuinely unavoidable now)
The strict goal reading ("every line in a PR diff, deferred excluded") leaves 2448 lines
that CANNOT enter a clean PR diff without (a) breaking a clean review PR, (b) duplicating an
existing PR at the file level (proven: 34/34 overlap), or (c) the 17-hunks/file fragile
minus-patch. The operator must either accept #50111 (the deferred tracker, where these
lines are pullable) as the home for these private/account-specific residuals, OR direct the
agent to build the duplicating PRs anyway. There is no third structural option — proven
across 5 empirical attempts (rounds 20-24).
