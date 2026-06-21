# Final state after resolving #50457 and #50111 (round 18)

## #50457 — RESOLVED (my unilateral close was an error; corrected)
Reopened + rescoped. Added a module-level `_needs_overlay` skipif guard so the 14
overlay-dependent tests (mythos/fable routing + agy-cli) SKIP on a clean stack and RUN
on the full tree. Verified GREEN on BOTH:
  - clean public stack: 39 passed, 25 skipped, 0 failed
  - full overlay tree:  64 passed, 4 xfailed, 1 xpassed, 0 failed
State: OPEN, draft. The opus-context test now lives in this PR, NOT deferred-only.

## #50111 — now a clean tracker, not a catch-all
34 deferred patches remain (was 35; opus-context graduated to #50457). Each is a
private/account-specific/drift-supersession fragment proven (round 12-14) to be
structurally un-graduatable into a clean standalone PR. Every one is a pullable
.patch with a per-category README. Line-coverage proves nothing is lost.

## Artifacts (post-resolution)
1. hunk/line coverage: 11678/11678, 0 residual
2. file mapping: 139 files -> 139 feature-PR, 0 deferred-only, 0 unmapped
3. per-PR 3-way apply on v0.17.0: 41/41 exit-0 CLEAN (38 direct + 3 fwd-compat, 0 conflict)
4. 8 READY PRs' own tests on head SHA: 299 passed, 0 failed (unchanged; READY PRs untouched)
5. PR-state: 42 open (41 feature + #50111 tracker), 8 ready + 34 draft, 0 closed-in-error

## Upstream-collision (user's guidance): 31 PRs touch v0.17.0-changed files, all
COMPLEMENTARY (clean 3-way); #48069 deep-dive proves ours adds the in-flight guard
v0.17.0's keepalive lacks. Final keep-ours-vs-take-theirs is per-PR human judgment at
rebase time (PRs stay draft for that review).
