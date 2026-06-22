# Council round-25 — integrated stack + review walk (recorded evidence)

## Item: integrated stack-apply of all PRs onto v0.17.0 + build + test (single worktree)
Built a single worktree off v0.17.0 (2bd1977d8), applied all feature PRs:
```
APPLIED=36 direct  +  APPLIED*(fc)=3 forward-compat  +  #50457 integration test
stack head: 3388c6619   v0.17.0 IS ancestor: YES
```
- BUILD: `compileall agent/ hermes_cli/ tools/ run_agent.py cli.py gateway/ tui_gateway/` → exit 0
- TEST (integrated campaign PR test files): 210 passed, 0 failed

## Item: review-comment walk (all 41 open PRs)
10 PRs have comment activity; almost all are arminanton's OWN campaign-evidence posts.
The REAL reviewer (alt-glitch, triage bot/maintainer) left cross-reference comments on 5:

| PR | reviewer note | disposition |
|----|---------------|-------------|
| #50086 | Related to #39894/#40805/#42467/#48049; confirms OUR inode-dedup fix is live + non-redundant on main | no action — affirms non-redundant |
| #49449 | Related to #42632/#29146/#29147 (existing Copilot-context PRs) | no action — complementary, already cross-ref'd |
| #50155 | Related to #50053 (additive grounding hooks) | no action — complementary |
| #50296 | Related to #27190 (merged — isolates review fork) | no action — complementary |
| **#50049** | **Duplicate of #29433 (earlier still-open, broader fix)** | **ADDRESSED**: acknowledged on-PR (comment 4763793706), recommended maintainers prefer #29433, deferred the close to the repo owner (not unilateral). |

So: 1 genuine duplicate finding (#50049) ADDRESSED with an on-PR note + operator-deferred
close; the rest are triage cross-references affirming our PRs are live and non-redundant.
No outstanding review comment is left unaddressed.

## Items proven prior rounds (carried, re-confirmed)
- 40/40 feature PRs + 34/34 deferred patches dry-apply CLEAN onto v0.17.0 (round 22).
- 8/8 READY PRs build exit 0 + 299 own-tests pass on head SHA (round 24).
- Hybrid thematic-PR partition RULED OUT with file-level evidence: 34/34 residual files
  overlap an existing PR (round 24).
- diff-of-diffs (deferred excluded): 9230/11678 in PR diffs, 2448 residual (round 24).

## The operator gate (unchanged, the sole remaining item)
2448 lines (private/account-specific, interleaved into already-PR'd files) cannot enter a
clean PR diff without breaking a clean PR or duplicating one (proven across 5 attempts).
Operator must ratify (B) #50111-as-residual-home, or direct (A) the duplicating PRs.
