# #50626 trim + final coverage re-balance (2026-06-22)

## Action (agent-actionable mechanical hygiene, per Council)
PR #50626 previously bundled TWO logical changes: (a) the `subdirectory_hints.py` RuntimeError
guard, and (b) a unique xAI display-label override. (a) is a **duplicate of maintainer-preferred
open #29433** (verified: same `except (OSError, ValueError, RuntimeError)` guard; #29433 also
ships its own test `test_subdirectory_hints_tilde.py`; the maintainer marked our earlier #50049 a
duplicate of #29433). Per one-PR-per-logical-change, #50626 was **trimmed** (force-pushed) to just
its unique half:

- #50626 net diff is now **exactly 1 file / 1 line**: `hermes_cli/providers.py` (+`"xai": "xAI"`).
- Verified net-new (not already on main), compiles, applies CLEAN on v0.17.0.
- The 2 subdir-hint files defer to upstream #29433 (a new SUPERSEDED class — deferring loses nothing).

## Coverage after the trim (re-run `verification/reproduce-coverage.sh`)
```
total src-delta files : 165
covered by open PR    : 129
DISCARD (.bak/intel/txt): 25
WITHDRAWN (maintainer): 9
SUPERSEDED (upstream #29433): 2
ORPHANS               : 0
check: 129 + 25 + 9 + 2 + 0 = 165 (== 165 ? YES)
```

## Set still stacks
The PR set is still 39 open code PRs (#50626 unchanged in count, just smaller). Trimmed #50626
applies CLEAN on v0.17.0; the set-level `stack-apply-v017.sh` result (39/39 CLEAN) is unaffected
(the trim only shrinks one PR's diff).
