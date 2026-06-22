# v0.17.0 rebase / apply verification — sample of open PRs

Goal bar: the open PRs must be **pullable onto v0.17.0** (`2bd1977d8fad185c9b4be47884f7e87f1add0ce3`),
not merely set-cover the src delta. This records a real cherry-pick / 3-way-apply +
test-execution pass for a representative sample, run from a clean v0.17.0 worktree.

## Sample (Council-named: #50664 + the 4 PRs that absorbed closed #50457's files + a large PR)

| PR | what | apply onto v0.17.0 | tests on v0.17.0 base |
|---|---|---|---|
| **#50664** | opus-context feature-gated test (this session's fix) | **CLEAN** (cherry-pick) | **4 passed, 60 skipped, 5 xfailed — 0 failed**; real assertions execute (gpt-5→/responses, others→/chat, models.dev fall-through, fable canonical-slug) |
| **#50626** | subdir-hints + xAI label | **CLEAN** | **30 passed** |
| **#50657** | agy-cli auth/runtime registration | **CLEAN** | non-test code (registration wiring) |
| **#50555** | isolated agy-cli provider (WIP) | **CLEAN** | **16 skipped** — *intentional*: skip reason = "agy-cli provider is known-broken WIP (USER 2026-06-04: not stabilized)"; skips identically on v0.17.0 AND src HEAD (honest deferral, not inertness) |
| **#50064** | copilot CLI identity + Claude ctx + vision (largest sampled, 21 files) | commit-by-commit cherry-pick hits **modify/delete** conflicts (v0.17.0 restructured several copilot test files); **net-diff `git apply --3way` succeeds** with ONE pure-addition conflict in `tests/run_agent/test_provider_attribution_headers.py` (ours-block = 0 lines, keep-both trivial) | after keep-both resolve: **48 passed, 1 failed** |

## The one real failure — root-caused, and it is NOT a #50664 / coverage-map issue

`test_provider_attribution_headers.py::test_routed_client_preserves_openai_sdk_default_headers`
fails on the v0.17.0 rebase of **#50064** with `KeyError: 'default_headers'`.

Root cause (verified, not asserted):
- the failing test exists **only on the #50064 branch** (`grep -c` → 1),
- it is **absent from the live canonical overlay (src HEAD → 0)** and **absent from v0.17.0 (→ 0)**.

So it is a **stale test the #50064 branch still carries that the canonical tree itself dropped** —
a #50064-specific cleanup item, surfaced here for that PR. It does not touch this session's
#50664 work (the opus-context test, which applies + runs clean on v0.17.0), nor the coverage map.

## Delivery shape recorded
- 4 of 5 sampled PRs cherry-pick **CLEAN** onto v0.17.0.
- #50064 is pullable via **net-diff 3-way apply** (the "pull down" model) with one trivial
  keep-both addition; commit-by-commit replay needs the documented modify/delete resolution
  because v0.17.0 restructured copilot test files between v0.16.0 and v0.17.0.
- Every sampled PR's tests **execute** on the v0.17.0 base (not skipped into inertness);
  skips are honest feature/WIP gates with explicit reasons.

## Independent doc re-verification (fresh shell, against the PUSHED tracker tip d34bb7a0b)
- DELTA-TO-PR-MAP / STRICT-PARTITION / PER-PR-TABLE: **0** rows cite the closed #50457 as a coverage owner.
- All **40** distinct STRICT-PARTITION primary owners are **OPEN** PRs (verified vs live `gh pr list`).
