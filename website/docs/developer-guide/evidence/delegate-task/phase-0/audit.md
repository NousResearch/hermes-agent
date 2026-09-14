Phase 0 independent Kensei audit

Verdict: GO
Auditor: KENSEI
Audit method: direct re-check of repository, live services, Git objects, test logs and evidence hashes; no implementer self-report accepted as proof.

Checks passed

- `HEAD` is `98a4aa453c1c576798a4461d5b2902d805eef71d`.
- `origin/main` is exactly the same SHA.
- All 14 active `hermes-gateway*` services resolve their working directory to `/home/kensei/repos/KenseiAgent` and their Git code SHA to the frozen baseline. No service was restarted.
- Current canonical branch remains `main`; no Phase 1 worktree existed during Phase 0.
- Expanded porcelain status contains only `docs/ADR.md` and `docs/evidence/delegate-task/phase-0/*`.
- No production source path changed.
- All listed evidence hashes match the manifest.
- The four mandated test paths collect 69 tests with collection exit 0.
- Focused baseline execution records 61 passed, 1 skipped and 7 failed. All seven failures are `tests/tools/test_delegate_profile.py` profile-contract REDs, so they are expected Phase 1 RED evidence rather than a Phase 0 regression.
- Every live upstream completion/routing anchor is an ancestor of `upstream/main` and absent from KenseiAgent `HEAD`.
- `a7c0d307d7fea45be40772a9b0663ec096779398` is not an ancestor of current `upstream/main`; the source map correctly records it as branch-only/superseded.
- The ADR reconciliation section and approved in-progress plan status are present.
- No Phase 1+ implementation, merge, push, restart or deployment occurred.

Exit criteria

- Frozen baseline and deployed revision match: PASS.
- No production source changed: PASS.
- Upstream follow-ups and superseded branch coverage: PASS.
- Per-commit dispositions and Kensei preservation matrix: PASS.
- Post-update symbol/test map: PASS.
- Baseline collection/failure classification: PASS.
- ADR and test contract updated: PASS.
- Rollback point recorded: `98a4aa453c1c576798a4461d5b2902d805eef71d`.

Authority

Phase 1 may now begin in a new isolated worktree from the frozen SHA. This GO does not authorise remote push, merge to local main, service restart, activation or production deployment. Each subsequent phase requires a new independent GO / REVISE / STOP audit.
