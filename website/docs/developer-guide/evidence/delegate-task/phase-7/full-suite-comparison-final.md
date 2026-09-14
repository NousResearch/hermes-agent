Phase 7 final full-suite comparison

Candidate source endpoint: 2cefc2d5a0f3b4ff5073f178ea8abfc6610d70a3
Candidate branch: delegate-task-phase-7-release
Candidate worktree: /home/kensei/repos/KenseiAgent-worktrees/delegate-task-phase-7
Frozen deployed baseline: 98a4aa453c1c576798a4461d5b2902d805eef71d
Baseline worktree: /tmp/KenseiAgent-baseline-repaired

Command used for both source trees:
HERMES_TEST_WORKERS=8 HERMES_TEST_FILE_RETRIES=0
.venv/bin/python scripts/run_tests_parallel.py -j 8 --file-timeout 300 --file-retries 0 -- -q --timeout=60 --timeout-method=signal -p no:cacheprovider

Candidate final result:
- 3,969 files
- 47,926 passed
- 572 failed
- 447 skipped
- 100% complete
- exit 1
- runtime 1,888.8 seconds

Repaired baseline result:
- 3,965 files
- 47,895 passed
- 583 failed
- 447 skipped
- 100% complete
- exit 1
- runtime 2,307.5 seconds

Failure-set comparison:
- Candidate-only failed test IDs: 0
- Baseline-only failed test IDs: 11
- Common failed test IDs: 572
- Candidate-only regression: none

Baseline-only failures:
- tests/tools/test_code_kernel_remote.py::TestIdleReapAndCapEviction::test_eviction_skips_kernels_with_a_running_cell
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_build_child_agent_accepts_profile
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_explicit_model_wins_over_profile
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_profile_fallback_chain_loaded
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_profile_model_overrides_parent
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_profile_model_wins_over_delegation_config
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_profile_toolsets_intersected_with_parent
- tests/tools/test_delegate_profile.py::TestProfileParameter::test_schema_has_profile_parameter
- tests/tools/test_delegate_toolset_scope.py::TestToolsetStarvationBackstop::test_flat_model_config_does_not_crash
- tests/tools/test_g4_retained_fleet_smoke_matrix.py::TestSyncDelegation::test_S3_bad_target_no_config
- tests/tui_gateway/test_compute_host_turn_protocol.py::test_turn_start_streams_deltas_then_turn_end_with_history_identity

Interpretation:
- The eight profile baseline failures are expected improvements covered by the candidate.
- The G4 S3 baseline failure is fixed by profile preflight ordering.
- The remaining two baseline-only failures are unrelated timing/environment failures.
- The candidate's earlier four reproducible candidate-only failures were fixed in 2cefc2d5a0; exact rerun: 4 passed.
- The relay metrics candidate-only failure was a transient SQLite lock; exact rerun passed on candidate and baseline.

Raw logs:
- candidate-full-isolated-final.txt
- baseline-full-isolated-repaired.txt
- regression-fix-exact.txt
- target-matrix-after-fix.txt

This is engineering evidence, not final release approval. The original requirements-building agent must independently audit this candidate before any final green light, push, merge, activation or deployment.
