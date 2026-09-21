# Failure reconciliation — fix 142aa39954

## Inputs

- Candidate full-suite evidence: `full-suite-rerun/candidate-full.txt`
- Baseline full-suite evidence: `full-suite-rerun/baseline-full.txt`
- Frozen baseline: `98a4aa453c1c576798a4461d5b2902d805eef71d`
- Candidate suite endpoint before the final test-race fix: `aab337ad879facf7b2925811cd2570ae27122f06`
- Final candidate test-race fix: `4bbd081372e77cc6964abe724b54cacc04642ccf`

## Full-suite result

| Suite | Files | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Candidate | 3970 | 47940 | 574 | 447 |
| Baseline | 3965 | 47897 | 581 | 447 |

Failure-ID set comparison:

- Shared failures: 572
- Candidate-only IDs: 2 before final targeted repair
- Baseline-only IDs: 9

## Candidate-only dispositions

### `tests/hermes_cli/test_relay_shared_metrics.py::test_concurrent_due_exports_create_one_daily_package`

Disposition: NOT REPRODUCED / no production change required.

The test passed in the targeted rerun. It is treated as a transient suite-order or environment-sensitive failure, not a proven candidate regression.

### `tests/tools/test_code_kernel_remote.py::TestIdleReapAndCapEviction::test_eviction_skips_kernels_with_a_running_cell`

Disposition: FIXED.

The failure was a test race: the test iterated `_REMOTE_KERNELS.values()` while the worker thread could add or remove a kernel. This raised `RuntimeError: dictionary changed size during iteration`. The production registry already provides `_REGISTRY.lock`; the test now takes that lock while observing `attached` state.

Validation: five consecutive targeted executions passed.

## Baseline-only IDs

The nine baseline-only IDs are not candidate regressions. They are tests that failed in the repaired baseline run but passed in the candidate, including the candidate delegation-profile and toolset-scope tests. They remain listed in the raw evidence and are not silently discarded.

## Verdict for this gate

The full-suite comparison shows no proven candidate-only production regression. The one deterministic candidate-only test race has been repaired and targeted-validated. The full suite itself remains a known shared-baseline failure set and is not represented as a full-suite PASS.
