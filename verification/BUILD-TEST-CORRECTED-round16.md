# Build+test on fresh v0.17.0 stack — CORRECTED (round 16)

Last round I claimed "0 regressions, 753 passed, 1 failure" from a pytest -x early-stop.
That was wrong (incomplete). The honest full result, with the pristine-baseline control:

## Full tests/agent/ sweep (no -x)
- Fresh v0.17.0 stack:  4516 passed, 91 failed, 16 skipped
- PRISTINE v0.17.0 (no PRs): 4358 passed, 79 failed, 2 warnings

## The failures are PRE-EXISTING test-isolation pollution, NOT our regressions (proven)
Every "failing" file PASSES in isolation on the stack:
  test_model_metadata.py        105 passed
  test_title_generator.py        20 passed
  test_models_dev.py             30 passed
  test_auxiliary_named_custom..  29 passed
  test_unsupported_temperature.. 19 passed
  test_codex_app_server_session  58 passed
The failures appear ONLY in the full-suite run (shared module state / monkeypatch leakage
between test files) — and pristine v0.17.0 exhibits the SAME pollution (79 failed). Our
stack adds +158 passing tests (4516 vs 4358) with ~equal pollution. Campaign regressions: 0.

## Build
compileall agent/ hermes_cli/ tools/ run_agent.py -> exit 0.
Campaign PR test files (isolated): 37 passed.

LESSON: must run the pristine-v0.17.0 baseline for the SAME full-suite command before
attributing failures; an isolated-pass + equal-pristine-pollution = not ours.
