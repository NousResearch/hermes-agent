Phase 0 test contract

Status: baseline recorded; implementation not started
Frozen baseline: 98a4aa453c1c576798a4461d5b2902d805eef71d
Canonical worktree: /home/kensei/repos/KenseiAgent

Baseline collection
- Command: PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH .venv/bin/python -m pytest --collect-only -q tests/tools/test_async_delegation.py tests/tools/test_delegate_profile.py tests/hermes_cli/test_model_switch_configured_provider_routing.py tests/run_agent/test_provider_fallback.py
- Result: 69 tests collected; collection exit 0.

Baseline focused execution
- Command: same four test paths with --timeout=60 --timeout-method=signal -p no:cacheprovider
- Result: 61 passed, 1 skipped, 7 failed; exit 1.
- Expected RED classification: all seven failures are in tests/tools/test_delegate_profile.py and report the current absence of the profile argument/schema. They are Phase 1 conformance RED evidence, not a newly introduced Phase 0 regression.
- Passed baseline areas include async delegation persistence/capacity/restart surface, existing profile-independent delegation, flat provider-routing model switch coverage and fallback-chain behaviour.

Phase 1 RED obligations
- profile parameter and profile-content child construction;
- target-profile model/toolset/fallback resolution;
- independent ungrouped completion units;
- grouped completion units and delivery-only grouping;
- whole-call task indexes and one-slot capacity accounting;
- child-level persistence, crash recovery and failed-child notices;
- per-model provider-routing overlay and model-variant matching;
- target-profile routing isolation from parent routing;
- OpenRouter-only request payload and direct-provider/Portal exclusion;
- authority, cycle and memory invariants.

No source code, configuration, service or gateway was changed in Phase 0.
