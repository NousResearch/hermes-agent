Phase 1 conformance-harness result

Status: implementation evidence complete; audit pending
Base: 98a4aa453c1c576798a4461d5b2902d805eef71d
Branch: delegate-task-phase-1-profile-and-units
Worktree: /home/kensei/repos/KenseiAgent-worktrees/delegate-task-phase-1

Scope

Test-only Phase 1 harness. No production source, configuration, gateway, service,
merge, push or deployment was changed.

Authoritative collection

- Command: env -u PYTHONPATH .venv/bin/python -m pytest --collect-only -q
- Files: per-model routing, completion units, profile runtime scope, profile contract,
  async delegation regression
- Result: 53 tests collected; exit 0
- Raw evidence: raw/phase-1-collection-authoritative.txt

Authoritative RED

- Command: env -u PYTHONPATH .venv/bin/python -m pytest -q <same files>
- Result: exit 1 by design; 17 intended assertion-level REDs, 36 passes, 1 skip,
  1 compatibility warning
- RED targets: per-model overlay/variant resolution and target-profile routing;
  completion-unit partitioning, shared slot/task indexes and partial child notices;
  profile schema, toolset/provider/credential/fallback scope
- No import, syntax, fixture or setup failures occurred.
- Raw evidence: raw/phase-1-red-authoritative.txt

Regression GREEN

- Command: env -u PYTHONPATH .venv/bin/python -m pytest -q tests/tools/test_async_delegation.py
- Result: 27 passed, 1 skipped, 1 compatibility warning; exit 0
- Raw evidence: raw/phase-1-green-regression.txt

Expected baseline compatibility warning

HermesPluginCompatWarning for agent.anthropic_adapter's compatibility import;
this is pre-existing and unrelated to the Phase 1 patch.

Delegation note

Two generic implementation-lane attempts were stopped after research-only output;
Codex CLI was unavailable. KENSEI completed the bounded test-only harness directly
in this isolated branch, with the same no-production-edit boundary.

Files changed by this phase

- tests/agent/test_per_model_provider_routing.py — new
- tests/tools/test_delegate_completion_units.py — new
- tests/tools/test_delegate_profile_runtime_scope.py — new
- tests/tools/test_delegate_profile.py — rewritten to assertion-level profile seam tests
- docs/evidence/delegate-task/phase-1/ — raw logs and this summary

Next gate

Independent Kensei audit must return GO before Phase 2. Phase 2 must implement only
after this RED evidence is preserved. Do not merge, push, activate or deploy.
