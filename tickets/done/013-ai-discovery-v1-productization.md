# 013 - AI Discovery V1 Productization

## Goal

Make the successful filter-planning discovery experiment the production
discovery-only path for the investment assistant.

## Scope

- Keep V1 discovery-only: no portfolio weights, no current holdings, no SEC
  enrichment, no market-data enrichment, no orders, no options plan.
- Productize the agent behavior that first builds theme layers, then designs
  Futu screener filters per layer, executes probes, and returns auditable
  candidates.
- Preserve agent-authored discovery. Deterministic code may validate, normalize,
  rate-limit, and store artifacts, but must not author investment conclusions.

## Completed

- Added typed schema fields for filter plans, executed Futu probes, layer filter
  audits, omissions, and next enrichment needs.
- Added `plugins/investment_assistant/discovery_v1.py` as the production
  PydanticAI discovery-v1 module.
- Registered file-style Futu catalog tools:
  - `list_futu_screener_catalog`
  - `read_futu_screener_catalog`
  - `run_futu_stock_filter`
- Wired `ia_portfolio_workflow(start/discover)` to discovery v1 by default.
- Updated Hermes-facing display to explain theme layers, filter plans, Futu
  probes, candidate seeds, omissions, and downstream gaps.
- Updated tool description so Hermes knows this is AI discovery v1, not the old
  Futu-assisted path.
- Added tests covering the discovery-v1 tool registration, package conversion,
  workflow stop point, failure path, and public-action guard.

## Still Out Of Scope

- Final portfolio-map architect.
- SEC/filing enrichment.
- Market-data enrichment.
- Calibration Agent.
- Full English Futu catalog view.

## Verification

- `python -m py_compile plugins/investment_assistant/discovery_v1.py plugins/investment_assistant/workflow.py plugins/investment_assistant/schemas.py plugins/investment_assistant/tools.py tests/plugins/test_investment_assistant.py`
- `scripts/run_tests.sh tests/plugins/test_investment_assistant.py::test_ai_discovery_v1_registers_file_tools_and_converts_to_theme_plan tests/plugins/test_investment_assistant.py::test_futu_assisted_theme_discovery_registers_futu_tool tests/plugins/test_investment_assistant.py::test_workflow_start_runs_discovery_only tests/plugins/test_investment_assistant.py::test_workflow_start_accepts_free_form_theme_for_discovery tests/plugins/test_investment_assistant.py::test_workflow_discover_only_runs_theme_discovery_and_stops tests/plugins/test_investment_assistant.py::test_workflow_start_with_discovery_only_uses_discover_path tests/plugins/test_investment_assistant.py::test_workflow_preserves_required_symbols_as_discovery_inputs tests/plugins/test_investment_assistant.py::test_workflow_public_actions_stop_before_candidate_pool_and_maps tests/plugins/test_investment_assistant.py::test_discovery_failure_is_workflow_status_not_tool_crash tests/plugins/test_investment_assistant.py::test_workflow_does_not_invoke_architect_from_public_actions tests/plugins/test_investment_assistant.py::test_downstream_workflow_actions_are_not_publicly_supported -q`
- `python -m ruff check plugins/investment_assistant/discovery_v1.py plugins/investment_assistant/workflow.py plugins/investment_assistant/schemas.py plugins/investment_assistant/tools.py tests/plugins/test_investment_assistant.py`

Full `tests/plugins/test_investment_assistant.py` was attempted, but the pytest
process stopped making progress mid-file and was killed. The targeted tests
above cover the files changed in this ticket.
