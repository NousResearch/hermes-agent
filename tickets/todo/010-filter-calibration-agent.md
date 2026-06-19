# 010 - Filter Calibration Agent

## Goal

Add a Filter Calibration Agent that turns LLM-selected Futu ranking probes into
auditable calibrated filters or rank-then-score plans.

## Background

Theme discovery can now choose relevant Futu screener fields such as turnover,
market value, revenue growth, gross margin, operating margin, ROIC, debt ratio,
and technical patterns. However, the current `stock_filter_specs` usually only
set field, period, quarter, and sort direction. They do not choose concrete
numeric thresholds such as `filter_min` or `filter_max`.

Without calibration, these specs are ranking probes rather than true filters.
Important candidates can still be lost later because downstream execution may
only enrich a small top-N slice or because overly strict guessed thresholds
would exclude cyclical names.

Update from the filter-planning discovery experiment: the current standalone
discovery agent can now produce a usable V1 candidate-discovery trace by
choosing subdomain-specific Futu filters directly. For the AI map test it used
plate relevance plus market cap, 20-day turnover, recent revenue growth, and
gross margin thresholds, and it successfully surfaced previously missed
bottleneck candidates such as `US.SNDK`, `US.WDC`, `US.COHR`, `US.LITE`, and
`US.MRVL`.

This is acceptable for the first version as a discovery-only candidate pool.
Calibration remains important, but it is not a blocker for the initial MVP as
long as the output is clearly labeled as uncalibrated discovery rather than a
final portfolio recommendation.

## Scope

- Add a `filter_calibration` workflow artifact.
- Add typed schemas for:
  - `CalibrationInputProbe`
  - `CalibrationTrial`
  - `CalibrationTrialResult`
  - `CalibratedFilter`
  - `FilterCalibrationArtifact`
- The Calibration Agent reads:
  - `normalized_user_intent`
  - `investment_constraints`
  - `workflow_guidance.calibration`
  - `theme_discovery` / query-plan artifact
  - available Futu screener catalog docs
- The agent may propose multiple trial filters per probe.
- Deterministic code executes each trial through Futu `get_stock_filter`.
- Each trial records:
  - filter specs
  - returned count
  - sample symbols
  - focus/must-check symbols included or missing
  - warnings/errors
  - whether the result is too broad, too narrow, unstable, or usable
- The agent selects one of:
  - `calibrated_filter` with numeric thresholds
  - `rank_then_score` when thresholds are unstable or subdomain-specific
  - `skip_probe` when the probe cannot be executed safely
- Persist all rejected trials and final selection rationale.

## Required Behavior

- Do not let the LLM invent unsupported Futu fields.
- Do not require numeric thresholds when ranking is safer.
- Do not use the same absolute thresholds across unrelated subdomains such as
  software, storage, optical communication, and power equipment.
- Limit trials per probe to avoid Futu rate-limit stalls.
- Prefer broad-to-narrow calibration:
  1. baseline ranking trial
  2. moderate quality/liquidity thresholds
  3. stricter threshold trial if the result remains too broad
  4. fallback to rank-then-score if thresholds are brittle
- Preserve important bottleneck candidates even if they fail strict filters by
  recording them in an omission/focus-symbol audit.
- Treat the current filter-planning discovery output as a valid V1 input when:
  - each important subdomain has an explicit filter plan,
  - every executed Futu probe is persisted with exact `stock_filter_specs`,
  - important omitted symbols are listed in an omission audit,
  - the artifact states that thresholds are LLM-selected and not yet calibrated.

## Acceptance Criteria

- Calibration can run on discovery probes without generating portfolio maps.
- Each executed trial is stored with Futu call arguments and result summary.
- A zero-result trial does not silently eliminate a probe; it is recorded and
  followed by a relaxed retry or `rank_then_score`.
- Calibration output includes final selected mode per probe:
  `calibrated_filter`, `rank_then_score`, or `skip_probe`.
- Tests cover:
  - broad trial narrowed by thresholds
  - zero-result trial relaxed
  - subdomain-specific thresholds
  - missing focus symbol captured in audit
  - unsupported field rejected before Futu execution

## Notes

LLM owns exploration strategy and trial interpretation. Deterministic code owns
Futu execution, field validation, result summarization, rate limits, and
persistence.

MVP stance: do not block the first discovery-only release on Calibration Agent.
The first version can use the filter-planning discovery agent's thresholds as
auditable exploratory probes. Calibration should be the next quality upgrade
before relying on the probes for candidate compression, scoring, portfolio-map
architect context, or any build/reduce/order-planning workflow.

## Progress

- Added first-pass calibration schemas in `plugins/investment_assistant/schemas.py`.
- Added `plugins/investment_assistant/filter_calibration.py` with:
  - PydanticAI planner and selector boundaries.
  - Futu `get_stock_filter` execution.
  - Cached screener-catalog field validation before Futu calls.
  - Trial diagnosis, focus-symbol audit, and artifact validation.
- Added focused tests for unsupported field rejection, artifact construction,
  selected-trial validation, and stock-filter compatibility.

Still pending:

- Wire `filter_calibration` into the workflow state machine and SQLite artifact
  persistence.
- Add live/recorded calibration diagnostics for broad-to-narrow and zero-result
  retry behavior.
- Feed calibrated filters into candidate-pool compression so important
  bottleneck candidates do not disappear downstream.
- Add a replay test using the AI filter-planning trace where the uncalibrated
  discovery output is accepted for V1, then calibration is run as an optional
  follow-up artifact.
