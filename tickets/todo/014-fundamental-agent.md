# 014 - Fundamental Agent

## Goal

Add a Fundamental Agent that reads discovery output plus precomputed financial
artifacts, then validates whether each candidate has enough financial,
operating, and filing evidence to remain in the theme candidate pool.

## Product Role

This is the "基本面/财报分析 Agent".

It does not discover new symbols, generate portfolio weights, judge technical
entry points, create option strategies, or produce orders. It answers:

> Does this candidate's financial and operating evidence support its role in
> the discovered theme map?

## Inputs

- `theme_discovery` artifact from AI discovery v1.
- Candidate symbols and layer mapping.
- Precomputed financial artifacts from async data jobs, for example:
  - SEC company facts.
  - latest 10-K / 10-Q metadata and selected sections.
  - earnings release summary.
  - earnings call summary.
  - segment revenue / guidance / backlog where available.
  - cash, debt, revenue, margin, net income, operating cash flow, capex.
- Artifact freshness metadata and data gaps.

## Outputs

Add a typed `fundamental_analysis` artifact containing one assessment per
candidate:

- `symbol`
- `theme_layer`
- `theme_fit`: `high | medium | low | unknown`
- `financial_quality`: `high | medium | low | unknown`
- `evidence_strength`: `high | medium | low | insufficient`
- `cycle_position`: optional business-cycle description.
- `key_evidence`: claims tied to source artifact ids.
- `red_flags`
- `data_gaps`
- `candidate_action`: `keep | watchlist | exclude | needs_refresh`
- `rationale`

The artifact should also include:

- global warnings.
- stale/missing source summary.
- candidate counts by action.
- PydanticAI runtime metadata.

## Agent Boundary

The LLM may:

- interpret financial evidence.
- judge theme fit.
- classify data gaps.
- recommend keep/watchlist/exclude for candidate-pool quality.
- explain why a candidate is strong, weak, cyclical, or incomplete.

The LLM must not:

- invent missing financial numbers.
- fetch data directly when a deterministic refresh tool is available.
- make technical-analysis calls such as RSI/MA/support/resistance.
- decide portfolio weights.
- generate trade orders.
- override deterministic freshness or source-validation checks.

## Workflow Shape

First implementation can stay decoupled from the full portfolio workflow:

```text
theme_discovery artifact
  + precomputed financial artifacts
  -> FundamentalAgent
  -> fundamental_analysis artifact
```

Later integration:

```text
THEME_DISCOVERY_COMPLETE
  -> FUNDAMENTAL_DATA_GATE
  -> FUNDAMENTAL_AGENT_ANALYSIS
  -> THESIS_SYNTHESIS
```

If required artifacts are stale or missing:

- mark the assessment `needs_refresh` when the gap is blocking.
- mark `watchlist` with data gaps when non-blocking.
- optionally emit refresh requests for async jobs.

## Tools

Start with read-only tools:

- `read_artifact`
- `inspect_artifact_freshness`
- `query_company_facts`
- `query_filing_sections`
- `query_earnings_release_summary`
- `query_earnings_call_summary`

Refresh should be separate and explicit:

- `request_fundamental_refresh`

## Non-Goals

- No technical indicator analysis. Build a separate `TechnicalAgent`.
- No portfolio-map generation. That remains `PortfolioArchitectAgent`.
- No broad web browsing in the first version unless the artifact layer marks
  data as missing and the workflow explicitly allows web research.
- No full-market financial preprocessing. Use layered data:
  universe index first, candidate enrichment second, deep research only for
  selected/core candidates.

## Acceptance Criteria

- FundamentalAgent can run from an existing discovery artifact and mocked
  financial artifacts.
- Output is typed and persisted as `fundamental_analysis`.
- Every claim with a numeric financial value references an input artifact.
- Missing/stale financial data does not produce fake conclusions.
- Candidates can be classified as `keep`, `watchlist`, `exclude`, or
  `needs_refresh`.
- Tests cover:
  - strong candidate with fresh evidence.
  - candidate with stale data marked `needs_refresh`.
  - candidate with missing evidence marked `watchlist` or `needs_refresh`.
  - candidate with weak theme fit marked `exclude`.
  - LLM failure produces workflow error, not fallback analysis.

## Design Note

This agent should remain narrow. The synthesis of discovery + fundamentals +
technical indicators + macro/events belongs to a later `ThesisSynthesisAgent`.
