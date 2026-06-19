# 017 - FMP Supplemental Data Provider

## Status

Done.

## Completed

- Added `plugins/investment_assistant/fmp_provider.py`.
- Added optional data-miner layers:
  - `fmp_profile`
  - `fmp_etf`
  - `fmp_analyst`
  - `fmp_transcripts`
  - `fmp_metrics`
  - `fmp_peers`
  - `fmp_ownership`
  - `fmp_insider`
  - `fmp_all`
- Kept FMP out of `DEFAULT_LAYERS`, so the SEC miner still works without
  `FMP_API_KEY`.
- Reads `FMP_API_KEY` from Hermes `.env`; non-secret settings are under
  `investment_assistant.fmp` in `config.yaml` with `IA_FMP_*` environment
  overrides kept only as compatibility/debug escapes.
- Added provider-level request spacing via
  `investment_assistant.fmp.rate_limit_delay_seconds` /
  `IA_FMP_RATE_LIMIT_DELAY_SECONDS` after live FMP batch testing exposed `429`
  rate-limit responses.
- Added endpoint metadata, timestamps, warnings, source status, raw payloads,
  and deterministic helper summaries.
- Added skipped/unavailable artifacts for missing keys, unsupported symbol
  paths, endpoint failures, and provider exceptions.
- Added miner manifest integration for FMP artifacts.
- Added tests for missing key, ETF success, partial endpoint failure, and miner
  manifest integration.

## Validation Results

```bash
scripts/run_tests.sh tests/plugins/test_investment_assistant.py -q
# 94 passed
```

## Context

Financial Modeling Prep can fill several gaps that are awkward to cover with
Futu or raw SEC filings alone. It should be treated as a supplemental data
provider, not the source of truth for real-time quotes or official filing
numbers.

Current provider boundaries:

- Futu: real-time market data, K-line, quote, options, screener, technical
  state.
- SEC / edgartools: official companyfacts, filing metadata, raw filing text,
  filing sections, numeric provenance.
- FMP: normalized market expectations, ETF exposure, peer groups, transcripts,
  ownership and comparable metrics.

## Goals

- Add an `FmpProvider` that fetches deterministic JSON artifacts for selected
  symbols or ETFs.
- Use FMP to fill gaps that matter for portfolio-map research:
  - ETF holdings and exposure for QQQ / SOXX / SMH.
  - Analyst estimates, price targets, ratings / grades.
  - Earnings call transcripts and transcript lists.
  - Normalized key metrics, ratios, financial scores.
  - Peer groups and screener-derived comparable sets.
  - Institutional ownership / 13F summary.
  - Insider trading summary.
- Preserve source timestamps, provider metadata, endpoint names, and warnings.
- Keep FMP data out of the final investment conclusion unless a later AI agent
  explicitly reads the artifact and cites it as supporting evidence.

## Non-Goals

- Do not replace SEC companyfacts as the authoritative source for filing-derived
  financial numbers.
- Do not persist Futu real-time market data through this provider.
- Do not generate portfolio weights, trade plans, or recommendations in the FMP
  provider.
- Do not require FMP for the existing offline SEC miner to work.

## Proposed Artifacts

Write FMP artifacts under each symbol data directory, for example:

```text
data/investment_assistant/<run>/symbols/US.NVDA/
  fmp_company_profile.json
  fmp_normalized_metrics.json
  fmp_analyst_expectations.json
  fmp_earnings_transcripts.json
  fmp_peer_group.json
  fmp_ownership_signal.json
  fmp_insider_signal.json

data/investment_assistant/<run>/symbols/US.QQQ/
  fmp_etf_exposure.json
```

Suggested artifact shapes:

- `fmp_etf_exposure.json`
  - ETF profile.
  - Holdings with weights.
  - Sector exposure.
  - Country exposure.
  - Concentration summary.
  - Overlap helper fields for later portfolio analysis.
- `fmp_analyst_expectations.json`
  - Annual and quarterly revenue / EPS estimates.
  - Price target consensus / summary.
  - Ratings or grades snapshot.
  - Latest upgrades / downgrades if available.
- `fmp_earnings_transcripts.json`
  - Transcript list.
  - Latest transcript text and metadata.
  - Fiscal quarter / year.
  - Source freshness.
- `fmp_normalized_metrics.json`
  - TTM ratios.
  - Key metrics.
  - Financial scores.
  - Growth metrics.
- `fmp_peer_group.json`
  - FMP peer list.
  - Source endpoint and as-of.
  - Optional peer classification notes for later LLM use.
- `fmp_ownership_signal.json`
  - Institutional ownership positions summary.
  - Holder analytics where available.
  - 13F quarter / date metadata.
- `fmp_insider_signal.json`
  - Recent insider transactions.
  - Aggregated buy / sell counts and dollar values.

Each artifact must include:

- `artifact_type`
- `symbol`
- `provider = "financialmodelingprep"`
- `generated_at`
- `source_status`
- `endpoints`
- `data_asof`
- `warnings`
- `raw` or structured payload, depending on endpoint stability

## Implementation Notes

- Add a provider module, likely
  `plugins/investment_assistant/fmp_provider.py`.
- Add miner integration as optional layers, likely:
  - `fmp_etf`
  - `fmp_analyst`
  - `fmp_transcripts`
  - `fmp_metrics`
  - `fmp_peers`
  - `fmp_ownership`
  - `fmp_insider`
- Add config/env support:
  - `FMP_API_KEY` secret in `.env`.
  - Non-secret timeouts, rate limits, enabled layers in config if needed.
- Add rate limiting and retry/backoff. Treat HTTP 429 and quota errors as
  recoverable artifact warnings, not crashes for the whole batch.
- Keep endpoint responses small enough for downstream LLM use by storing raw
  data separately or summarizing only deterministically.

## Priority

P0:

- ETF holdings / sector / country exposure.
- Analyst estimates and price target / ratings snapshot.
- Earnings transcripts.

P1:

- Normalized ratios, key metrics, financial scores.
- Peer groups and screener-based comparable sets.
- Institutional ownership / 13F summary.

P2:

- Insider trading summary.
- ESG / DCF / macro calendar integrations.

## Validation

- Missing `FMP_API_KEY` should create skipped / unavailable layer artifacts, not
  fake data.
- ETF artifact should populate for QQQ and SOXX and include holdings weights.
- Analyst artifact should preserve endpoint metadata and not overwrite SEC
  companyfacts.
- Transcript artifact should include latest transcript metadata and text when
  available.
- Batch miner should still complete if one FMP endpoint fails for one symbol.
- Tests should cover provider success, API-key missing, quota/rate-limit error,
  ETF vs operating-company paths, and manifest layer status.

## Open Questions

- Which FMP plan is required for ETF holdings, transcripts, analyst estimates,
  and 13F data?
- Should transcript text be stored directly in JSON, separate `.md`, or both?
- Should FMP bulk endpoints be preferred for large batches once the schema is
  stable?
- How should FMP normalized ratios be reconciled with SEC-derived ratios when
  they disagree?
