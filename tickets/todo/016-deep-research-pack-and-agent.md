# Deep Research Pack And Agent Design

## Summary

Build the stage after offline data mining. This stage reads atomic symbol data
files, assembles compact LLM-facing `research_pack.json` files, and runs the
PydanticAI deep-research agent to interpret the evidence. It is separate from
the data miner so raw data collection remains deterministic and reusable.

## Boundary

The data miner owns:

- `market.json`
- `sec_companyfacts.json`
- `filing_metadata.json`
- ETF raw/structured files when available
- `manifest.json`

This stage owns:

- `research_pack.json`
- `deep_research.json`
- LLM interpretation, summaries, evidence strength, and candidate actions

## Goals

- Build a compact per-symbol research pack from atomic data files.
- Keep exact numeric facts traceable to data files.
- Let the deep-research agent interpret, compare, and assign candidate actions.
- Produce artifacts suitable for later thesis synthesis and portfolio architect.

## Non-Goals

- No data fetching from Futu or SEC.
- No portfolio weights or trade plans.
- No direct mutation of source data files.

## Research Pack Inputs

Read from:

```text
data/investment_assistant/symbols/US.NVDA/
  manifest.json
  market.json
  sec_companyfacts.json
  filing_metadata.json
```

Optional later inputs:

```text
filing_text.json
filing_summary.json
earnings_materials.json
earnings_summary.json
news_events.json
options_surface.json
```

## Research Pack Output

```text
data/investment_assistant/symbols/US.NVDA/research_pack.json
```

It should contain compact, LLM-readable sections:

- symbol and layer role
- market snapshot summary with asof
- technical/liquidity summary
- SEC numeric facts with source paths
- filing freshness/event metadata
- missing data and stale data
- evidence ids / file references

## Deep Research Output

```json
{
  "artifact_type": "deep_research",
  "theme": "...",
  "generated_at": "...",
  "candidates": [
    {
      "symbol": "US.NVDA",
      "theme_fit": "direct | enabling | second_order | weak",
      "evidence_strength": "high | medium | low",
      "fundamental_quality": "strong | mixed | weak | unknown",
      "action": "promote | keep | watch | exclude | needs_refresh",
      "key_evidence": [],
      "red_flags": [],
      "data_gaps": [],
      "rationale": ""
    }
  ],
  "layer_summary": [],
  "promoted_symbols": [],
  "watch_symbols": [],
  "excluded_symbols": [],
  "needs_refresh_symbols": [],
  "warnings": []
}
```

## Validation Rules

- Symbols must come from the selected triage queue.
- The agent cannot invent missing numeric facts.
- Numeric claims must cite fields from atomic data files.
- No target weights, target prices, orders, or trade plans.
- Missing data must be surfaced as `data_gaps`.
- Stale data cannot be treated as fresh evidence.

## Build Later

Do this after `015-offline-data-miner` is implemented and tested.
