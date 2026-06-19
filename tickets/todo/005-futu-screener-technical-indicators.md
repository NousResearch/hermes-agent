# Futu Screener Technical Indicators

## Goal

Use Futu `get_stock_filter` and SDK indicator fields to expand candidate discovery beyond theme seed and plate constituents.

## Scope

- Add screener calls for:
  - liquidity
  - relative strength / change rate
  - MA/EMA alignment where SDK supports it
  - RSI/MACD/BOLL/KDJ signals where SDK supports them
- Merge screener candidates with existing seed and plate candidates.
- Preserve source tags and score breakdown.

## Acceptance Criteria

- Candidate `source_tags` can include `futu_stock_filter`.
- Candidate pool generation remains independent from current holdings.
- Tests cover merge/dedupe behavior and source attribution.

## Notes

Do not rely on screener alone. Use it as another candidate source and keep deterministic scoring.
