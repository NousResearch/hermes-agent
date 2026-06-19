# Execution Plan V2

## Goal

After a user selects a target portfolio map, generate deterministic execution plans in the next version.

## Scope

- Read current portfolio only after map selection.
- Generate:
  - phased buy/sell plan
  - support/resistance based tranche triggers
  - sell put candidates
  - covered call candidates
  - simulated order parameters
- Keep all orders `SIMULATE` in V2.

## Acceptance Criteria

- No portfolio data is read before target map selection.
- All order quantities, cash usage, max risk, and trigger prices are code-computed.
- Tests cover no holding, overweight reduction, cash shortage, sell put, and covered call gating.

## Notes

This ticket depends on V1 target map review being stable.
