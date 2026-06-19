# PydanticAI Portfolio Architect

## Goal

Replace the deterministic placeholder map architect with a typed PydanticAI agent while preserving artifact and schema constraints.

## Scope

- Input is structured artifacts only:
  - `final_candidate_pool`
  - `theme_exposure_map`
  - `market_regime`
  - `fundamental_quality`
  - `valuation_context`
  - `correlation_and_diversification`
  - `benchmark_context`
  - `options_surface`
- Output must validate as `PortfolioMaps`.
- Agent cannot add symbols outside `final_candidate_pool`.
- Agent can flag missing exposure or data gaps.

## Acceptance Criteria

- `build_portfolio_maps` uses PydanticAI as the only map generation path.
- Missing `pydantic_ai`, missing model configuration, missing API key, model failure, or invalid agent output fails the workflow explicitly.
- No deterministic portfolio-map fallback is generated.
- Tests cover PydanticAI architect failure, schema/constraint validation through workflow failure, and normal typed map creation with a mocked architect.
- A real smoke test confirms PydanticAI can call the configured `gpt-5.5` model and return typed `PortfolioMaps`.

## Notes

Do not send raw K-line tables or full option chains to the LLM. Send compact artifacts.

## Completion Notes

- Added `pydantic-ai==1.96.0` to Hermes lazy dependency allowlist.
- Implemented the PydanticAI architect in `plugins/investment_assistant/agents.py`.
- Added hard validation for out-of-pool symbols, duplicated holdings, missing required symbols, cash reserve changes, sleeve overflow, and single-name limit breaches.
- Confirmed the real PydanticAI path returned 3 typed maps from `gpt-5.5` in a smoke test.
