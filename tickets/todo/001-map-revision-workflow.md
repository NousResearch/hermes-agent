# Map Revision Workflow

## Goal

Allow users to revise generated target portfolio maps without reading current holdings or generating orders.

## Scope

- Add a structured `portfolio_map_revision` artifact.
- Support natural-language review inputs such as:
  - exclude symbols, e.g. `不要 US.SNDK`
  - prefer roles, e.g. `提高设备链`
  - adjust cash reserve, e.g. `现金 20%`
  - cap single-name weight, e.g. `单票不超过 5%`
  - switch objective/risk tone, e.g. `更稳健`
- Rebuild `portfolio_maps` as a new artifact version.
- Return to `NEEDS_PORTFOLIO_MAP_REVIEW`.

## Acceptance Criteria

- Revision path does not call `PortfolioAdapter`.
- Revision path does not create `current_portfolio` or `construction_plan` artifacts.
- If a requested include symbol is not in `final_candidate_pool`, return a `requested_data_refresh` style warning instead of adding it.
- Tests cover exclude symbol, cash reserve change, and max single-name cap.

## Notes

Keep LLM responsibility limited to parsing revision intent. Filtering, policy mutation, and map rebuilding should be deterministic code.
