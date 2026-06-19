# Hermes Multiturn Routing For Portfolio Review

## Goal

Make Hermes reliably continue the same investment-assistant workflow when the user responds after map review.

## Scope

- Improve tool schema descriptions and `next_instruction_for_agent`.
- Clarify that after `NEEDS_PORTFOLIO_MAP_REVIEW`, the next user message can be either:
  - selecting an `option_id`
  - requesting revision
  - asking for explanation
- Ensure workflow can resume latest session by tenant when `session_id` is omitted.

## Acceptance Criteria

- Real LLM smoke test: user starts a theme map, then says `不要 SNDK，现金 20%`; Hermes calls `ia_portfolio_workflow` again.
- Audit output shows same workflow session is used.
- No duplicate workflow session is created for a follow-up review message.

## Notes

This is partly prompt/schema design and partly workflow affordance. Keep Hermes as orchestrator; durable state remains in the plugin workflow DB.
