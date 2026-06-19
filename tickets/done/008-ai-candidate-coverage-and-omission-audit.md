# 008 - AI Candidate Coverage And Omission Audit

## Status

Done on 2026-05-20.

## Bug

The AI portfolio workflow could miss or under-explain important AI-bottleneck
candidates.

Observed from session `iaw_4b94722aa3c8416a`:

- `US.SNDK` did not enter `candidate_pool`.
- `US.COHR` entered `candidate_pool`, was Futu-enriched and eligible, but was
  not selected in any target map.
- `US.MRVL` entered `candidate_pool`, was Futu-enriched and eligible, but was
  not selected in the final pasted maps.

## Fix

- Added `initial_thesis` and `coverage_requirements` to theme discovery so the
  first LLM pass produces an active research map before Futu/SEC enrichment.
- Replaced static US AI guardrails with generic LLM-authored
  `coverage_requirements`. Discovery now asks the model to derive the
  value-chain sleeves, bottlenecks, beneficiaries, and must-consider candidates
  from the theme and user description instead of hard-coding AI-specific
  tickers or sleeves in code.
- Removed Futu/SEC coupling from the discovery prompt. Discovery is now
  provider-agnostic; Futu, SEC, filings, fundamentals, and news belong to
  enrichment and thesis synthesis stages.
- Added ASCII ticker validation to reject visually similar non-ASCII ticker
  characters.
- Added `omitted_candidates` to each portfolio map. If an eligible
  must-consider candidate is not selected, the architect must explain why and
  name substitutes when applicable.
- Surfaced important omissions in the user-facing workflow response.
- Changed theme discovery from one-shot instruction-only generation to a
  ReAct-like audit/revise loop: the discovery agent proposes a plan, code audits
  coverage/ticker validity, and the agent revises until the plan passes or the
  workflow fails explicitly.
- Inserted a `thesis_synthesis` artifact between candidate reflection and
  portfolio-map construction. The Thesis Synthesis Agent considers discovery,
  Futu, SEC/fundamental, technical, liquidity, options, market-regime, and
  risk/data-quality metrics before the Portfolio-Map Architect assigns weights.

## Verification

- `tests/plugins/test_investment_assistant.py`
- 29 tests passed.
