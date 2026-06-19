# SEC Earnings And News Provider

## Goal

Replace placeholder event/fundamental artifacts with real structured data providers.

## Scope

- Use SEC.gov/companyfacts or a maintained SEC API wrapper for filings and fundamental metrics.
- Add an earnings calendar provider.
- Keep MinerU/document parsing as a format-specific ingestion option, not the primary SEC data source.
- Populate:
  - `earnings_event_calendar`
  - richer `fundamental_quality`
  - event-driven `risk_flags`

## Acceptance Criteria

- Artifacts explicitly show source, generated time, and stale status.
- Missing provider/data returns `not_available` or `stale`, never LLM-filled facts.
- Tests cover provider unavailable and stale data behavior.

## Notes

Start with US equities only. Do not block V1 map generation when filings/news are unavailable; return warnings and source gaps.

## Completion Notes

- Added an optional edgartools-backed SEC provider for US equity candidates.
- SEC identity is read from `SEC_EDGAR_IDENTITY` or `EDGAR_IDENTITY`; missing identity returns explicit `not_configured` artifacts.
- Populates `sec_filings_context`, enriches `fundamental_quality`, partially populates `earnings_event_calendar`, and merges filing risk flags.
- Forward earnings dates and news ingestion remain future-provider work.
