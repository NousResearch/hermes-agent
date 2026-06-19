# 007 - MinerU Filing Narrative Summary

## Goal

Add a qualitative filing-summary pipeline for long SEC filings and issuer PDFs.

## Scope

- Use structured SEC/companyfacts and Futu snapshot fields as the only source for
  financial numbers.
- Use MinerU to parse long filings, investor PDFs, or earnings materials into
  structured text sections.
- Use a sub LLM to summarize narrative items only: management discussion,
  risk factors, segment commentary, guidance language, capex themes, customer
  concentration, and competitive positioning.
- Mark all sub-LLM summaries as qualitative evidence and prohibit them from
  producing or overwriting key numeric fields.

## Output Contract

- `narrative_evidence.source_status`
- `narrative_evidence.parser = mineru`
- `narrative_evidence.summarizer = sub_llm`
- `narrative_evidence.numeric_extraction_allowed = false`
- `narrative_evidence.sections[]`

## Verification

- Numeric fields remain unchanged when narrative summaries are added.
- Architect prompt treats narrative summaries as qualitative context only.
- Empty narrative summaries do not block target portfolio-map generation.
