# 011 - English Futu Catalog View

## Status

Todo.

## Context

The discovery agent prompt and tool descriptions are primarily English, while
the cached Futu screener catalog currently contains many Chinese UI labels from
Futu App/OpenAPI, especially plate names and screener category labels.

This can make theme-to-catalog mapping less stable. For example, an English
theme hypothesis such as `memory_storage`, `optical_networking`, or
`power_grid` must be mapped to Chinese catalog entries such as `存储概念股`,
`光通信`, `核电`, `电气设备及零件`, or `独立电力生产商`.

## Problem

The model can usually translate these labels, but the mapping is implicit. That
creates avoidable drift:

- A relevant Chinese plate may be read but not selected for probing.
- English layer names may bias candidate compression differently from Chinese
  plate names.
- Prompt/tool examples use English filter concepts, while catalog labels may be
  Chinese.
- Refreshing the Futu catalog can reintroduce mixed-language labels.

## Proposed Direction

Keep raw Futu catalog data unchanged as source evidence, but expose a
LLM-facing English catalog view for discovery:

- English category titles and choice labels.
- English `StockField` labels and aliases.
- English plate aliases for concept/industry names.
- Preserve exact Futu enum values and plate codes.
- Preserve raw names in metadata for audit/debugging.

## Acceptance Criteria

- Discovery tools can return an English catalog view without losing raw Futu
  codes, enum values, or original names.
- English searches like `storage`, `optical networking`, `power`, `software`,
  `AI chips`, and `data center` find the relevant Futu plate codes.
- The catalog refresh script can regenerate the English view deterministically.
- Existing raw catalog snapshots remain available for audit.
- Tests cover at least the key AI-related mappings:
  - `Storage Concept Stocks` -> `US.LIST23925`
  - `Optical Communications` -> `US.LIST23979`
  - `AI Chips` -> `US.LIST2548`
  - `Internet Data Center Concept` -> `US.LIST2521`
  - `Nuclear Power` -> `US.LIST2583`

## Notes

This is not a portfolio-quality issue by itself. It is a discovery ergonomics
and reliability issue. Do not hardcode theme recommendations from these aliases;
use them only to improve catalog search and plate selection.
