---
title: "Xlsx — Create, read, edit Excel .xlsx workbooks and CSVs"
sidebar_label: "Xlsx"
description: "Create, read, edit Excel .xlsx workbooks and CSVs"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Xlsx

Create, read, edit Excel .xlsx workbooks and CSVs.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/xlsx` |
| Version | `1.0.0` |
| Author | Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `excel`, `spreadsheet`, `xlsx`, `csv`, `openpyxl`, `productivity` |
| Related skills | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Xlsx Skill

Work with Excel .xlsx workbooks using Python and openpyxl: build styled
multi-sheet workbooks with formulas and charts, inspect or dump existing
files, edit cells and structure, and convert to/from CSV. All helper
scripts are argparse CLIs that print JSON and use explicit UTF-8 I/O.

## When to Use

- Creating .xlsx reports: multiple sheets, number formats, styling,
  merged cells, freeze panes, autofilter, conditional formatting,
  charts, data-validation dropdowns.
- Reading a workbook: sheet inventory, dumping data as JSON or CSV,
  listing formulas vs cached values.
- Editing existing files: set cells, append rows, insert/delete
  rows/columns, copy/rename sheets.
- CSV interop with type inference and non-UTF-8 encodings.
- Not for the legacy .xls binary format (use LibreOffice to convert
  first: `soffice --headless --convert-to xlsx old.xls`).

## Prerequisites

- Python 3.10+ with `openpyxl` (`pip install openpyxl`). No other
  third-party packages are needed; everything else is stdlib.
- Optional: LibreOffice (`soffice`) for headless recalculation or
  format conversion.

## How to Run

Run the helper scripts with the `terminal` tool from this skill's
`scripts/` directory (every script supports `--help`):

```bash
python scripts/xlsx_create.py spec.json report.xlsx   # build from JSON spec
python scripts/xlsx_read.py report.xlsx --sheets      # inventory
python scripts/xlsx_read.py report.xlsx --json --sheet Data
python scripts/xlsx_read.py report.xlsx --formulas
python scripts/xlsx_edit.py report.xlsx --sheet Data --set B2=42 --recalc
python scripts/csv_to_xlsx.py data.csv out.xlsx --encoding utf-8
python scripts/xlsx_to_csv.py report.xlsx out.csv --sheet Data
```

Author the JSON spec with `write_file`, inspect script JSON output with
`read_file` or directly from stdout.

## Quick Reference

| Task | Command |
|---|---|
| Create workbook from spec | `xlsx_create.py spec.json out.xlsx` |
| Sheet names + dimensions | `xlsx_read.py f.xlsx --sheets` |
| Dump sheet as JSON | `xlsx_read.py f.xlsx --json --sheet S` |
| Dump sheet as CSV | `xlsx_read.py f.xlsx --csv --out d.csv` |
| List formulas + cached values | `xlsx_read.py f.xlsx --formulas` |
| Set a cell / formula | `xlsx_edit.py f.xlsx --set "A1==SUM(B:B)"` |
| Append a row | `xlsx_edit.py f.xlsx --append '[1,"x",true]'` |
| Insert 2 rows before row 3 | `xlsx_edit.py f.xlsx --insert-rows 3:2` |
| Copy / rename sheet | `--copy-sheet Src:New --rename-sheet Old:New` |
| Force recalc on open | `xlsx_edit.py f.xlsx --recalc` |
| CSV -> styled xlsx | `csv_to_xlsx.py in.csv out.xlsx` |
| xlsx -> CSV | `xlsx_to_csv.py f.xlsx out.csv --encoding utf-8` |

## Procedure

1. **Create**: write a JSON spec (schema documented in
   `xlsx_create.py --help` and its docstring). Each sheet supports
   `rows` (scalars or styled cell objects), sparse `cells` overrides,
   `column_widths`, `row_heights`, `merges`, `freeze_panes`,
   `autofilter`, `conditional_formats` (cell_is rules and color
   scales), `charts` (bar/line/pie from cell ranges), and
   `validations` (list dropdowns). Typed values: JSON numbers/bools
   pass through; dates use `{"value": "2026-01-31", "type": "date"}`.
   Number formats are Excel format strings: currency `"$#,##0.00"`,
   percent `"0.0%"`, date `"yyyy-mm-dd"`.
2. **Formulas**: set with `"formula": "SUM(B2:B9)"` in the spec or
   `--set "C1==SUM(A:A)"` in the editor. When writing formulas, add
   `"full_calc_on_load": true` (spec) or `--recalc` (editor); this sets
   the workbook's `fullCalcOnLoad` flag so Excel/LibreOffice recompute
   everything on open. openpyxl itself NEVER evaluates formulas.
3. **Read**: `--sheets` for inventory (names, dimensions, merged
   ranges, chart count), `--json`/`--csv` for data, `--formulas` to
   pair each formula string with its cached result. Cached results
   exist only if the file was last saved by a real spreadsheet app;
   files fresh from openpyxl return `null` there. To materialize
   results headlessly: `soffice --headless --convert-to xlsx file.xlsx`
   then reload with `--data-only`.
4. **Edit**: `xlsx_edit.py` applies renames/copies first, then
   structural row/column changes, then `--set`/`--append`. It edits in
   place unless `--out` is given — copy the file first if you need the
   original.
5. **CSV interop**: `csv_to_xlsx.py` infers int/float/bool/ISO-date
   per cell and styles the header row; `xlsx_to_csv.py` writes ISO
   dates and blank strings for empty cells. Both default to UTF-8 and
   accept `--encoding` (e.g. `utf-8-sig` for Excel-friendly BOM,
   `cp1252` for legacy Windows exports).

## Pitfalls

- **openpyxl does not calculate.** Formula results are available only
  via `load_workbook(path, data_only=True)` and only when the file was
  previously saved by Excel/LibreOffice. Otherwise you get `None`.
- **Insert/delete does not shift references.** `insert_rows`,
  `delete_cols`, etc. move cell values but do NOT update merged-cell
  ranges, formula references, chart anchors, or conditional-format
  ranges. After structural edits on sheets with merges or formulas,
  re-check them with `--sheets` and `--formulas` and fix manually.
- **`data_only=True` then save** silently discards all formulas
  (cached values replace them). Never save a workbook loaded that way
  unless that is the goal.
- **Loading strips charts/images**: openpyxl does not round-trip
  charts, so editing a charted workbook and saving drops the charts.
  Re-add charts after editing, or avoid re-saving charted files.
- **CSV locale traps**: always pass explicit encodings (the scripts
  already do) and remember European CSVs often use `;` delimiters and
  decimal commas — use `--delimiter ';'` and expect strings like
  `"12,5"` to stay strings.
- **Dates are datetimes**: Excel stores dates as serial numbers;
  openpyxl returns `datetime`/`date` objects. Dumps here emit ISO
  strings.
- Sheet names are capped at 31 chars and reject `[ ] : * ? / \`.

## Verification

- After creating: `xlsx_read.py out.xlsx --sheets` and confirm sheet
  names, dimensions, merged ranges, and chart counts match intent.
- Dump data with `--json` and compare against the source values.
- After edits: re-dump the touched range; if formulas were written,
  confirm `--formulas` lists them and that `--recalc` was applied.
- For a full visual check, open in LibreOffice:
  `soffice --headless --convert-to pdf out.xlsx` and inspect the PDF.
