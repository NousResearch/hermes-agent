---
title: "Pdf — Create, read, merge, fill, and secure PDF files"
sidebar_label: "Pdf"
description: "Create, read, merge, fill, and secure PDF files"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pdf

Create, read, merge, fill, and secure PDF files.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/pdf` |
| Version | `1.0.0` |
| Author | Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `pdf`, `documents`, `forms`, `reportlab`, `pypdf`, `pdfplumber` |
| Related skills | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint), [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# PDF Skill

Create PDFs from structured specs, extract text/tables/metadata, merge/split/rotate/watermark pages, fill AcroForm form fields, and encrypt/decrypt — using pypdf, reportlab, and pdfplumber. Scanned (image-only) PDFs contain no text layer: OCR is explicitly out of scope here — when a page is image-only, stop and use the `ocr-and-documents` skill instead of pretending to extract text.

## When to Use

- Generate a report, invoice, or multi-page document as PDF.
- Pull text, tables (JSON/CSV), metadata, or form-field values out of a PDF.
- Merge, split, rotate, extract page subsets, watermark, bookmark, or compress PDFs.
- Fill or flatten AcroForm forms; encrypt or decrypt with passwords.
- NOT for scanned/image-only PDFs (use `ocr-and-documents`) and NOT for pixel-perfect HTML-to-PDF rendering (use a headless browser).

## Prerequisites

- Python 3.10+ with `pypdf`, `reportlab`, `pdfplumber`:
  `python3 -m pip install pypdf reportlab pdfplumber`
- Each helper script checks imports lazily and prints an install hint if a dependency is missing.

## How to Run

All helpers live in `scripts/` and are argparse CLIs — run them with the `terminal` tool; every one supports `--help`. They read/write JSON strictly as UTF-8, print JSON results to stdout, and exit non-zero on failure.

```bash
python3 scripts/pdf_create.py spec.json -o out.pdf         # build PDF from JSON spec
python3 scripts/pdf_read.py doc.pdf --text                 # per-page text (JSON)
python3 scripts/pdf_read.py doc.pdf --tables --csv-dir t/  # tables to JSON + CSV files
python3 scripts/pdf_read.py doc.pdf --meta                 # metadata, page sizes, encrypted/scanned flags
python3 scripts/pdf_read.py form.pdf --fields              # form fields: name, type, value
python3 scripts/pdf_merge.py a.pdf b.pdf -o merged.pdf [--bookmarks]
python3 scripts/pdf_split.py doc.pdf --pages 1-3,7 -o part.pdf [--rotate 90]
python3 scripts/pdf_fill_form.py form.pdf --fields-json values.json -o filled.pdf [--flatten]
python3 scripts/pdf_secure.py doc.pdf --encrypt -o enc.pdf --user-password your-password
python3 scripts/pdf_secure.py enc.pdf --decrypt -o dec.pdf --password your-password
python3 scripts/pdf_watermark.py doc.pdf --stamp mark.pdf -o stamped.pdf [--under]
```

## Quick Reference

| Task | Tool | Command / API |
|---|---|---|
| Create doc (headings, tables, images) | reportlab platypus | `pdf_create.py spec.json -o out.pdf` |
| Per-page text | pdfplumber | `pdf_read.py f.pdf --text` |
| Tables → JSON/CSV | pdfplumber | `pdf_read.py f.pdf --tables` |
| Metadata / sizes / encrypted / scanned | pypdf + pdfplumber | `pdf_read.py f.pdf --meta` |
| Merge (+ outline) | pypdf | `pdf_merge.py a.pdf b.pdf -o m.pdf` |
| Split / extract / rotate | pypdf | `pdf_split.py f.pdf --pages 2-5 --rotate 90` |
| List / fill / flatten form | pypdf | `pdf_read.py --fields`, `pdf_fill_form.py` |
| Encrypt / decrypt (AES-256) | pypdf | `pdf_secure.py --encrypt/--decrypt` |
| Watermark / stamp | pypdf | `pdf_watermark.py f.pdf --stamp w.pdf` |
| Compress content streams | pypdf | `pdf_split.py f.pdf --pages 1-N --compress` |

## Procedure

1. **Inspect first.** Run `pdf_read.py file.pdf --meta`. Check `encrypted` (if true, decrypt first with `pdf_secure.py --decrypt`) and `likely_scanned_pages`. If pages are image-only, hand off to the `ocr-and-documents` skill — do not report empty text as "no content".
2. **Create.** Write a JSON spec with `write_file` (elements: `heading`, `paragraph`, `table`, `image`, `pagebreak`; optional `title`/`author` metadata; page numbers are added automatically), then run `pdf_create.py`. Verify visually with `vision_analyze` on a rendered page image if layout matters.
3. **Extract.** `--text` gives a JSON list of per-page strings; `--tables` gives row arrays per page and can also emit CSV files. Read results with `read_file`; never eyeball a binary PDF directly.
4. **Manipulate.** `pdf_merge.py` concatenates and can add one bookmark per source file; `pdf_split.py` handles page ranges (1-based, e.g. `1-3,5,9-`), rotation in 90° steps, and `--compress`. Watermark by preparing a single-page stamp PDF (e.g. via `pdf_create.py`) and overlaying it with `pdf_watermark.py`.
5. **Forms.** List fields (`--fields`) to learn exact names and types, write a UTF-8 JSON of `{"FieldName": "value"}` with `write_file` (checkboxes accept `true`/`false`; radio/choice values must match the field's export options), then `pdf_fill_form.py`. Re-read with `--fields` to confirm values landed.
6. **Secure.** Encrypt with distinct user/owner passwords and AES-256. To remove a password you know, `--decrypt` writes an unencrypted copy.
7. **Verify** (see below) before reporting success.

## Pitfalls

- **Scanned PDFs**: empty `extract_text()` plus page images means there is no text layer. Route to `ocr-and-documents`; do not fabricate text.
- **Flattening limits**: `pdf_fill_form.py --flatten` uses pypdf's flatten support, which converts widget appearances into page content. It is reliable for plain text fields and checkboxes but can drop or misrender exotic widgets (rich text, custom appearance streams, some radio groups). Verify the flattened output visually with `vision_analyze`; for bulletproof flattening use an external renderer (e.g. Ghostscript or `pdftoppm`+reassembly) as a fallback.
- **NeedAppearances**: after filling, viewers only render values if appearance streams exist. The fill script sets the AcroForm `NeedAppearances` flag so conforming viewers regenerate them; some minimal viewers ignore it — flatten if display fidelity matters.
- **Non-Latin form values**: values are stored correctly (UTF-16), but the field's default font may lack glyphs, so a viewer can show blanks even though the data round-trips. Verify with `--fields`, not just visually.
- **Compression expectations**: `--compress` only deflates content streams. Typical savings are 0–20%; it does nothing for PDFs dominated by images or already-compressed streams. It is not a substitute for image downsampling (Ghostscript territory).
- **Permission flags don't enforce**: owner-password permission bits (no-print, no-copy) are polite requests that viewers may honor; any library (including pypdf) can read and strip them. Only the user password actually gates content via encryption. Never present permission flags as security.
- **Table extraction is heuristic**: pdfplumber detects tables from ruling lines/word alignment; borderless or merged-cell tables may need `table_settings` tuning or manual cleanup.
- **Page indexing**: helper CLIs take 1-based pages; pypdf APIs are 0-based. The scripts convert — don't double-convert.
- Rotation must be a multiple of 90; encrypted inputs must be decrypted before any other operation.

## Verification

- After create/merge/split: `pdf_read.py out.pdf --meta` — confirm `page_count`, and per-page `rotation` when you rotated.
- After extraction: check the JSON is non-empty and spot-check a known string or cell.
- After form fill: `pdf_read.py filled.pdf --fields` and compare values (exact match, including non-ASCII).
- After encrypt: `--meta` shows `"encrypted": true` and opening without a password fails; after decrypt, text extraction matches the original.
- For anything visual (watermarks, flattened forms), render and inspect with `vision_analyze`.
