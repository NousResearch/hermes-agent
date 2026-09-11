---
sidebar_position: 3
title: "Document Extraction"
description: "How read_file converts PDFs, Office documents, and notebooks to text — and what to do when a PDF is scanned images"
---

# Document Extraction

The `read_file` tool automatically converts common document formats to readable text, so the agent can inspect a PDF or spreadsheet the same way it reads source code.

## Supported formats

| Format | Extensions | Converter | Availability |
|--------|-----------|-----------|--------------|
| Jupyter notebooks | `.ipynb` | Built-in (stdlib) | Always |
| Word documents | `.docx` | Built-in (stdlib) | Always |
| Excel workbooks | `.xlsx` | Built-in (stdlib) | Always |
| PDF | `.pdf` | Optional `anydoc` converter | Auto-installed on first use* |
| Legacy Office | `.doc`, `.ppt`, `.xls`, `.pptx`, and variants | Optional `anydoc` converter | Auto-installed on first use* |
| OpenDocument | `.odt`, `.ods`, `.odp` | Optional `anydoc` converter | Auto-installed on first use* |
| Rich text / eBooks | `.rtf`, `.epub` | Optional `anydoc` converter | Auto-installed on first use* |

\* The optional converter is the `firecrawl-anydoc` package, installed lazily where installs are permitted (`security.allow_lazy_installs` in `config.yaml`). Without it, the three stdlib formats still work; other formats fall back to the binary-file guard.

Conversion output is Markdown, paginated through `read_file`'s normal `offset`/`limit` window. Documents over 50 MB are refused to keep tool turns bounded.

Extraction works with remote terminal backends (Docker, Modal, SSH): the file's bytes are transferred across the backend boundary and converted host-side, so a document inside a sandbox reads the same as a local one.

## Scanned PDFs: the coverage warning

PDF conversion reads the **text layer only**. Pages that are scanned images — common in legal documents, resale packages, signed contracts, faxes — contain no text layer and silently convert to nothing. The telltale signature is section headers with empty bodies.

When a meaningful share of pages yields no text (over 20% of the document, or 10+ pages absolute), `read_file` prepends a warning to the extraction. Each unreadable gap is labeled with the last text extracted before it — usually a section divider — so the agent can target only the gaps it actually needs instead of OCRing the whole document:

```
[EXTRACTION COVERAGE WARNING: 198 of 311 pages in this PDF yielded no
text. ... Unreadable gaps, each labeled with the last text extracted
before it:
  pages 42-77 (36 pages) — after "Antigua Maintenance Corp Bylaws" (p41)
  pages 92-213 (122 pages) — after "... Covenants, Codes and Regulations" (p91)
  page 224 (1 page) — after "... Insurance Declaration Pages" (p223)
Decide which gaps you actually need — do NOT OCR or render everything. ...]
```

The warning lists the exact page ranges and the recovery paths:

1. **A few pages — render + vision.** Convert the pages to images and read them with the vision tool:
   ```bash
   pdftoppm -jpeg -r 150 -f 92 -l 94 document.pdf /tmp/page
   ```
   Then inspect each image with `vision_analyze`. Zero extra dependencies (poppler is required for the detection itself).
2. **Many pages — OCR.** The `ocr-and-documents` skill covers bulk OCR with marker-pdf (90+ languages, handles equations and tables; ~3-5 GB install).

Detection uses poppler's `pdftotext` for per-page text counts. If poppler is not installed, extraction still works — the coverage check is silently skipped.

:::tip
The agent handles the warning on its own — it will offer to render or OCR the missing pages. If you're reading extractions yourself, treat "header with an empty body" as a scanned section, not a missing one.
:::
