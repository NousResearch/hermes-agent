---
title: "Ocr And Documents — Extract text from PDFs/scans (pymupdf, marker-pdf)"
sidebar_label: "Ocr And Documents"
description: "Extract text from PDFs/scans (pymupdf, marker-pdf)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ocr And Documents

Extract text from PDFs/scans (pymupdf, marker-pdf).

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/ocr-and-documents` |
| Version | `2.3.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `PDF`, `Documents`, `Research`, `Arxiv`, `Text-Extraction`, `OCR` |
| Related skills | [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# PDF & Document Extraction

For DOCX: use `python-docx` (parses actual document structure, far better than OCR).
For PPTX: see the `powerpoint` skill (uses `python-pptx` with full slide/notes support).
This skill covers **PDFs and scanned documents**.

## Step 1: Remote URL Available?

If the document has a URL, **always try `web_extract` first**:

```
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
web_extract(urls=["https://example.com/report.pdf"])
```

This handles PDF-to-markdown conversion via Firecrawl with no local dependencies.

Only use local extraction when: the file is local, web_extract fails, or you need batch processing.

## Step 2: Choose Local Extractor

| Feature | pymupdf (~25MB) | liteparse (optional) | marker-pdf (~3-5GB) |
|---------|-----------------|----------------------|---------------------|
| **Text-based PDF** | ✅ | ✅ | ✅ |
| **Scanned PDF (OCR)** | ❌ | ⚠️ optional/light OCR, slower | ✅ (90+ languages) |
| **Tables** | ✅ (basic) | ⚠️ row-ish plain text, not markdown tables | ✅ (high accuracy) |
| **Equations / LaTeX** | ❌ | ❌ | ✅ |
| **Code blocks** | ❌ | ❌ | ✅ |
| **Forms** | ❌ | ❌ | ✅ |
| **Headers/footers removal** | ❌ | ❌ | ✅ |
| **Reading order detection** | ❌ | ✅ useful fast check | ✅ |
| **Images extraction** | ✅ (embedded) | ❌ | ✅ (with context) |
| **Images → text (OCR)** | ❌ | ⚠️ optional/light OCR | ✅ |
| **EPUB** | ✅ | ❌ | ✅ |
| **Markdown output** | ✅ (via pymupdf4llm) | ❌ plain text despite output_format hint | ✅ (native, higher quality) |
| **Install size** | ~25MB | small optional package | ~3-5GB (PyTorch + models) |
| **Speed** | Instant | fastest for plain text when OCR is disabled | ~1-14s/page (CPU), ~0.2s/page (GPU) |

**Decision**: Use pymupdf/pymupdf4llm unless you need OCR, equations, forms, or complex layout analysis. Use optional `liteparse` only for local text PDFs where speed/plain text or reading-order checks matter more than markdown fidelity.

If the user needs marker capabilities but the system lacks ~5GB free disk:
> "This document needs OCR/advanced extraction (marker-pdf), which requires ~5GB for PyTorch and models. Your system has [X]GB free. Options: free up space, provide a URL so I can use web_extract, or I can try pymupdf which works for text-based PDFs but not scanned documents or equations."

---

## pymupdf (lightweight)

```bash
pip install pymupdf pymupdf4llm
```

**Via helper script**:
```bash
python scripts/extract_pymupdf.py document.pdf              # Plain text
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # Tables
python scripts/extract_pymupdf.py document.pdf --images out/ # Extract images
python scripts/extract_pymupdf.py document.pdf --metadata    # Title, author, pages
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # Specific pages
```

**Inline**:
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## liteparse (optional fast plain-text fallback)

Use `liteparse` only for **local text PDFs** when you need fast plain text or a quick reading-order check and can tolerate weak markdown/table fidelity. Do **not** replace `pymupdf4llm` with it for agent-ready markdown chunking: in testing, `liteparse` returned plain text even with `output_format="markdown"`, and tables needed cleanup.

```bash
uv pip install liteparse
# or, inside an existing non-PEP-668 environment:
pip install liteparse
```

**Via helper script**:
```bash
python scripts/extract_liteparse.py document.pdf
python scripts/extract_liteparse.py document.pdf --pages 1-3
python scripts/extract_liteparse.py document.pdf --max-pages 5
python scripts/extract_liteparse.py document.pdf --ocr       # slower; only for light OCR attempts
```

**Smoke test against a local text PDF**:
```bash
python scripts/extract_liteparse.py samples/controlled_agent_brief_table_layout.pdf --max-pages 1 \
  | grep -E "Agent Ingestion Brief|Decision Table"
```

Expected result: the command prints those headings quickly. If `liteparse` is missing, install it with `uv pip install liteparse`. If the output needs markdown headings, pipe tables, OCR quality, or complex layout semantics, switch back to `pymupdf4llm` or `marker-pdf`.

---

## marker-pdf (high-quality OCR)

```bash
# Check disk space first
python scripts/extract_marker.py --check

pip install marker-pdf
```

**Via helper script**:
```bash
python scripts/extract_marker.py document.pdf                # Markdown
python scripts/extract_marker.py document.pdf --json         # JSON with metadata
python scripts/extract_marker.py document.pdf --output_dir out/  # Save images
python scripts/extract_marker.py scanned.pdf                 # Scanned PDF (OCR)
python scripts/extract_marker.py document.pdf --use_llm      # LLM-boosted accuracy
```

**CLI** (installed with marker-pdf):
```bash
marker_single document.pdf --output_dir ./output
marker /path/to/folder --workers 4    # Batch
```

---

## Arxiv Papers

```
# Abstract only (fast)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# Full paper
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# Search
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## Split, Merge & Search

pymupdf handles these natively — use `execute_code` or inline Python:

```python
# Split: extract pages 1-5 to a new PDF
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# Merge multiple PDFs
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# Search for text across all pages
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"Page {i+1}: {len(results)} match(es)")
        print(page.get_text("text"))
```

No extra dependencies needed — pymupdf covers split, merge, search, and text extraction in one package.

---

## Notes

- `web_extract` is always first choice for URLs
- pymupdf/pymupdf4llm is the safe default — instant, no models, works everywhere, and gives better markdown for text PDFs
- liteparse is optional for speed-first local text-PDF plain text; keep OCR disabled unless explicitly testing it, because OCR can make it much slower
- marker-pdf is for OCR, scanned docs, equations, complex layouts — install only when needed
- Helper scripts accept `--help` for full usage
- marker-pdf downloads ~2.5GB of models to `~/.cache/huggingface/` on first use
- For Word docs: `pip install python-docx` (better than OCR — parses actual structure)
- For PowerPoint: see the `powerpoint` skill (uses python-pptx)
