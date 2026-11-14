---
title: "Learn From Materials — Books and PDFs into cited study pages, upstream-kept"
sidebar_label: "Learn From Materials"
description: "Books and PDFs into cited study pages, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Learn From Materials

Books and PDFs into cited study pages, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/learn-from-materials` |
| Path | `optional-skills/research/learn-from-materials` |
| Version | `1.1.0` |
| Author | dmoshehun-prog |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `learning`, `study`, `pdf`, `books`, `knowledge-base`, `quiz`, `methodology`, `html`, `zh`, `en` |
| Related skills | [`llm-wiki`](../../optional/research/research-llm-wiki.md), [`pdf`](../../bundled/productivity/productivity-pdf.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# learn-from-materials (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [dmoshehun-prog/learn-from-materials](https://github.com/dmoshehun-prog/learn-from-materials).
> The repository root is the skill directory (`SKILL.md` + `scripts/`,
> `references/`, `templates/`, `assets/`, `examples/`).
> `hermes skills install official/research/learn-from-materials` pulls the
> current tree live from that repo. Like any hub install, it is quarantined and
> scanned first. This directory holds only the catalog metadata.

learn-from-materials turns a book, PDF, slide deck, document or web page into
a source-grounded knowledge base. It produces an offline interactive learning
page (HTML) and matching Markdown, in English or Chinese. Every claim links
back to the passage it came from, and the page keeps material-based content,
inference and outside checks visibly apart. It covers quick overviews or full
systematic study (a chapter-by-chapter ledger with a second read-back pass),
relationship maps between frameworks, a question bank with one-at-a-time
self-tests, and a reusable method library that can be saved, compared and
applied to a real problem.

## Prerequisites

- Python 3.10+. Deterministic work (extraction, assembly, checks, rendering)
  is done by `scripts/pipeline.py` and siblings; the agent does the reading
  and writing. `python3 scripts/pipeline.py preflight --input <file>`
  reports which extractors are available. The pipeline never installs
  packages itself.
- PyMuPDF (`pip install pymupdf`) for PDFs. Node.js + Playwright are optional
  (browser layout checks only). Scanned PDFs need the agent's vision tool or
  an OCR pass first (the `pdf` skill covers OCR).
- The upstream SKILL.md body, validator errors and some extraction headers
  are in Chinese; the generated pages support English (`meta.language: en`)
  and follow the user's language. A quick-mode English page for a 15-page
  paper took ~7 minutes in testing, ~3.9 MB single-file HTML.
- Installs pull the whole repository (~300 files, ~35 MB; `examples/` and
  `assets/` are most of it) from GitHub, which takes a few minutes. The fetch
  is pinned to one tree SHA, recorded in the bundle metadata.

## When to prefer it

- Long or dense material the user wants to *learn*, not just summarize:
  textbooks, papers, manuals, course slides.
- Answers that must cite where in the material they come from.

For a one-off summary or a quick question about a document, read it directly
instead; for a persistent cross-document wiki, use `llm-wiki`.

Full documentation: https://github.com/dmoshehun-prog/learn-from-materials#readme
