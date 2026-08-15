---
title: "Powerpoint — Create, read, edit .pptx decks with python-pptx"
sidebar_label: "Powerpoint"
description: "Create, read, edit .pptx decks with python-pptx"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Powerpoint

Create, read, edit .pptx decks with python-pptx.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/powerpoint` |
| Version | `1.0.0` |
| Author | Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `pptx`, `powerpoint`, `presentations`, `slides`, `office`, `python-pptx` |
| Related skills | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Powerpoint Skill

Create, inspect, and edit PowerPoint (.pptx) presentations using the
python-pptx library. Four helper scripts cover deck creation from a JSON
spec, structured read-back, in-place edits, and template-driven brand
decks — all offline, no PowerPoint installation required.

## When to Use

- The user asks to build a slide deck, report presentation, or pitch deck.
- You need to extract text, notes, tables, chart data, or images from a
  .pptx someone shared.
- You need to update an existing deck: replace text, refresh chart data,
  swap a logo, remove or reorder slides.
- You must produce an on-brand deck from a company .pptx template.
- Do NOT use this for .ppt (legacy binary) files — convert them first with
  `soffice --convert-to pptx old.ppt` if LibreOffice is available.

## Prerequisites

- Python 3.10+ with `python-pptx` installed
  (`pip install python-pptx`). Pillow is optional (only if you need to
  probe image dimensions yourself).
- Optional: LibreOffice (`soffice`) for rendering slides to images for
  visual verification. Degrade gracefully if absent — all create/read/edit
  operations work without it.
- Check availability via `terminal`:
  `python3 -c "import pptx; print(pptx.__version__)"` and `which soffice`.

## How to Run

All scripts live in `scripts/`, take `--help`, print JSON to stdout, and
exit non-zero on failure. Run them with `terminal`:

```bash
python3 scripts/pptx_create.py deck.json out.pptx
python3 scripts/pptx_read.py deck.pptx --outline      # full JSON outline
python3 scripts/pptx_read.py deck.pptx --notes        # speaker notes
python3 scripts/pptx_read.py deck.pptx --images ./img # export pictures
python3 scripts/pptx_edit.py deck.pptx --replace-text "Old Corp" "New Corp"
python3 scripts/pptx_edit.py deck.pptx --chart-data update.json
python3 scripts/pptx_edit.py deck.pptx --remove-slide 3 --move-slide 2 0
python3 scripts/pptx_from_template.py brand.pptx out.pptx --values vals.json
```

Author JSON specs with `write_file`; inspect script output and generated
JSON with `read_file`.

## Quick Reference

| Task | Command |
|---|---|
| New deck from spec | `pptx_create.py spec.json out.pptx` |
| 16:9 vs 4:3 | `"slide_size": "16:9"` or `"4:3"` in the spec |
| Outline as JSON | `pptx_read.py deck.pptx --outline` |
| Export images | `pptx_read.py deck.pptx --images DIR` |
| Replace text | `pptx_edit.py deck.pptx --replace-text OLD NEW` |
| Update chart | `pptx_edit.py deck.pptx --chart-data spec.json` |
| Swap picture | `pptx_edit.py deck.pptx --swap-image N NAME new.png` |
| Remove slide | `pptx_edit.py deck.pptx --remove-slide N` |
| Reorder slide | `pptx_edit.py deck.pptx --move-slide FROM TO` |
| Fill template | `pptx_from_template.py tpl.pptx out.pptx --values v.json` |

## Procedure

### 1. Create a deck

Write a JSON spec (see `pptx_create.py --help` for the full format), then
run `pptx_create.py`. Per slide you can set: `layout` (title,
title_content, section, two_content, title_only, blank), `title`,
`subtitle`, `bullets` (strings, or dicts with `level` 0-4, `size` pt,
`bold`, `italic`, `font`, `color` hex), `images` (path + left/top/width/
height in inches), `tables` (`rows` as list-of-lists), `shapes`
(rectangle, rounded_rectangle, oval, diamond, right_arrow, chevron, with
`fill` hex + optional `text`), `charts` (bar, bar_h, line, pie with
`categories` + `series`), and `notes` (speaker notes).

### 2. Read a deck

`pptx_read.py deck.pptx --outline` returns slide size, layout inventory,
and per slide: layout name, all shape texts, table cells, image inventory
(filename/ext/bytes), chart categories/series/values, and speaker notes.
Use `--images DIR` to dump embedded pictures to files, then
`vision_analyze` on any exported image if you need to see its content.

### 3. Edit a deck

`pptx_edit.py` combines operations in one pass; use `--output` to keep the
original. Text replacement scans slide shapes, table cells, and notes.
Chart update uses `chart.replace_data()` with a JSON spec naming the
slide/chart index and new categories/series. Image swap retargets the
picture's relationship id so position and size are preserved. Slide
removal drops the relationship and the `<p:sldId>` entry; reorder moves
the `<p:sldId>` element within `<p:sldIdLst>` (python-pptx has no public
API for either — the script does the XML-level work).

### 4. Build from a template

`pptx_from_template.py` opens a brand .pptx, replaces every
`{{token}}` from a values JSON across slides/tables/notes, and can append
new slides that use the template's own layouts (by layout name or index)
so they inherit the master's fonts and colors. Tip: to start from a
template with zero slides, delete existing ones afterward with
`pptx_edit.py --remove-slide`.

### 5. Visual verification (optional)

If `soffice` exists, render slides to PNG and inspect with
`vision_analyze`:

```bash
soffice --headless --convert-to png --outdir ./render deck.pptx  # slide 1
soffice --headless --convert-to pdf --outdir ./render deck.pptx  # all slides
```

PNG export renders only the first slide; convert to PDF for all slides
(then `pdftoppm -png render/deck.pdf render/slide` if poppler is
available). When `soffice` is absent, rely on the JSON outline from
`pptx_read.py` — it verifies content and structure, just not visuals.

## Pitfalls

- **Run splitting**: PowerPoint fragments paragraph text into multiple
  runs at spell-check and formatting boundaries. `--replace-text`
  preserves formatting exactly when a match lies within one run; when the
  match spans runs, the paragraph is rewritten with only the first run's
  formatting. Verify important slides after replacement.
- **Reordering is XML-level**: python-pptx has no supported reorder API.
  `--move-slide` manipulates `<p:sldIdLst>` directly; it is safe for
  ordinary decks but re-read the deck afterward to confirm.
- **Copying slides between decks is unsupported** — layouts, images, and
  relationships would need deep cloning. Rebuild the slide in the target
  deck instead.
- Chart edits replace the whole data set; you cannot patch a single cell.
  Adding/removing series works, but changing chart *type* does not.
- The default python-pptx template is 4:3; the create script sets 16:9
  unless the spec says otherwise. Custom templates keep their own size.
- Layout indexes vary by template. For brand templates, list layout names
  first: `pptx_read.py template.pptx --outline` (`layouts_available`).
- `slide.shapes.title` is None on blank layouts — the create script
  handles this, but remember it when writing ad-hoc python-pptx code.
- Always pass `encoding="utf-8"` when writing spec files; tokens like
  `{{city}}` may be filled with non-ASCII values.

## Verification

1. After any create/edit, run `pptx_read.py OUT.pptx --outline` and check
   slide count, texts, tables, notes, and chart values match intent.
2. `--images DIR` then file-size check confirms pictures embedded.
3. For high-stakes decks, render via `soffice` (see Procedure step 5) and
   review each slide image with `vision_analyze`.
4. The bundled test suite is the full contract:
   `python3 -m pytest tests/ -q` (requires python-pptx + pytest).
