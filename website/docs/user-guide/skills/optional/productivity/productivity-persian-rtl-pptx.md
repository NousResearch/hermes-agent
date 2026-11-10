---
title: "Persian Rtl Pptx — Build verified Persian/RTL .pptx decks with embedded fonts"
sidebar_label: "Persian Rtl Pptx"
description: "Build verified Persian/RTL .pptx decks with embedded fonts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Persian Rtl Pptx

Build verified Persian/RTL .pptx decks with embedded fonts.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/persian-rtl-pptx` |
| Path | `optional-skills/productivity/persian-rtl-pptx` |
| Version | `0.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `persian`, `rtl`, `pptx`, `presentation`, `powerpoint` |
| Related skills | [`pptx-author`](../../optional/finance/finance-pptx-author.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Persian RTL Presentation Skill

Produce polished, fact-checked Persian (Farsi) `.pptx` decks: right-to-left
paragraphs on every text shape, Persian digits, one Vazirmatn-family font
embedded inside the file, a consistent green/olive/gold design system, and a
programmatic QA loop that renders the deck and checks the Persian text.

## When to Use

- The user asks for a Persian/Farsi presentation about any topic.
- A non-Persian deck needs an RTL edition, or an existing Persian `.pptx`
  arrives with broken, reversed or disconnected text that must be fixed.
- Do NOT use for quick English decks — the plain `pptx-author` skill is faster.
  RTL history/heritage decks benefit most; research the facts first.

## Prerequisites

- `python-pptx` — if `import pptx` fails, install with `python -m pip install
  python-pptx` (never bare `pip` when a venv python is active; use the same
  interpreter, `python -m ensurepip --upgrade` first if the venv lacks pip).
- `matplotlib` + `numpy` for `scripts/make_graphics.py`, and `Pillow` for
  `verify_deck.py` overflow estimation (both optional; each degrades cleanly).
- LibreOffice (`soffice`) + `pdftoppm`/`pdftotext` (poppler) for render QA.
- A Persian font: download Vazirmatn TTFs (e.g. from
  `https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/fonts/ttf/`)
  into `~/.local/share/fonts` and run `fc-cache -f`. Needed both for local
  rendering and for embedding.

## How to Run

1. Research the topic with `web_search` (Persian queries) until every date,
   name, place and figure is confirmed by at least two independent sources.
   Unverifiable facts are dropped, never guessed.
2. Write a JSON deck spec (see `references/spec.md`); `templates/example_spec.json`
   reproduces the Operation Sarhani deck.
3. `python scripts/rtl_pptx.py spec.json out.pptx`
4. `python scripts/embed_fonts.py out.pptx --typeface Vazirmatn --regular R.ttf --bold B.ttf`
5. `python scripts/verify_deck.py out.pptx` — digits, margins, overlaps, overflow.
6. Render: `soffice --headless --convert-to pdf --outdir DIR out.pptx`, then
   `pdftotext -enc UTF-8` the PDF and `read_file` it to confirm connected
   Persian glyphs and correct facts. If `vision_analyze` is available, inspect
   each slide PNG too.

## Quick Reference

| Task | Command |
|---|---|
| New deck from spec | `scripts/rtl_pptx.py spec.json out.pptx` |
| Embed fonts in .pptx | `scripts/embed_fonts.py out.pptx --typeface X --regular r.ttf --bold b.ttf` |
| Structural QA | `scripts/verify_deck.py out.pptx` |
| Gold star-pattern PNG | `scripts/make_graphics.py --kind pattern pattern.png` |
| Spec schema | `references/spec.md` |
| Worked example spec | `templates/example_spec.json` |

## Procedure

1. **Verify facts.** Queries in Persian; ≥2 corroborating sources per fact;
   keep a source list — the deck ends with a sources slide citing URLs.
2. **Prepare assets.** Generate any pattern/map art with
   `scripts/make_graphics.py` or matplotlib. Raster images must be TEXT-FREE;
   Persian labels are overlaid as native pptx text boxes on top (see Pitfalls).
   For maps, emit `fx/fy` label fractions (0..1 of the image) alongside the PNG.
3. **Author the spec** as `write_file` with `encoding="utf-8"`. Slide types:
   `title`, `agenda`, `bullets`, `split` (image + bullets + map labels),
   `table`, `timeline`, `cards`, `sources`, `closing`. ≤6 bullets per slide,
   ≤~12 words per bullet, body ≥18pt.
4. **Build and embed fonts** as in How to Run. Deliver the pptx somewhere
   durable (not the 24h scratch dir).
5. **QA loop.** Run `verify_deck.py`, fix every reported issue, rebuild,
   re-run. Then render to PDF: confirm via extracted text that Persian is
   connected (search for a multi-letter word like «شرهانی» — if letters were
   reversed it would not match), that no Latin digits or `\ufffd` appear, and
   that key facts render. Re-check after any edit.

## Pitfalls

- **matplotlib cannot shape Persian.** Its glyphs come out disconnected or
  reversed. All Persian text lives in native pptx text boxes; images are
  graphics-only. PowerPoint and LibreOffice shape/join the script correctly
  from the stored logical-order text.
- **python-pptx has no RTL API.** The builder sets, per paragraph,
  `<a:pPr rtl="1" algn="r">` and per run `lang/szCs/bCs/cs/rtl` with the
  complex-script typeface. This combination is what makes both PowerPoint and
  Impress render properly — skipping `cs` silently falls back to a default
  Arabic font on other machines.
- **Font embedding is XML surgery.** `ppt/presentation.xml` gains
  `<p:embeddedFontLst>` right after `<p:notesSz>` (schema order matters),
  plus relationships and a `fntdata` content-type Default. LibreOffice ignores
  embedded fonts, so install the font locally for render QA anyway.
- **Preserve the venv interpreter.** `pip` may target a different python than
  the one running the builder — always `python -m pip`.
- **Bullet glyphs:** Vazirmatn covers `•` (U+2022) but not `●`/`◆`; check
  with `fc-match "Vazirmatn:charset=25CF"` before picking markers.
- **Persian digits** are the spec author's job for prose, but the builder can
  auto-convert 0-9 (default on; `"persian_digits": false` opts out — it would
  also mangle model numbers like F-14).
- **Map labels overlap.** `verify_deck.py` catches pairwise text-box
  overlaps; expect 1-2 fix iterations of nudging `fx/fy` coordinates.
- Labels under ~18pt (cartographic annotations) consciously deviate from the
  body minimum — keep everything else ≥18pt.
- `verify_deck.py` flags only geometry; approve screenshots with `vision_analyze`
  when available — formatting bugs it cannot see.

## Verification

A deck is done when: `verify_deck.py` prints no problems; the rendered PDF's
extracted text proves the Persian renders connected and the facts landed.
Search it for single Persian WORDS or short fragments — `شرهانی`, `۱۳۶۴`,
`حمزه` — never full phrases: `pdftotext` wraps RTL lines at bidi boundaries and
inserts bidi control characters, so a date like `۲۰ تیر ۱۳۶۴` will not match
literally even on a perfect render. Also confirm: no Latin digits, no
replacement characters, and slide count matches the spec. Report any facts
that could not be verified rather than asserting them.

Skill smoke test: building `templates/example_spec.json` must yield 15 slides
and `verify_deck.py` exit 0 — run it once after any change to the builder
scripts before trusting them on a new spec.
