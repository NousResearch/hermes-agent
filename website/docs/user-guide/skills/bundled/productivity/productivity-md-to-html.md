---
title: "Md To Html — Render .md to styled HTML with a script, never LLM-written"
sidebar_label: "Md To Html"
description: "Render .md to styled HTML with a script, never LLM-written"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Md To Html

Render .md to styled HTML with a script, never LLM-written.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/md-to-html` |
| Version | `1.0.0` |
| Author | Geoffrey Anderson (geoffreya), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `markdown`, `html`, `rendering`, `css`, `documentation`, `cost-savings` |
| Related skills | [`pdf`](../../bundled/productivity/productivity-pdf.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# md-to-html Skill

Render a Markdown file to a styled, self-contained HTML page with a deterministic script — tables, fenced code with syntax highlighting, TOC-friendly headings, and colored status indicators. The agent runs the script; it never composes the HTML markup itself. Markdown-to-HTML is mechanical: an LLM adds cost and variance and no quality, and using the foreground discussion model for it is the most expensive possible way to do a free conversion.

## When to Use

- The user asks to view, render, share, or publish a `.md` file as HTML.
- A `.md` was just edited and its HTML companion should be regenerated.
- Any impulse to write HTML markup by hand to mirror markdown content — run the script instead.

## Prerequisites

- The Python `markdown` package (bundled with Hermes installs; also `pygments` for code highlighting).
- No network access needed. Works offline.

## How to Run

From the terminal, with the skill directory on disk:

```bash
python3 <skill-dir>/scripts/md2html.py report.md
```

Batch mode (each input gets a sibling `.html`):

```bash
python3 <skill-dir>/scripts/md2html.py a.md b.md c.md
```

The default stylesheet lives at `templates/style.css` inside the skill and is **inlined** into every output page, so each HTML file is fully self-contained — open it from disk, mail it, or move it anywhere. To change styling, edit that one CSS file and rerun; never hand-patch generated HTML.

## Quick Reference

| Goal | Command |
|------|---------|
| Convert one file | `python3 scripts/md2html.py notes.md` |
| Convert several | `python3 scripts/md2html.py a.md b.md` |
| Print HTML to stdout | `... md2html.py notes.md --stdout` |
| Custom output path | `... md2html.py notes.md --out out/report.html` |
| Alternate stylesheet | `... md2html.py notes.md --css dark.css` |

## Procedure

1. Identify the `.md` file(s) to render (confirm the user's target if ambiguous).
2. Run the script with `terminal`, per How to Run above.
3. Check the printed `source -> destination` line, then verify the output with `read_file` (a quick structural check: `<table>`, `<h1>`, no markdown syntax leaking through).
4. Report the absolute path of the generated HTML to the user.
5. If the user wants different styling, edit `templates/style.css` with `patch` and regenerate — one stylesheet, not per-file fixes.

## Pitfalls

- **Never let the model write the HTML.** If the script errors, fix the script; do not fall back to composing markup in the response.
- **Do not overwrite hand-made HTML.** If the sibling `.html` already exists and was authored by a human (a designed page, not generated), convert to a fresh output name with `--out`.
- **`--out` takes exactly one input file.** For batches, let the script pick sibling paths.
- **Front matter is not parsed.** YAML front matter in the input renders as a code block; strip it first if the page should not show it.
- **Title comes from the first `# heading`**, falling back to the filename. Add an H1 if the page needs a proper title.

## Verification

- The script prints one `md2html: <src> -> <dst>` line per input; confirm the destination exists with `read_file` or a directory listing.
- Sanity-check the output: it should start with `<!DOCTYPE html>`, contain the inlined `<style>` block, and have no literal `**bold**` or `| table |` markup left unrendered.
- The repo test `tests/skills/test_md_to_html_skill.py` covers frontmatter, rendering, and the CLI end to end.
