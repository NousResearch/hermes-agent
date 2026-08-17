---
title: "Docx — Create, read, edit, and template Word .docx files"
sidebar_label: "Docx"
description: "Create, read, edit, and template Word .docx files"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Docx

Create, read, edit, and template Word .docx files.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/docx` |
| Version | `1.0.0` |
| Author | Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `word`, `docx`, `documents`, `office`, `templates` |
| Related skills | [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Docx Skill

Create, read, edit, and template Microsoft Word `.docx` files with
python-docx via four small CLIs. It handles text, styles, lists, tables,
images, headers/footers, and `{{token}}` templating. It does not render
documents to PDF, edit legacy `.doc` binaries, or accept/reject tracked
changes (it only detects them — see Pitfalls).

## When to Use

- The user asks to generate a Word document (report, letter, contract).
- You need the text, outline, styles, or embedded images of a `.docx`.
- You must change an existing `.docx`: replace text, edit table cells,
  insert/delete paragraphs, apply styles.
- You have a `.docx` template with `{{placeholders}}` to fill from data.
- Not for: `.doc` (legacy), `.odt`, PDF conversion, or WYSIWYG layout work.

## Prerequisites

- Python 3.10+ with `python-docx` installed:
  `pip install python-docx` (import name is `docx`).
- For image blocks: the image files must exist locally (PNG/JPEG).

## How to Run

All helpers live in `scripts/` next to this file. Run them with the
`terminal` tool; each supports `--help` and prints JSON to stdout.

```bash
python scripts/docx_create.py spec.json out.docx
python scripts/docx_read.py out.docx --text
python scripts/docx_edit.py replace out.docx --find old --replace new
python scripts/docx_template.py tpl.docx values.json filled.docx
```

## Quick Reference

| Task | Command |
| --- | --- |
| Create from JSON spec | `docx_create.py spec.json out.docx` |
| Full text (body+tables+headers/footers) | `docx_read.py f.docx --text` |
| Heading outline + table shapes | `docx_read.py f.docx --structure` |
| Styles actually used | `docx_read.py f.docx --styles` |
| Extract embedded images | `docx_read.py f.docx --images outdir/` |
| Detect tracked changes/comments | `docx_read.py f.docx --revisions` |
| Find/replace (formatting kept) | `docx_edit.py replace f.docx --find A --replace B -o out.docx` |
| Set a table cell | `docx_edit.py set-cell f.docx --table 0 --row 1 --col 2 --text X` |
| Insert paragraph before index N | `docx_edit.py insert f.docx --index N --text X --style Normal` |
| Delete paragraph N | `docx_edit.py delete f.docx --index N` |
| Apply style to paragraph N | `docx_edit.py style f.docx --index N --style "Heading 1"` |
| Fill `{{tokens}}` | `docx_template.py tpl.docx values.json out.docx --strict` |

## Procedure

1. **Create.** Write a JSON spec with `write_file`, then run
   `scripts/docx_create.py`. The spec supports: `page` (size + margins in
   mm), `header`/`footer` strings, `styles` (custom paragraph styles with
   font, size, bold/italic, hex `color`), and `blocks` — `heading`
   (level 1-9), `paragraph` (either `text` or a `runs` list where each run
   may set `bold`/`italic`/`underline`), `bullet_list`, `numbered_list`,
   `table` (`header` row rendered bold, `rows`, optional built-in table
   `style` such as `Table Grid`), `image` (`path`, optional `width_mm`),
   and `page_break`. The full spec format is documented at the top of
   `scripts/docx_create.py` — read it with `read_file` when composing.
2. **Read.** Use `scripts/docx_read.py` with exactly one mode flag.
   `--text` returns body paragraphs, all table cell text, and
   header/footer text as JSON. `--structure` returns the heading outline
   plus paragraph/table/section counts. `--images DIR` copies every file
   under `word/media/` out of the package.
3. **Edit.** Use `scripts/docx_edit.py`. `replace` walks body, tables
   (nested included), headers and footers, and preserves run formatting;
   add `--body-only` to skip headers/footers. Pass `-o out.docx` to keep
   the original; omit it to edit in place. Paragraph indices for
   `insert`/`delete`/`style` refer to `--structure`/`--text` body order.
4. **Template.** Put `{{name}}`-style tokens in the document (letters,
   digits, `_`, `.`, `-`; optional inner spaces like `{{ name }}` are
   accepted). Run `scripts/docx_template.py` with a JSON object of
   values. Use `--strict` to fail when tokens remain unfilled; the JSON
   output lists `filled` counts and `unfilled_tokens` either way.
5. **Verify** (always): re-read the output with `--text` or
   `--structure` and confirm the expected content is present.

## Pitfalls

- **Tokens split across runs.** Word often fragments `{{name}}` into
  several runs. The replace helpers handle this by collapsing the runs;
  the replacement inherits the formatting of the run where the match
  starts. Mid-token formatting changes are therefore flattened.
- **Tracked changes.** `--revisions` only *detects* insertions,
  deletions, format changes, and comments. Text extraction returns the
  as-is body (insertions included, deletions omitted, i.e. roughly the
  accepted view), but this skill cannot accept/reject revisions or read
  comment text. Say so to the user rather than guessing.
- **Style names must exist.** Applying a style that isn't defined in the
  document raises `KeyError`. Built-ins like `Heading 1`, `List Bullet`,
  `List Number`, `Table Grid` exist in the default template; custom
  styles must be declared in the create spec first.
- **Numbered lists restart.** `List Number` relies on Word's default
  numbering; separate lists in one document may continue numbering
  instead of restarting. Acceptable for simple docs; warn users needing
  precise multi-list numbering.
- **Cell writes replace formatting.** `set-cell` uses `cell.text = ...`,
  which resets runs in that cell to plain formatting.
- **Encoding.** All JSON specs/values files are read as UTF-8 explicitly;
  never rely on locale defaults when writing your own glue code.
- **Don't unzip-and-sed the XML.** Edit through the scripts (or
  python-docx); raw text substitution in `document.xml` corrupts files
  easily. Use `patch`/`write_file` only for the JSON inputs, never on the
  `.docx` itself.

## Verification

- After create/edit/template, run `docx_read.py out.docx --text` and
  check the expected strings appear (and old strings are gone).
- For templates run with `--strict`, or check `unfilled_tokens == []`.
- Structure checks: `--structure` should show the expected heading
  outline and table shapes; `--styles` confirms custom styles applied.
- A valid `.docx` opens with `Document(path)` without exception — the
  read script exiting 0 is itself a sanity check.
