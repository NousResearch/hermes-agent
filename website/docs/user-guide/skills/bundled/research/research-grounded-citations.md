---
title: "Grounded Citations — Ground answers and documents in cited, verifiable sources"
sidebar_label: "Grounded Citations"
description: "Ground answers and documents in cited, verifiable sources"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Grounded Citations

Ground answers and documents in cited, verifiable sources.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research\grounded-citations` |
| Version | `1.1.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Research`, `Citations`, `Grounding`, `Sources`, `Web`, `Reports` |
| Related skills | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), `ocr-and-documents` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Grounded Citations

Every claim taken from an outside source gets an inline numbered citation and a
`Sources:` list, Perplexity-style. A ledger script owns the `url → [n]` mapping
so the numbers and URLs come from retrieval, never from memory — the model only
ever emits small integers it was handed.

For high-stakes work the same ledger doubles as a fact-checking chain: verbatim
quotes are attached to each source (rejected unless they literally appear in
the fetched page text), claims from model knowledge are flagged `[unverified]`,
and `verify --evidence` fails any draft whose cited sources carry no evidence.

This skill covers answers in chat, written documents (markdown, PDF, docx,
slides), and research reports. It does not cover academic BibTeX pipelines —
for conference papers use the `arxiv` skill, which this skill
feeds (see `references/citation-formats.md`).

## When to Use

Use whenever an answer or artifact rests on information you fetched rather than
knew:

- Research, comparisons, news summaries, "what is the current state of X"
- Any deliverable you write to disk that quotes, paraphrases, or reports
  outside facts — reports, briefs, docs, decks, wiki pages
- Fact-finding where the user will want to check your work
- Multi-source synthesis where conflicting sources must be attributed

Skip inline citations when the retrieval is incidental to another task — a
quick syntax/version lookup mid-coding, casual conversation, creative writing.
Mention a URL only if the user would plausibly want the link.

## Prerequisites

None beyond the standard toolset. `scripts/sources.py` is stdlib-only Python 3.
Retrieval comes from whatever is configured: `web_search`, `web_extract`,
`browser_navigate`, or `terminal` (curl, CLIs).

Ledger location: `$HERMES_HOME/cache/citations/ledger.json` (profile-aware).
Override per task with `--ledger <path>` or `HERMES_CITATION_LEDGER`.

## How to Run

```bash
S=~/.hermes/skills/research/grounded-citations/scripts/sources.py

python "$S" reset                                  # start a clean ledger
python "$S" add https://example.com/a --title "A"  # prints: [1]
python "$S" add https://example.com/b --title "B"  # prints: [2]
python "$S" list                                   # ledger table
python "$S" render                                 # Sources: block
python "$S" verify draft.md                        # catch bad citations
```

`add` is idempotent and URL-normalized: the same page always returns the same
id within a ledger, so ids stay stable across many search/extract rounds.

## Quick Reference

| Action | Command |
|---|---|
| Fresh ledger for a new task | `sources.py reset` |
| Register a source, get its id | `sources.py add <url> [--title T]` |
| Register several at once | `sources.py add <url1> <url2> ...` |
| Register from JSON tool output | `sources.py ingest results.json` |
| Attach verbatim evidence to a source | `sources.py quote <id> --text "exact wording" --from page.txt` |
| Show ledger | `sources.py list [--json]` |
| Render the Sources block | `sources.py render [--style markdown\|plain\|footnotes\|bibtex\|evidence] [--only 1,3]` |
| Render only what a draft cites | `sources.py render --cited-in draft.md` |
| Rewrite a draft's Sources block in place | `sources.py render --replace-in draft.md` |
| Check a draft's citations | `sources.py verify draft.md [--strict] [--min-coverage 0.6] [--evidence]` |

## Procedure

① **Reset the ledger** at the start of a task that will produce a grounded
answer or document. Skip the reset when continuing work whose ids are already
in a draft — reusing the ledger keeps the numbering stable.

② **Register every source at retrieval time.** After each `web_search` /
`web_extract` / `browser_navigate` / fetch, pass the URLs to `sources.py add`
(or pipe the raw JSON through `sources.py ingest`). Do this *before* writing
prose. Registering later, from memory, is the failure mode this skill exists to
prevent.

③ **Write cite-while-drafting.** Place the bracketed id(s) immediately after
each sentence the source supports:

```
Ice floats because it is less dense than liquid water.[1][2]
```

- No space before the bracket; each id in its own brackets.
- Max 3 ids per sentence. Cite per sentence, not one dump at the end.
- Only ids the ledger returned. Never invent an id or a URL.
- Claims from your own knowledge get no citation.
- Conflicting sources: present both readings, each with its own id.
- Quote exact figures, dates, and names as the source states them; flag gaps
  explicitly ("no source found for X") instead of smoothing them over.

④ **Append the Sources block** with `sources.py render --cited-in <draft>` so
the id → URL mapping is generated mechanically from the ledger, not retyped.
For non-markdown targets pick the matching `--style` and follow
`references/citation-formats.md` for placement (footnotes in docx, endnotes in
PDF/LaTeX, a Sources slide in decks, per-page source lists in wiki output).

⑤ **Verify before delivering** — `sources.py verify <draft>` exits non-zero on
unknown ids, on a Sources block that disagrees with the ledger, or (with
`--min-coverage`) on prose that is too thinly cited. Fix and re-run.

⑥ **Chat answers** follow the same steps with the draft in your reply: register
sources, cite inline, end with the rendered `Sources:` list. For a short answer
you may render the block from `sources.py render --only <ids>` instead of
writing to a file.

## Fact-Checking Mode

For work where the reader must be able to check the chain — medical, legal,
financial, safety, disputed claims, or when the user asks for fact-checking —
upgrade from citations to evidence:

① **Attach a verbatim quote per source.** After extracting a page, save its
text to a file and attach the sentence(s) that carry each claim:

```bash
python "$S" quote 1 --text "Ice is about 9% less dense than liquid water." --from page1.txt
```

The quote is rejected unless it appears verbatim in the evidence text
(insensitive to whitespace, case, and markdown markup — inline links like
`_[ERAP1](https://…)_` in extracted text match the plain prose a reader sees),
so a paraphrase or misremembered figure cannot masquerade as evidence.
Copy-paste from the fetched text; never retype. Quote the sentence as the
reader sees it — the matcher sees through the extractor's markup for you, so
you don't have to reproduce link syntax or escaped asterisks in your quote.

② **Flag model-knowledge claims with `[unverified]`.** A load-bearing claim
you could not source gets an explicit marker instead of a citation:

```
The refactor likely predates the 2.0 release.[unverified]
```

`verify --min-coverage` counts `[unverified]` sentences as covered — the goal
is declared provenance for every claim, not a citation on every sentence.
If a key claim can be checked, check it; `[unverified]` is for what genuinely
cannot be, and a fact-check deliverable dominated by `[unverified]` markers
should say so in its summary.

③ **Cross-check disputed facts against a second independent source.** When two
sources disagree, cite both readings with their own ids and quotes, and say
which you weight and why. One source is reporting; two independent sources are
corroboration.

④ **Verify with the evidence gate and render the evidence block:**

```bash
python "$S" verify report.md --evidence --min-coverage 0.5
python "$S" render --style evidence --replace-in report.md
```

`--evidence` fails the draft if any cited source has no attached quote. The
`evidence` render style prints each source's quotes beneath its URL, so the
deliverable shows claim → source → exact supporting text with nothing taken on
faith. Use `--replace-in <draft>` to rewrite an existing Sources block in place
(idempotent — safe to re-run after attaching more quotes); `--cited-in` prints
to stdout instead. Both emit the heading `## Sources` (`--style plain` emits
`Sources:`).

**What `--min-coverage` counts.** Coverage is
`sentences with declared provenance / prose sentences`. A prose sentence is a
non-empty line fragment of 4+ words after the Sources block, headings (`#`),
table rows (`|`), and fenced code are dropped; blockquote markers are stripped.
Provenance is declared by either a `[n]` citation or an `[unverified]` marker,
so a sentence carrying both counts once. Run `verify` without a threshold first
and read the `info: stats:` line to see the counts before picking a number.

## Pitfalls

- **Registering after writing.** The ledger must be populated from tool output,
  not reconstructed from the draft — that reintroduces exactly the hallucinated
  -URL risk the numbering removes.
- **Renumbering mid-task.** Never hand-edit ids in a draft. Ids are ledger
  identities; if a draft cites `[4]`, `[4]` must stay that source. Run `reset`
  only between tasks.
- **Retyping URLs into the Sources block.** Always `render`. A hand-typed URL
  is an unverified claim.
- **Citing a search snippet as if you read the page.** A `web_search`
  description supports only what it literally says. Cite the extracted page
  when the claim needs the body — `web_extract` it first.
- **Over-citing.** Three ids on a sentence is the ceiling; a citation on every
  clause makes text unreadable and hides which source carries the load.
- **Citing the ledger in code/config artifacts.** Source comments belong in
  prose deliverables and doc headers, not inside generated code.
- **Parallel subagents.** Each subagent has its own working directory; point
  them all at one ledger with `--ledger` (or `HERMES_CITATION_LEDGER`) if their
  outputs get merged, otherwise their ids will collide.
- **Quoting from a snippet instead of the page.** Evidence quotes must come
  from the extracted page text, not a search-result description — `web_extract`
  first, save the text, then `quote --from` that file.
- **Paraphrasing into `quote --text`.** The verbatim check will reject it; the
  fix is to find the actual sentence, not to reword until something matches.
- **Using `[unverified]` as an escape hatch.** It marks the rare claim that
  genuinely cannot be sourced; if most sentences carry it, the task needed more
  retrieval, not more markers.
- **Hand-editing the Sources block.** Use `render --replace-in <draft>`; slicing
  the file yourself risks a stale or duplicated block that `verify` then flags.

## Verification

```bash
python "$S" verify report.md --strict --min-coverage 0.5
```

Green means: every `[n]` in the draft exists in the ledger, the Sources block
lists exactly the cited ids with the ledger's URLs, and the cited share of
source-bearing sentences meets the threshold. Read the warnings even when the
exit code is 0 — uncited registered sources usually mean a claim lost its
attribution during editing.
