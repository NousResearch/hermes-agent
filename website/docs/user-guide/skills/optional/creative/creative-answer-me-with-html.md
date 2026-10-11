---
title: "Answer Me With Html — One-page HTML explainer answers, upstream-kept"
sidebar_label: "Answer Me With Html"
description: "One-page HTML explainer answers, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Answer Me With Html

One-page HTML explainer answers, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/answer-me-with-html` |
| Path | `optional-skills/creative/answer-me-with-html` |
| Version | `0.4.11` |
| Author | QingYunA |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `html`, `explainer`, `diagram`, `visual-answer`, `flow`, `comparison`, `markdown`, `single-file`, `teaching` |
| Related skills | [`archify`](../../optional/creative/creative-archify.md), [`concept-diagrams`](../../optional/creative/creative-concept-diagrams.md), [`architecture-diagram`](../../bundled/creative/creative-architecture-diagram.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Answer me with HTML (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [QingYunA/answer-me-with-html](https://github.com/QingYunA/answer-me-with-html):
> the project ships a self-contained skill directory
> (`skills/answer-me-with-html/`, a `SKILL.md` plus one bundled zero-dependency
> Node CLI, `scripts/am.mjs`). `hermes skills install
> official/creative/answer-me-with-html` pulls the current tree live from that
> repo (quarantined and scanned like any hub install) — this directory holds
> only the catalog metadata, so the component set can never go stale.

Instead of a wall of prose, the agent writes a short extended-Markdown
**draft** (3–8 panels: flow, sequence, comparison table, tree, timeline,
callout, key-value block) and the bundled `am` CLI renders it into one
self-contained HTML page — layout, colours, dark mode, pan/zoom and diagram
coordinates all handled by the CLI, never hand-written HTML. The CLI also
lints the draft with a controlled-English/Chinese writing check (STE), can
`patch` one panel of an existing page in place, and can turn a draft into a
3Blue1Brown-style narrated explainer video. The draft language follows the
user's question (English, Chinese, Japanese and more).

## Prerequisites

- Node.js 20+ on `PATH`. `scripts/am.mjs` is a single pre-built file (~350 KB)
  with no `npm install`; a render takes well under a second.
- The upstream `SKILL.md` invokes the CLI through another harness's
  skill-directory variable (`node "${…_SKILL_DIR}/scripts/am.mjs"`); in
  Hermes substitute the absolute path of the installed skill directory (the
  SKILL.md says so for other agents). Pages land in
  `~/.answer-me-with-html/pages/` by default (`AM_HOME` overrides).
- On its first run, then weekly, the CLI spawns a detached child that fetches
  `package.json` from the upstream GitHub repo to print a new-version hint; it
  never downloads or installs anything. Disable with `am config set
  update_check false`, or `AM_NO_UPDATE_CHECK=1` / `CI=1` in the parent
  environment. The hint names other harnesses' update commands; in Hermes the
  equivalent is `hermes skills update answer-me-with-html`.
- Upstream's `/answer-me-with-html config …` slash form and its `$ARGUMENTS`
  placeholder are another harness's dispatch; in Hermes just run `am config
  set <key> <value>` when the user asks for a setting change.
- `open` defaults to on (a render launches the browser). Pass `--no-open` in
  gateway, cron and other headless runs and hand the user the `file://` link.
- Optional for `am video`: narration via ElevenLabs when `ELEVENLABS_API_KEY`
  is already exported, a local OpenAI-compatible speech server (`AM_TTS_URL`),
  or system TTS; `--voice off` for a silent video.
- Installs pull ~2 files (~370 KB) from GitHub; the fetch is pinned to one
  tree SHA, recorded in the bundle metadata.

## When to prefer it

- A how-it-works, comparison, diagnosis or multi-step answer that a reader
  would rather see than scroll through; "draw it", "explain visually",
  "画个图", "没看懂".
- `archify` is the better fit for validated architecture/sequence diagrams
  with receipts; `concept-diagrams` and the bundled `architecture-diagram`
  for a single SVG visual. Answer-me-with-HTML is for the whole answer as one
  readable page.

Full documentation: https://github.com/QingYunA/answer-me-with-html#readme
