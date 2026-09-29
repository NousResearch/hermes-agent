---
title: "System Atlas — Build explorable isometric architecture atlases as HTML"
sidebar_label: "System Atlas"
description: "Build explorable isometric architecture atlases as HTML"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# System Atlas

Build explorable isometric architecture atlases as HTML.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/system-atlas` |
| Path | `optional-skills/creative/system-atlas` |
| Version | `1.0.0` |
| Author | Harshyt Goel (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `architecture`, `diagrams`, `isometric`, `documentation` |
| Related skills | [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`excalidraw`](/docs/user-guide/skills/optional/creative/creative-excalidraw) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# System Atlas Skill

An atlas is one data file (`data.mjs`) that renders two views: an **interactive isometric map** (a single self-contained `atlas.html` — hover to read, click to pin, go inside for steps, moving data packets you can inspect, chapters that reveal the system a few structures at a time), and a **generated text twin** (`SYSTEM.md`) with the decisions table, every structure, the flows, and the open questions by ID. The data file is the only thing anyone edits; both views rebuild from it. It sits beside a hand-written glossary (`CONTEXT.md`) and ADRs.

**Does:** interactive architecture maps with progressive disclosure, question tracking across feedback rounds, a generated text twin, and a repeatable update loop.
**Doesn't:** static one-off diagrams (use the architecture-diagram or excalidraw skill), finished systems that only need a README, or a single diagram for a PR.

## When to Use

Use whenever someone wants to discuss, design, review, or explain an architecture visually — "make an atlas", "map the system", "make the architecture explorable", "visualize the codebase/agent/pipeline so we can talk about it", "a diagram I can click around", "walk me through how it fits together" — or when an architecture discussion is producing a pile of open questions that need tracking across feedback rounds. Also use it to update an existing atlas after decisions change. Best when the system is new enough that vocabulary, decisions, and questions are still moving and there will be more than one feedback round.

## Prerequisites

- Node.js (any recent version; the build uses only `node:fs`, `node:path`, `node:url` — no npm install needed).
- A static server for verification (`npx serve` or `python3 -m http.server`).

## How to Run

```bash
mkdir -p <atlas home>/atlas
cp <skill>/assets/{template.html,build.mjs} <atlas home>/atlas/
cp <skill>/assets/data.example.mjs <atlas home>/atlas/data.mjs   # then fill it in
node <atlas home>/atlas/build.mjs   # writes ../SYSTEM.md and ../atlas.html
```

Every field of the data file is documented in `assets/data.example.mjs`.

## Quick Reference

| File | Role | Edit it? |
|---|---|---|
| `atlas/data.mjs` | Single source of truth: structures, flows, chapters, decisions, questions, prose | Yes |
| `atlas/template.html` + `atlas/build.mjs` | Renderer + generator | Presentation only |
| `atlas.html` | Built atlas; republish at the same URL after every rebuild | No (generated) |
| `SYSTEM.md` | Built text twin | No (generated) |
| `CONTEXT.md` | Glossary, one line per noun | By hand |
| `adr/` | Hard-to-reverse decisions | By hand |
| `research/` | Deep-dive evidence | Append-only |

## Procedure

Follow the order — each step was earned by a correction the first time round.

1. **Read the inputs before drawing.** The vision doc, the repo's existing surfaces, and whatever prior art the user allows (ask — they may forbid a branch or a source). If you will build on a framework, read its docs first; hand long docs to a subagent via `delegate_task` with your specific design questions and have it return a primer with gotchas and a "what it does not give us" list. Drawing before this produces boxes that don't map to anything real.
2. **Discuss before drawing.** Propose the structure in chat, mapped to the runtime's real primitives, and ask only the questions you cannot derive from the repo. Take defaults for the rest and say which.
3. **First atlas — the whole system.** Copy `assets/` into the atlas home (rename `data.example.mjs` to `data.mjs`), fill the data, build with `node`, publish. **Where the atlas home is depends on the repo's docs policy** — ask before committing anything. Docs-friendly repos: `docs/<system>/atlas/` in-tree. Repos that commit only ADRs and `CONTEXT.md`: put the atlas, `SYSTEM.md`, and `research/` in a git-ignored scratch dir and attach `SYSTEM.md` + research to the spec issue when published. (Committing the whole set once produced a 3,900-line docs PR and four review rounds reconciling three restatements of one design.) Load the design-md or architecture-diagram skill via skill_view for HTML-artifact guidance if useful; read `references/design-language.md` for the visual rules either way.
4. **Progressive disclosure.** A whole system at once reads as noise. Ten-ish chapters; each adds at most three structures and runs one small flow that only touches revealed structures; the last chapter shows everything with a flow picker. Unrevealed structures stay in the index, dimmed, with their chapter number. Panels are summary-first: one sentence, then *Read more* and *Steps* folded.
5. **Shapes and labels.** Letters on boxes are not enough. Give each role a shape and put a readable name label on the canvas under every structure — see design-language.
6. **Text twin.** `CONTEXT.md` is a glossary and nothing else (the nouns, one line each); ADRs only for decisions that are hard to reverse, surprising without context, and the result of a real trade-off — these two are the in-tree pieces. `SYSTEM.md` is generated and `research/` holds evidence; both live with the atlas (scratch dir or `docs/`, per step 3). Don't open issues unless asked.
7. **Feedback by question ID.** Every question is `Q-<code><n>` with a state: open (a string), resolved `{q, r}` (answer + date), or routed `{q, to}` (handed to a named next step). Record the user's words. If they call something "not a question", drop it; if they say "I don't get this", explain with a concrete example *before* resolving. After each round: rebuild, republish, update memory.
8. **Deep dives feed back.** Research with subagents (`delegate_task`) against one shared brief (the interface we own, the requirements that separate candidates, a usage model for cost, a fixed deliverable shape). Write a synthesis with a normalized cost/fit grid. Fold resolutions into the data as `{q, r: '… (from the deep dive, date)'}`. If the user rejects a proposal, sweep *every* file and rewrite — a banner on top of a stale section is not enough.
9. **Keep it current.** One source, rebuild and republish after every change, never hand-edit generated files, and leave a `README.md` in the docs folder explaining the set (table in `references/process-and-lessons.md`).

## Publishing

`atlas.html` is one self-contained file — no build step, no external assets beyond a Google Fonts stylesheet. Serve the folder with any static server (`npx serve`, `python3 -m http.server`) and hand over the URL, or let the repo's pages host serve the committed file. One URL, republished after every data change, never a second copy. If you keep a stable published URL, put it in `META.artifactUrl` so `SYSTEM.md` links to it.

## Pitfalls

- Keep `<!doctype html>` first and `<meta charset="utf-8">` immediately after — otherwise quirks mode and mojibake arrows.
- The renderer rebuilds its whole scene on every draw: a stray `render()` in a hover handler detaches the element under the cursor and the browser stops synthesising clicks — the map looks perfect in a screenshot while nothing responds.
- Some in-app browsers render `file://` as a static snapshot; verify via a static server, not from disk.
- Never delete a question — resolve or mark it dropped, so IDs stay stable.
- After every decision, grep the outputs for stale words (`pending`, the old model name, the rejected design) — the person reads everything.
- Large HTML/JS via shell heredocs is brittle; use `write_file` and keep the data block JSON-serializable.

## Verification

- `node <atlas home>/atlas/build.mjs` exits 0 and writes both `SYSTEM.md` and `atlas.html`.
- Syntax-check the built script (`new Function(js)`), then open the served page in a real browser at ~1280×800; check a first chapter, a middle chapter, the last chapter, an inside view, and the light theme.
- Click a structure and confirm the panel says **pinned** and offers *Go inside*; click a packet dot and confirm the payload opens.
- Every structure has `one`, `what`, `how`, a `short` label, a role `kind`, and its questions; ghosts are marked; chapters exist with per-chapter flows; the last chapter is the whole system.
- `SYSTEM.md` carries the decisions table, the question index with IDs and states, and the "how this file is maintained" footer.
- Project memory records the atlas URL, docs paths, locked decisions with dates, what the user rejected and why, and the next step.
