---
title: "Reladraw — Text diagrams with stated placement to SVG, upstream-kept"
sidebar_label: "Reladraw"
description: "Text diagrams with stated placement to SVG, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Reladraw

Text diagrams with stated placement to SVG, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/reladraw` |
| Path | `optional-skills/creative/reladraw` |
| Version | `0.13.0` |
| Author | reladraw contributors |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `diagram`, `architecture`, `dataflow`, `pipeline`, `deployment`, `svg`, `text-to-diagram`, `mermaid-alternative` |
| Related skills | [`architecture-diagram`](../../bundled/creative/creative-architecture-diagram.md), [`excalidraw`](../../optional/creative/creative-excalidraw.md), [`archify`](../../optional/creative/creative-archify.md), [`concept-diagrams`](../../optional/creative/creative-concept-diagrams.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# reladraw (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [reladraw/reladraw](https://github.com/reladraw/reladraw): the project ships
> its own agent skill directory (`.claude/skills/reladraw/`, a `SKILL.md` plus
> the full syntax reference). `hermes skills install official/creative/reladraw`
> pulls the current tree live from that repo (quarantined and scanned like any
> hub install) — this directory holds only the catalog metadata, so the
> language reference can never lag the CLI you actually run.

reladraw is a text language for node-and-line diagrams where the source states
the arrangement — `right of api`, `below app.ui`, `level with store`,
`between web and worker` — and the compiler works out only the distances.
Mermaid, Graphviz and D2 hand placement to a layout engine; draw.io and
Excalidraw make you pick coordinates. reladraw sits in between: relative
positions you can read back out of the source without looking at the picture,
which is exactly the property an agent that cannot see its own SVG needs.
Anything ambiguous is refused with a line-numbered error instead of guessed.

Source in, standalone SVG out (`reladraw diagram.reladraw -o diagram.svg`;
`-o -` for stdout). A `<reladraw-diagram>` web component renders the same
source inside a page.

## Prerequisites

- Node.js 18+ on `PATH`. `npx -y reladraw --help` works without installing
  (`-y` skips the interactive install prompt in non-TTY agent shells);
  `npm install -g reladraw` installs the command. The package has no runtime
  dependencies (~700 KB installed).
- The upstream skill is `SKILL.md` + `reference/syntax.md` (~120 KB total,
  no scripts); the fetch is pinned to one tree SHA, recorded in the bundle
  metadata.
- Nothing in the skill makes network calls; the CLI reads the `.reladraw`
  file and writes the SVG.

## When to prefer it

- The user has a specific picture in mind (this box under that one, the
  database off to the right) and auto-layout keeps rearranging it.
- A diagram will be edited later by re-reading its source: the arrangement is
  in the text, so a change to one statement moves one thing.
- Not for data charts (bar, line, pie) and not for hand-drawn or interactive
  output — use `excalidraw` for the sketch look, `archify` for validated
  pan/zoom HTML with Mermaid import, and the bundled `architecture-diagram`
  skill for zero-dependency dark-themed SVG-in-HTML.

Full documentation: https://github.com/reladraw/reladraw#readme
