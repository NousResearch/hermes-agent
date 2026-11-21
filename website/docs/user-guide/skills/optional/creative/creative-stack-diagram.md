---
title: "Stack Diagram — Stack diagrams with official logos, upstream-kept"
sidebar_label: "Stack Diagram"
description: "Stack diagrams with official logos, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stack Diagram

Stack diagrams with official logos, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/stack-diagram` |
| Path | `optional-skills/creative/stack-diagram` |
| Version | `1.0.0` |
| Author | caru-ini |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `diagram`, `architecture`, `tech-stack`, `logos`, `aws`, `gcp`, `azure`, `html`, `png` |
| Related skills | [`architecture-diagram`](../../bundled/creative/creative-architecture-diagram.md), [`archify`](../../optional/creative/creative-archify.md), [`excalidraw`](../../optional/creative/creative-excalidraw.md), [`concept-diagrams`](../../optional/creative/creative-concept-diagrams.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# stack-diagram (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [caru-ini/stack-diagram](https://github.com/caru-ini/stack-diagram). The
> project ships a self-contained skill directory (`skills/stack-diagram/`) with
> the HTML template, four stdlib-only Python scripts, two references, generic
> glyphs and OFL fonts. `hermes skills install official/creative/stack-diagram`
> pulls the current tree live from that repo (~22 files, ~170 KB). Like any hub
> install, it is quarantined and scanned first. This directory holds only the
> catalog metadata, so the installed copy can never go stale.

stack-diagram turns "Next.js on Vercel, Hono on Cloudflare Workers, Neon
Postgres" into a cloud-vendor-style architecture diagram. Real brand logos sit
on white tiles inside labeled group boxes, with orthogonal arrows that never
cross label text. Logos are fetched at run time from svgl, devicon and
gilbarbara/logos, plus AWS, Google Cloud and Azure's own official icon packages,
which it follows each vendor's drawing rules for. The output is one
self-contained HTML file; PNG (light + dark), a React/TSX component, or an
artifact are derived from it.

## Prerequisites

- Python 3.9+ (`fetch_icons.py`, `build.py`, `share.py` use only the standard
  library).
- Network access to the icon sources when fetching logos.
- PNG output and the skill's render-and-look step need Playwright with
  Chromium (`pip install playwright && python3 -m playwright install chromium`).
  If those are missing, the HTML is still the full deliverable.
- Upstream's optional Step 7 can post a diagram to a public gallery
  (stacks.caru.moe). It runs only after two explicit yeses from the user, on
  content they were shown. Skip it when the diagram names internal systems.

Pick it over the bundled `architecture-diagram` when the user wants real
product logos or AWS/GCP/Azure-style group boxes. Pick `archify` for validated
receipts, Mermaid input, or sequence/state-machine diagrams.

Full documentation: https://github.com/caru-ini/stack-diagram#readme
