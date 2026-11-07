---
title: "Hallmark — Build new web pages that look designed, upstream-kept"
sidebar_label: "Hallmark"
description: "Build new web pages that look designed, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hallmark

Build new web pages that look designed, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/hallmark` |
| Path | `optional-skills/creative/hallmark` |
| Version | `2.0.0` |
| Author | Hallmark contributors (Nutlope) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `design`, `frontend`, `web-design`, `landing-page`, `themes`, `anti-slop`, `lint` |
| Related skills | [`impeccable`](../../optional/creative/creative-impeccable.md), [`claude-design`](../../bundled/creative/creative-claude-design.md), [`popular-web-designs`](../../bundled/creative/creative-popular-web-designs.md), [`design-md`](../../optional/creative/creative-design-md.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hallmark (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Nutlope/hallmark](https://github.com/Nutlope/hallmark) (MIT).
> `hermes skills install hallmark` pulls the current `skills/hallmark` bundle
> live from that repo (quarantined and scanned like any hub install), so the
> copy you run is never stale. This directory holds only catalog metadata.

Hallmark is a design skill for building NEW websites and pages that look made,
not generated. It sets a hard, non-waivable "floor" against the defaults models
fall into (token-only colours and fonts, no `transition: all`, no eyebrows or
tracked-caps labels, no purple-to-blue gradients, no invented metrics, hero
visible at 1280x800, about 80 words per screen), deals one of 50 material themes
so consecutive builds do not converge, and ships a deterministic linter,
`sloplint`, that checks the output (optionally in a real browser).

It is NOT for restyling an existing design unless the user names Hallmark
(`/hallmark redesign <path>`, `/hallmark audit <path>`). For critique and polish
of an existing UI, use `impeccable`; for a fixed brand system, use `design-md`.

## After install

- Start with `/hallmark <what to build>`; verbs: `redesign <path>`, `audit <path>`.
- Requires Node for the linter: `node <skill-dir>/scripts/sloplint.mjs <output>`.
  `--render` also needs a local Chrome/Chromium plus `puppeteer-core`
  (`npm i -g puppeteer-core`, or point `HALLMARK_PUPPETEER` at one); without
  it the render pass reports "not run" instead of failing.
- Optional: `TOGETHER_API_KEY` enables the upstream comp/image step
  (`scripts/imagegen.mjs`, calls api.together.xyz). Everything else is local.
  In Hermes, prefer `image_generate` for imagery when that key is not set.
- Do NOT run `scripts/install.mjs` or `scripts/install-hook.mjs`: they copy the
  skill into other agents' skill folders and write another harness's hook
  settings. `hermes skills install` already placed it; Hermes does not use them.
- The upstream body mentions an optional "Inspo" MCP server (signal 8); skip
  that signal unless such a server is configured in Hermes.

Full documentation: https://github.com/Nutlope/hallmark
