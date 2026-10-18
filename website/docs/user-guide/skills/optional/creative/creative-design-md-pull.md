---
title: "Design Md Pull — Fetch a site's DESIGN.md design-system doc by slug"
sidebar_label: "Design Md Pull"
description: "Fetch a site's DESIGN.md design-system doc by slug"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Design Md Pull

Fetch a site's DESIGN.md design-system doc by slug.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/design-md-pull` |
| Path | `optional-skills/creative/design-md-pull` |
| Version | `0.1.0` |
| Author | Bauke Molenaar, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `design`, `DESIGN.md`, `design-system`, `UI`, `VoltAgent`, `awesome-design-md`, `tokens` |
| Related skills | [`popular-web-designs`](../../bundled/creative/creative-popular-web-designs.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# DESIGN.md Pull Skill

Fetch a site's DESIGN.md design-system document (visual theme, color palette,
typography, component stylings, layout, depth & elevation, do's/don'ts,
responsive behavior, agent prompt guide) from the VoltAgent/awesome-design-md
catalog by site slug, and expose it as agent context for UI-generation work.

The skill is read-only: it pulls the raw DESIGN.md file from GitHub and prints
it. It never modifies the catalog, and it makes no claim of visual identity or
endorsement by any brand in the catalog.

Raw URL pattern (verified, no auth):
https://raw.githubusercontent.com/VoltAgent/awesome-design-md/main/design-md/&lt;site>/DESIGN.md

## When to Use

- "Show me the Cursor DESIGN.md", "what are Linear's design tokens",
  "fetch vercel's design-system doc as agent context".
- Grounding UI generation in real design systems: spacing scales, breakpoints,
  color roles, component styling, responsive behavior.
- Priming design context before generating a mockup or frontend for a target
  product.

Don't use for:
- Live brand assets (logos, images) — the catalog ships DESIGN.md text only.
- Authoritative brand guidance — DESIGN.md is a third-party extraction shipped
  "as is"; confirm against the site's own documentation.
- Fully offline flows without priming the cache first (see Pitfalls).

## Prerequisites

- Python 3 — standard library only (`urllib`), nothing to install.
- Network access to raw.githubusercontent.com. Without it, the skill falls
  back to an optional local cache or returns a clear error.

## How to Run

Invoke through `terminal` (curl) or the skill's helper script:

```bash
# curl — returns the DESIGN.md text as agent context
curl -fsS "https://raw.githubusercontent.com/VoltAgent/awesome-design-md/main/design-md/cursor/DESIGN.md"
```

or via the shipped script (adds caching, TTL, and clear errors):

```bash
python3 optional-skills/creative/design-md-pull/scripts/fetch_design_md.py cursor
```

In a Hermes session frame the call as `terminal(command="python3 scripts/fetch_design_md.py cursor", cwd="<repo checkout>", timeout=30)`; equivalent lookups also work through `web_extract` on the same raw URL.

## Quick Reference

| Slug | Site | Design context |
|---|---|---|
| cursor | Cursor | AI code editor UI: dark theme, gradient accents |
| vercel | Vercel | Black/white precision, Geist font system |
| linear.app | Linear | Ultra-minimal dark-mode SaaS, precise spacing |
| notion | Notion | Warm minimalism, serif headings, soft surfaces |
| x.ai | xAI | Stark monochrome, futuristic minimalism |
| warp | Warp | Dark IDE-like terminal interface |
| supabase | Supabase | Dark emerald, code-first developer dashboard |
| figma | Figma | Vibrant multi-color design-tool UI |
| lovable | Lovable | Playful gradients, AI full-stack builder UI |
| raycast | Raycast | Sleek dark chrome, macOS launcher UI |

## Procedure

1. Pick a slug from the Quick Reference (any catalog slug under
   `design-md/` works).
2. Fetch it: `terminal(command="curl -fsS <raw URL>", timeout=30)` or
   `python3 scripts/fetch_design_md.py <slug>`.
   Completion criterion: DESIGN.md text prints to stdout, exit code 0.
3. On failure, read the stderr message: HTTP 404 means the slug is absent
   from the catalog; network errors fall back to the optional cache when one
   exists.
4. Paste the returned DESIGN.md into agent context as design grounding.
   Completion criterion: every cited token or rule is quoted from the fetched
   file, not from memory.

## Pitfalls

- **Network / rate limits.** raw.githubusercontent.com is unauthenticated and
  rate-limited. `fetch_design_md.py` performs no retries; on failure it prints
  a clear stderr message and, when a cache copy exists (any age), returns it
  with a "stale content" warning instead of failing.
- **Main-branch mutability.** The catalog lives on `main` with no per-site
  pin — a DESIGN.md can change or be removed without notice. Treat content as
  current-at-fetch-time, not versioned; optionally pin a known-good commit SHA.
- **No brand endorsement.** Tokens are extracted public CSS values; VoltAgent
  disclaims ownership of every site's visual identity. Do not imply Cursor,
  Vercel, or any other brand endorses Hermes, and do not republish trade dress
  as Hermes' own.
- **No machine-local paths.** Reference the script repo-relative
  (`optional-skills/creative/design-md-pull/scripts/fetch_design_md.py`);
  never bake machine-local paths into the skill.

## Verification

- `python3 scripts/fetch_design_md.py cursor` prints Cursor's DESIGN.md and
  exits 0.
- `scripts/run_tests.sh tests/skills/test_design_md_pull_skill.py -q` passes
  with the network blocked — every fetch is mocked.
- Offline: run against an unreachable host with no cache → clear stderr error,
  non-zero exit. Prime the cache once online, then re-run offline → stale
  cache returns with a warning.

## Attribution and disclaimer

Source: VoltAgent/awesome-design-md (MIT license),
https://github.com/VoltAgent/awesome-design-md. DESIGN.md files are shipped
"as is" with no warranty; the design tokens are extracted public CSS values.
This skill makes no claim of visual identity or endorsement by Cursor, Vercel,
Linear, Notion, xAI, Warp, Supabase, Figma, Lovable, Raycast, or any other
brand in the catalog, and VoltAgent does not represent those brands.
