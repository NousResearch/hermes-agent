# Migration flags — Hermes Agent docs

Items without a one-to-one Mintlify equivalent during migration.

## UserStoriesCollage (`user-stories.mdx`)

- **What:** Docusaurus React collage with filters, category colors, and masonry layout.
- **Pages:** `/user-stories`
- **What we did:** Rendered all 395 stories as static `<AccordionGroup>` entries grouped by category, preserving headline, author, source link, and quote.
- **Recommendation:** Optional custom React snippet later if interactive filters are required.

## AutomationBlueprintsCatalog (`reference/automation-blueprints-catalog.mdx`)

- **What:** Client-side fetch of `/docs/api/automation-blueprints-index.json` with copy buttons and “Send to App” deep links.
- **Pages:** `/reference/automation-blueprints-catalog`
- **What we did:** Inlined catalog content from live JSON API (titles, schedules, commands, tags).
- **Recommendation:** Wire live JSON fetch in a snippet if copy/deep-link UX must match source exactly.

## Skills Hub & Plugin Catalog pages

- **What:** Standalone React catalogs at `/skills` and `/plugins` (not docs-plugin pages).
- **Pages:** `/skills` (`skills.mdx` + `snippets/skills-hub.jsx`), `/plugins` (`plugins.mdx` + `snippets/plugins-catalog.jsx`), navbar tabs, inline links in features docs
- **What we did:** Per review, both catalogs are ported into Mintlify and restyled in the preview's Hermes design, linked in the sidebar as "Browse Skills" / "Browse Plugins" (as in the source sidebar), with the navbar Skills and Plugins tabs pointing to `/skills` and `/plugins`. Per review, the Skills tab now owns `/skills` (so it, not Docs, is highlighted there), and "Browse Skills" moved to a sidebar-top global anchor because Mintlify highlights the first tab that lists a page; the old external redirects for both paths were removed. Each fetches its live catalog at runtime from `nousresearch.github.io/hermes-agent/docs/api/` (`skills.json` ~65 MB, `plugins.json` ~5 MB), as the source pages do. Per-plugin pages (`/plugins/<name>`) and maintainer pages (`/plugins/by/<slug>`) are generated routes on the source site and were not ported; plugin cards link to them on `hermes-agent.nousresearch.com`.
- **Recommendation:** Port the per-plugin and maintainer pages if the preview should be self-contained.

## Docusaurus `Link` and inline styles on docs index

- **What:** `@theme/Link` CTAs and custom hero layout on Docusaurus docs homepage.
- **Pages:** `index.mdx`
- **What we did:** Replaced with `<CardGroup>` linking to real doc paths and external download/GitHub.
- **Recommendation:** Design agent can refine hero in phase 2.

## Locale switcher (zh-Hans)

- **What:** Full Chinese translation tree under `i18n/zh-Hans/`.
- **Pages:** Entire docs set
- **What we did:** Excluded from manifest (`excluded`); English only migrated.
- **Recommendation:** Separate locale pass if customer needs i18n on Mintlify.

## Algolia DocSearch

- **What:** Hosted search index and crawler config in Docusaurus theme.
- **Pages:** Global chrome
- **What we did:** Dropped; Mintlify provides built-in search.
- **Recommendation:** No action unless customer requests Algolia specifically.

## Mermaid diagrams

- **What:** `@docusaurus/theme-mermaid` fenced blocks.
- **Pages:** Various (where ` ```mermaid ` appears)
- **What we did:** Left as mermaid code fences (Mintlify supports mermaid).
- **Recommendation:** Verify render in preview; fix any that fail individually.

## Wiki-style `[[links]]` in skill docs

- **What:** Obsidian-style wikilinks in `research-llm-wiki` skill doc examples.
- **Pages:** `user-guide/skills/optional/research/research-llm-wiki.mdx`
- **What we did:** Left verbatim as documentation examples (warnings only).
- **Recommendation:** Escape or convert to markdown links if they break rendering.

## Finance skill LaTeX / KaTeX notation

- **What:** Spreadsheet-style `E\` and similar in finance skill docs.
- **Pages:** `finance-comps-analysis`, `finance-dcf-model`, `finance-lbo-model`
- **What we did:** Left source text; KaTeX warnings in validate.
- **Recommendation:** Backtick-wrap formula fragments that are not meant as math.