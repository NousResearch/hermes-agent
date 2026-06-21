---
title: "Productivity Integrations"
sidebar_label: "Productivity Integrations"
description: "Productivity integrations umbrella for Google Workspace, Notion, Airtable, documents/PDFs, maps, presentations, meeting pipelines, and placement outreach sys..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Productivity Integrations

Productivity integrations umbrella for Google Workspace, Notion, Airtable, documents/PDFs, maps, presentations, meeting pipelines, and placement outreach systems.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/productivity-integrations` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Productivity Integrations

Use this skill for working with productivity platforms, documents, office artifacts, maps, meetings, CRMs/tables, and structured outreach systems.

## Universal rules

1. Identify the platform, account/workspace, target object, permissions, and side-effect scope.
2. Read/search before writing to avoid duplicates and wrong targets.
3. Use platform CLIs/APIs where available; preserve IDs/URLs for verification.
4. For writes, report what changed and include a readback link/ID/status.
5. Do not expose tokens, OAuth secrets, private emails, or contact data unnecessarily.

## Workspace apps

- **Google Workspace**: Gmail, Calendar, Drive, Docs, and Sheets via `gws` CLI or Python bridge. Start with setup/auth status, then use the narrow service API.
- **Notion**: pages/databases/blocks through Notion API or `ntn`; inspect schema before creating/updating database rows.
- **Airtable**: REST CRUD, filtering, upserts, and schema reads; use correct field shapes and pagination.

## Documents and presentations

- **PDF/OCR**: choose lightweight extraction for digital PDFs, OCR/marker for scans, and verify extracted text quality.
- **nano-pdf**: use for targeted PDF text edits; keep original and verify page render/text after editing.
- **PowerPoint**: read existing slide structure before editing; validate generated decks and media paths.

## Maps and locations

- Use geocoding, reverse lookup, POI search, routing, and timezone endpoints with clear coordinates and units.
- Report uncertainty and source when geocoding ambiguous place names.

## Meeting and outreach workflows

- **Teams meeting pipeline**: inspect pipeline status, replay jobs, and manage Graph subscriptions through Hermes CLI commands.
- **Academic/practicum placement**: intake constraints, search approved portals first, research outside leads second, rank leads, draft outreach, and track status in a structured table.
## Notes and knowledge bases

- **Obsidian**: read, search, create, append, and patch markdown notes in the configured vault. Use it for durable user-facing notes and knowledge bases, not ephemeral task progress. Search before creating duplicates and preserve wiki links/frontmatter conventions.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
