---
title: "Cyber Vc Analyst"
sidebar_label: "Cyber Vc Analyst"
description: "Use when analyzing an early-stage cybersecurity startup or a cybersecurity market theme for venture investment"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cyber Vc Analyst

Use when analyzing an early-stage cybersecurity startup or a cybersecurity market theme for venture investment. Produces a structured company IC memo or thematic market memo using vault notes, Return on Security MCP market intelligence when available, and clearly separated facts, inferences, and assumptions.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/cyber-vc-analyst` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cybersecurity`, `venture`, `investing`, `startup`, `obsidian`, `mcp`, `research` |
| Related skills | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`llm-wiki`](/docs/user-guide/skills/bundled/research/research-llm-wiki), `native-mcp` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cybersecurity Venture Analyst

Produce a structured investment-committee memo for either:

- a pre-seed or seed-stage cybersecurity startup
- a cybersecurity market theme, category, or architectural shift

The output should be grounded in evidence and reusable inside a long-term
knowledge base.

This skill combines three evidence streams:

1. prior private context from the user's Obsidian vault
2. market intelligence from Return on Security MCP when available
3. public information from Hermes web tools when needed and available

Never blur those sources together. Preserve a clear line between verified fact,
reasonable inference, and assumption.

## When to Use

Use this skill when the user asks for:

- an investment view on a cybersecurity startup
- a seed or pre-seed cybersecurity company memo
- a market thesis on a cybersecurity theme, category, or architecture shift
- a thematic map of a cyber segment and the likely winners or losers
- a venture assessment, IC memo, or market map entry for a cyber company
- comparison-ready notes on founders, product, category, or defensibility
- a write-back into the user's vault so future analyses compound

Do not use this skill for:

- public-equity valuation or late-stage financial modeling
- a generic TAM deck with no company-specific assessment
- technical security due diligence on a product implementation
- penetration testing, threat hunting, or incident response

## Inputs

Expected user input for company mode:

- company name
- optional company URL
- optional founder names
- optional product description or category hint
- optional specific notes, folders, or prior memos to include

Expected user input for theme mode:

- theme or category name
- optional geography
- optional time horizon
- optional company set to include or exclude
- optional specific vault notes, folders, or prior memos to include

If key identifiers are missing, infer cautiously from the conversation and state
the assumption in the memo.

## Mode Selection

Choose the mode before gathering evidence.

Use **company mode** when the prompt centers on one company, founder team, or
startup investment decision.

Use **theme mode** when the prompt centers on:

- a cybersecurity category
- an investment theme
- a buyer workflow shift
- an architectural transition
- a market map or landscape
- a question about which cyber areas are attractive or crowded

If the prompt mentions both a company and a broader theme, default to company
mode and use the theme only as context unless the user explicitly asks for a
thematic memo.

For concrete prompt patterns, read `references/example-invocations.md`.

## Vault Rules

Resolve the Obsidian vault path first using the `obsidian` skill conventions:

- use `OBSIDIAN_VAULT_PATH` when set
- otherwise read `~/Library/Application Support/obsidian/obsidian.json` and use the
  currently open vault path when available
- only fall back to `~/Documents/Obsidian Vault` if neither source is available

Use Hermes file tools once the vault path is known:

- use `search_files` for filename and content search
- use `read_file` for note inspection
- use `write_file` for final memo creation or full-note replacement
- use `patch` only when a targeted anchored update is safer than rewriting

Treat vault notes as private first-party context. They are not public evidence.
Use them to recover:

- who the user has met
- what prior conversations imply about the company, founders, and market
- what the user's existing mental models, market maps, and competitor lists contain

Search order:

1. configured `cyber_vc_analyst.meeting_search_roots`
2. configured `cyber_vc_analyst.company_search_roots`
3. broader vault search only if the configured roots are empty or yield little

Prefer targeted searches for:

- company name and aliases
- founder names
- product names
- category phrases
- competitor names already mentioned in the vault
- theme names and adjacent category terms when in theme mode

For content searches, prefer `search_files` with `target: "content"` and
`file_glob: "*.md"`. For note listing, prefer `target: "files"`.

Read only the most relevant notes. Do not scan the whole vault unless the
configured roots are empty and narrower search failed.

## MCP Rules

Before using broad public search, inspect the available tool list for either:

- `mcp_returnonsecurity_*`
- `mcp_signal_mcp_*`

If Return on Security MCP tools are available:

- use them as the preferred source for market structure, category context,
  competitor discovery, trend validation, likely acquirers, and adjacent themes
- prefer the smallest tool or tool sequence that answers the question
- cite the MCP-derived point as market intelligence, not as a firsthand company fact

If Return on Security MCP tools are unavailable:

- continue with vault notes plus public sources if Hermes web tools exist
- explicitly note in the memo that ROS MCP market intelligence was unavailable
- reduce confidence where the missing market view materially weakens the conclusion

If both ROS MCP and public web tools are unavailable, still produce the memo from
vault context and user-provided information, but call out the evidence gap.

For setup instructions or alias expectations, read
`references/mcp-setup.md`.

## Workflow

1. Identify the subject.
   In company mode, confirm the legal or commonly used company name, website,
   founders, and claimed category if known. In theme mode, confirm the theme,
   scope, buyer context, and time horizon.

2. Recover prior vault knowledge.
   Search configured meeting and company roots first. Pull only notes that
   materially change the assessment. In theme mode, bias toward category notes,
   Matter/Readwise intelligence, and any existing company notes relevant to the
   theme.

3. Gather market context.
   Use `mcp_returnonsecurity_*` or `mcp_signal_mcp_*` tools when available. If
   they are absent and public web tools are available, use public sources to
   fill category, competitive, and market-evolution gaps.

4. Normalize the subject.
   In company mode, classify the company using the rubric in
   `references/taxonomy-rubric.md`. In theme mode, classify the theme, the
   customer problem, the market maturity, and the adjacent categories.

5. Draft the memo.
   Follow the correct mode-specific output structure in
   `references/output-template.md`.

6. Distinguish evidence quality.
   Every major conclusion should separate:
   - facts
   - reasonable inferences
   - assumptions or unknowns

7. Save the memo back to the vault when appropriate.
   In company mode, write the finished Markdown note under the configured
   `cyber_vc_analyst.output_root` using the company template's frontmatter and
   filename rules.
   In theme mode, save only when the user asks to save or maintain the theme in
   the vault; write it under `cyber_vc_analyst.theme_output_root` using the
   thematic template and filename rules.

## Output Rules

The deliverable is a Markdown memo suitable for chat output and optional vault
storage. It must:

- use the correct mode-specific template
- avoid marketing language
- be concise but analytical
- rate and score where the template asks for ratings or scores
- state missing information that would change the decision
- stay comparison-friendly across companies in company mode
- stay thesis-friendly and market-comparable across themes in theme mode

For all scoring:

- use the rubric anchors in `references/taxonomy-rubric.md`
- do not inflate uncertain companies toward the middle by default
- explain each score in one or two direct sentences

When using vault-only evidence, label it clearly as prior private context.
When using ROS MCP, label it clearly as market intelligence.
When using public sources, label them as public evidence.

## Vault Write-Back

Write the final company-mode note to:

- `<vault>/<output_root>/<company-slug>.md`

Use lowercase kebab-case for the filename slug.

Prefer `write_file` with the full final Markdown content unless there is a
strong reason to make a smaller anchored update with `patch`.

Include YAML frontmatter matching the template, including:

- company
- aliases
- date
- stage
- recommendation
- overall_score
- primary_theme
- secondary_themes
- cyber_category
- customer_type
- buyer
- geography
- confidence
- source_note_paths
- used_returnonsecurity_mcp
- tags

If an existing note already exists for the same company:

- read it first
- preserve useful historical context
- replace stale conclusions rather than appending contradictory summaries
- keep the current memo date accurate

Write the final theme-mode note to:

- `<vault>/<theme_output_root>/theme-<theme-slug>.md`

Use theme mode only for market, category, and thesis analysis. Thematic notes
should include:

- scope
- geography
- time_horizon
- primary_theme
- related_themes
- key_companies
- used_returnonsecurity_mcp
- source_note_paths
- confidence
- tags

Do not store a theme note under `4.Resources/Companies` unless the user
explicitly asks for that override.

## Common Pitfalls

1. Treating vault notes as public evidence.
   Keep private context separate from externally verifiable claims.

2. Overstating certainty from thin startup data.
   If the company is early and evidence is sparse, lower confidence instead of
   inventing precision.

3. Using generic cyber taxonomy.
   Force the classification, themes, thesis mapping, and buyer analysis to fit
   the actual company, not a canned category description.

4. Letting competitor lists become exhaustive.
   Focus on the set that sharpens the investment view, not every company in the
   category.

5. Forcing a company memo onto a thematic question.
   Switch to theme mode when the user is asking about a market or thesis rather
   than one startup.

6. Writing a memo that cannot be compared later.
   Preserve the exact section order, rating scales, and frontmatter fields.

7. Depending on a specific ROS MCP tool name.
   Depend only on the `mcp_returnonsecurity_*` or `mcp_signal_mcp_*` prefixes
   and choose tools from the actual available list.

## Verification Checklist

- [ ] Vault path resolved before reading or writing notes
- [ ] Configured search roots used before broader vault search
- [ ] ROS MCP availability checked via `mcp_returnonsecurity_*` or
      `mcp_signal_mcp_*` tool prefix
- [ ] Correct mode selected before drafting
- [ ] Company memo or thematic memo structure present as required
- [ ] Major conclusions distinguish facts, inferences, and assumptions
- [ ] Frontmatter populated when writing back to the vault
- [ ] Output note saved under the correct company or theme output root
- [ ] Missing information and confidence gaps stated explicitly
