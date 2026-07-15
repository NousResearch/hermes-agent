---
title: "Cyber Vc Competitors"
sidebar_label: "Cyber Vc Competitors"
description: "Use when building a competitive landscape, positioning map, or reusable category watchlist for cybersecurity investing"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cyber Vc Competitors

Use when building a competitive landscape, positioning map, or reusable category watchlist for cybersecurity investing.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/cyber-vc-competitors` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cybersecurity`, `venture`, `competitors`, `market-map`, `research` |
| Related skills | [`cyber-vc-analyst`](/docs/user-guide/skills/bundled/research/research-cyber-vc-analyst), [`cyber-vc-theme`](/docs/user-guide/skills/bundled/research/research-cyber-vc-theme), [`cyber-vc-company`](/docs/user-guide/skills/bundled/research/research-cyber-vc-company) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cyber VC Competitors

Use this skill for category landscaping and positioning work such as:

- `browser security`
- `deep AI SOC`
- `machine identity security`

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use competitors mode, not company mode.
2. Reuse relevant theme and company artifacts before broader rescans.
3. Default to `research-state` persistence for deep landscapes.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this competitor landscape to the vault?`
