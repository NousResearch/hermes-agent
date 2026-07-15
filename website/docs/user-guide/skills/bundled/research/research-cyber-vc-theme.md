---
title: "Cyber Vc Theme"
sidebar_label: "Cyber Vc Theme"
description: "Use when evaluating a cybersecurity market theme, category, or architectural shift as an investment thesis"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cyber Vc Theme

Use when evaluating a cybersecurity market theme, category, or architectural shift as an investment thesis.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/cyber-vc-theme` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cybersecurity`, `venture`, `investing`, `theme`, `research` |
| Related skills | [`cyber-vc-analyst`](/docs/user-guide/skills/bundled/research/research-cyber-vc-analyst), [`cyber-vc-company`](/docs/user-guide/skills/bundled/research/research-cyber-vc-company), [`cyber-vc-competitors`](/docs/user-guide/skills/bundled/research/research-cyber-vc-competitors) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cyber VC Theme

Use this skill for thematic analysis such as:

- SOC Automation / AI SOC
- machine identity security
- browser security

Support terse invocations like `deep SOC Automation / AI SOC` and richer
free-form prompts.

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use theme mode, not company mode.
2. Bias toward relevant prior company, compare, and Matter-style category
   notes before broader rescans.
3. Default to `research-state` persistence for deep or long-running theme work.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this theme memo to the vault?`
