---
title: "Cyber Vc Compare — Use when comparing two cybersecurity startups or adjacent cyber investment opportunities"
sidebar_label: "Cyber Vc Compare"
description: "Use when comparing two cybersecurity startups or adjacent cyber investment opportunities"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cyber Vc Compare

Use when comparing two cybersecurity startups or adjacent cyber investment opportunities.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/cyber-vc-compare` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cybersecurity`, `venture`, `compare`, `startup`, `research` |
| Related skills | [`cyber-vc-analyst`](/docs/user-guide/skills/bundled/research/research-cyber-vc-analyst), [`cyber-vc-company`](/docs/user-guide/skills/bundled/research/research-cyber-vc-company), [`cyber-vc-theme`](/docs/user-guide/skills/bundled/research/research-cyber-vc-theme) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cyber VC Compare

Use this skill for side-by-side comparison work such as:

- `Red Access Security vs Noma Security`
- `deep Wiz vs Orca Security`

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Normalize whether the comparison is direct, adjacent, or thesis-level.
2. Reuse prior company and theme artifacts before broader rescans.
3. Persist `research-state` when one or both companies need deeper follow-up.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this comparison to the vault or expand either company into a full memo?`
