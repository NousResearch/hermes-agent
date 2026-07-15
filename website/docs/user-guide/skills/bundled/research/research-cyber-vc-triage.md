---
title: "Cyber Vc Triage — Use when giving a fast first-pass read on whether a cybersecurity startup deserves a full memo"
sidebar_label: "Cyber Vc Triage"
description: "Use when giving a fast first-pass read on whether a cybersecurity startup deserves a full memo"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cyber Vc Triage

Use when giving a fast first-pass read on whether a cybersecurity startup deserves a full memo.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/cyber-vc-triage` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cybersecurity`, `venture`, `triage`, `startup`, `research` |
| Related skills | [`cyber-vc-analyst`](/docs/user-guide/skills/bundled/research/research-cyber-vc-analyst), [`cyber-vc-company`](/docs/user-guide/skills/bundled/research/research-cyber-vc-company) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cyber VC Triage

Use this skill when the user wants a fast IC-style first pass rather than a
full memo.

Support terse invocations like:

- `Red Access Security`
- `quick Red Access Security`

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use triage mode and stay compact.
2. Identify whether the company deserves escalation into `cyber-vc-company`.
3. Carry forward gathered context if the user upgrades the run into a full memo.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this triage note to the vault or expand it into a full memo?`
