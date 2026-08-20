---
title: "Interface Design — UI/UX craft reviews: typography, color, a11y, layout"
sidebar_label: "Interface Design"
description: "UI/UX craft reviews: typography, color, a11y, layout"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Interface Design

UI/UX craft reviews: typography, color, a11y, layout.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/interface-design` |
| Path | `optional-skills/web-development/interface-design` |
| Version | `1.0.0` |
| Author | Jakub Krehel (ported by Hermes Agent) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `ui`, `ux`, `design`, `typography`, `color`, `accessibility`, `layout`, `ux-writing`, `frontend`, `review`, `web-development` |
| Related skills | [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Interface Design — build and review great UIs

Use this skill when building UI components, reviewing frontend code or a PR's
interface changes, choosing fonts or colors, fixing accessibility, writing
interface copy, or when a screen "feels off" and you need to say why. It is a
hub over eight domain guides (originally the `better-*` skills from
[interfaces.dev](https://interfaces.dev/)); load only the domains the task
touches.

> **Hermes adaptation notes**
> - Upstream ships these as 8 separate skills that reference each other by
>   name ("the `better-typography` skill"). Here each is a reference dir:
>   when a doc says "the `better-X` skill", read
>   `references/better-X/better-X.md` via
>   `skill_view(name="interface-design", file_path="references/better-X/better-X.md")`.
> - Relative links inside a domain doc (e.g. `choosing-fonts.md`) resolve to
>   siblings in the same `references/<domain>/` directory.
> - Upstream's `interface-review` is user-invoked change review (diff/branch/PR
>   scoped). In Hermes, run it when the user asks to review interface changes;
>   use `git diff`/`gh pr diff` via the terminal tool for scope resolution.
> - For visual verification of a running UI, pair with the browser tool +
>   `vision_analyze` screenshots; upstream assumes a human looking at the
>   screen.

## Domain map — load what the task needs

| Task | Load |
| --- | --- |
| Holistic review of a screen/flow/product (routes all domains, one ranked verdict) | `references/better-interface/better-interface.md` |
| Change-scoped review: uncommitted work, a branch, or a PR | `references/interface-review/interface-review.md` |
| Visual polish: radius, shadows, borders, animations, micro-interactions, icons | `references/better-ui/better-ui.md` |
| Fonts, type scale, line-height, wrapping, truncation, smart punctuation | `references/better-typography/better-typography.md` |
| Color palettes, OKLCH, contrast, gamut, theming | `references/better-colors/better-colors.md` |
| Focus, keyboard, ARIA, forms, screen readers, hit areas, motion prefs | `references/better-accessibility/better-accessibility.md` |
| Layout structure, grouping, alignment, reading order, breakpoints | `references/better-layout/better-layout.md` |
| Interface copy: buttons, errors, settings, empty states | `references/better-writing/better-writing.md` |

Each domain doc has its own sub-references (tables at the top of each doc) —
follow those links for deep dives (e.g. `css-cheat-sheet.md`,
`palette-generation.md`, `screen-readers.md`).

## Workflow

1. **Classify the request.** Building new UI → load the relevant domain docs
   before writing code. Reviewing → holistic (`better-interface`) or
   change-scoped (`interface-review`).
2. **Recon before judgment.** Identify the framework, styling system,
   component library, design tokens, and viewports. Write every fix in the
   project's own idiom — never introduce a second styling system alongside
   the existing one.
3. **Load the domain docs the scope touches** and apply their rules. Domain
   docs are the sources of truth; do not improvise rules the docs don't state.
4. **Report findings** in the format of the domain's `review-output.md`
   (severity scale, findings table, verification, verdict). Under a holistic
   review, `better-interface`'s consolidated format governs instead.
5. **Verify fixes.** Re-run the project's preview/test commands; for visual
   changes, screenshot via the browser tool and inspect with `vision_analyze`.

## Pitfalls

- Don't staple independent domain audits together for a holistic ask — use
  `better-interface`'s orchestration and its finding caps (quick=5, full=15).
- A documented project convention is not evidence the convention is good;
  "it's in the style guide" does not retire a finding — report it once
  against the guideline source.
- Never imply uninspected surfaces were reviewed; state the scope boundary.
- Severity discipline: domain docs' escalation triggers and severity scales
  are the source of truth; where a case isn't listed verbatim, HIGH =
  broken/blocking or fails a hard accessibility requirement (WCAG AA), and
  polish nits are LOW. Don't inflate.

## Verification

- `skill_view(name="interface-design", file_path="references/better-ui/better-ui.md")`
  loads a domain doc; every path in the domain map above resolves.
- The review-output format check: findings tables include location,
  severity, rule source, and a concrete fix in the project's idiom.
