---
title: "Accessibility Audit — Use when auditing a UI for WCAG 2.2 AA, keyboard and AT"
sidebar_label: "Accessibility Audit"
description: "Use when auditing a UI for WCAG 2.2 AA, keyboard and AT"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Accessibility Audit

Use when auditing a UI for WCAG 2.2 AA, keyboard and AT.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/accessibility-audit` |
| Path | `optional-skills/web-development/accessibility-audit` |
| Version | `0.1.0` |
| Author | rlaope (github.com/rlaope), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `accessibility`, `a11y`, `wcag`, `keyboard`, `screen-reader`, `frontend`, `audit` |
| Related skills | [`dogfood`](../../bundled/software-development/software-development-dogfood.md), [`adversarial-ux-test`](../../optional/dogfood/dogfood-adversarial-ux-test.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Accessibility Audit Skill

Audit a web UI (or a component library) against WCAG 2.2 AA: semantic structure, keyboard reach and
focus order, accessible names and announcements, target size, contrast and reflow, forms and status
messages. Findings carry a stable rule ID, severity, location, WCAG criterion and a fix class that
says whether the fix can be derived from markup (`auto`) or needs to know what the content means
(`manual`). The verdict is PASS only on observed keyboard and accessibility-tree evidence taken after
the latest UI change; a source read or an automated scan alone is HOLD.

Ported from [rlaope/oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT), `omh-accessibility-audit`.

## When to Use

- "Is this checkout flow WCAG 2.2 AA / keyboard / screen-reader OK?"
- An a11y pass before launch, or re-checking after fixes.
- Turning an axe/Lighthouse report into a ranked, rerunnable finding list.

Don't use for: general exploratory QA (`dogfood`), visual/pixel QA, or native mobile apps (rule IDs
still apply, but the observation steps below are web-only).

## Prerequisites

A browser tool: `browser_exec` (Browser Use helpers `js()` and `cdp()`) or `browser_cdp`. Optional:
Node for `npx lighthouse` or `npx @axe-core/cli`. For credentialed or destructive flows, a staging or
read-only path.

## Quick Reference

Run inside `browser_exec` after navigating to the page:

```python
# Accessibility tree: roles + accessible names (filter before printing; it is large)
nodes = cdp('Accessibility.getFullAXTree')['nodes']
named = [(n['role']['value'], n.get('name', {}).get('value', '')) for n in nodes if not n.get('ignored')]
print([r for r in named if r[0] in ('button', 'link', 'textbox', 'combobox', 'checkbox') and not r[1]])

# Keyboard walk: Tab 40 times, record what holds focus
for i in range(40):
    for t in ('keyDown', 'keyUp'):
        cdp('Input.dispatchKeyEvent', type=t, key='Tab', code='Tab', windowsVirtualKeyCode=9)
    print(i, js("(() => {const e = document.activeElement; const r = e.getBoundingClientRect(); return [e.tagName, e.getAttribute('aria-label') || e.textContent.trim().slice(0, 40), Math.round(r.width), Math.round(r.height), getComputedStyle(e).outlineStyle]})()"))
```

Automated scan (evidence, not a verdict), through `terminal`:
`npx --yes lighthouse <url> --only-categories=accessibility --output=json --quiet --chrome-flags="--headless"`.

## Procedure

1. **Scope.** Platform, surfaces (routes/components), the critical tasks (sign in, checkout, submit
   form), and the standard (default WCAG 2.2 AA). Done when each critical task has a start URL.
2. **Read structure.** Source (`read_file`, `search_files`) and the live accessibility tree: landmarks,
   heading order, page `lang`, labels and names, roles and states, live regions.
3. **Walk the keyboard** through every critical task: reachability, order matches visual order,
   visible focus, no trap, skip link, Escape closes dialogs and focus returns.
4. **Measure** target size (24×24 CSS px minimum, WCAG 2.5.8), contrast (4.5:1 text, 3:1 large text
   and UI components), reflow at 320 CSS px / 400% zoom, pointer-gesture alternatives.
5. **Forms and status:** errors announced and associated, required state exposed, status messages in
   a live region, autofill purpose on personal-data fields.
6. **Write findings** from `references/a11y-rules.md`, all at once after reading the whole surface:
   `rule ID | severity | location | WCAG | fix class | fix`. Split half-structural fixes into an
   `auto` row and a `manual` row.
7. **Verdict.** PASS only if steps 2–5 were observed after the latest UI edit and cover every critical
   task; HOLD when some evidence is missing (name it); BLOCK on any critical finding.
8. **Handoff.** `auto` rows go to whoever edits the code as one batch; `manual` rows go back with the
   question each needs answered (what does this image show, where does this link go). Re-audit after.

## Pitfalls

- **A meaning-dependent fix marked `auto` is a defect:** it produces confident, wrong alt text that
  reads worse than none.
- **Scans miss most keyboard and announcement problems.** A clean Lighthouse score is HOLD, not PASS.
- **Stale evidence:** screenshots or walks from before the latest UI change do not count.
- A source that hides the condition (runtime-computed names, fully dynamic bindings) is
  "unverifiable from source", not a pass — check it in the live tree.
- This checks automation-visible semantics; a real screen-reader session (NVDA, VoiceOver) is
  still the strongest evidence — say when it was not done.

## Verification

- Every critical task has a recorded keyboard walk and accessibility-tree read after the last edit.
- Every finding has rule ID, severity, location, WCAG criterion and fix class.
- The verdict names any missing evidence; PASS appears only with none missing.
