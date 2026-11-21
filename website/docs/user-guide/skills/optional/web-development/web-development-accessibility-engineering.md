---
title: "Accessibility Engineering — Audit and fix web UI for WCAG 2.2 AA accessibility"
sidebar_label: "Accessibility Engineering"
description: "Audit and fix web UI for WCAG 2.2 AA accessibility"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Accessibility Engineering

Audit and fix web UI for WCAG 2.2 AA accessibility.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/accessibility-engineering` |
| Path | `optional-skills/web-development/accessibility-engineering` |
| Version | `1.0.0` |
| Author | Addy Osmani (addyosmani/agent-skills), adapted by Nous Research |
| License | MIT |
| Platforms | linux, macos |
| Tags | `accessibility`, `a11y`, `wcag`, `axe-core`, `pa11y`, `screen-reader`, `keyboard`, `contrast`, `audit` |
| Related skills | [`dogfood`](../../bundled/software-development/software-development-dogfood.md), [`claude-design`](../../bundled/creative/creative-claude-design.md), [`popular-web-designs`](../../bundled/creative/creative-popular-web-designs.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Accessibility Engineering Skill

> Ported from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/8c99fa36d90068c82d547f7c12a9d67a3e95f4ee/skills/accessibility-engineering) (MIT, © 2025 Addy Osmani) at `8c99fa3`. The upstream checklist and auditor persona are vendored under `references/`.

## Overview

Accessibility is not a polish step — it's a legal requirement in many jurisdictions, an engineering quality standard, and a prerequisite for shipping. This skill audits existing interfaces, triages the findings, and remediates verified violations. The goal is WCAG 2.2 AA compliance supported by automated tools and manual testing.

## When to Use

- Before shipping a user-facing change that needs an accessibility audit
- When a11y issues are reported (e.g., "screen reader can't read this", "keyboard navigation broken")
- When auditing an existing interface for compliance
- When triaging axe-core or pa11y findings
- When remediating verified accessibility violations

**Do NOT use** for building new UI components from scratch; use `claude-design` or `popular-web-designs` for construction and come back once an interface exists. Do not use for backend-only changes, CLIs, or library code with no UI output.

## Workflow

### Phase 1: Route New UI Work

This skill audits and fixes an interface that already exists. For building one, use a design skill first, then return here once there is a URL or file to audit.

### Phase 2: Run Automated Audit

Execute accessibility checks in this order:

```bash
# axe-core rules through pinned @axe-core/cli (the axe-core package ships no CLI)
npx --yes @axe-core/cli@4.13.0 http://localhost:3000 --exit

# Keep the JSON for the accessibility-auditor
npx --yes @axe-core/cli@4.13.0 http://localhost:3000 --dir ./a11y-reports --save axe.json

# Pinned pa11y for HTML validation (WCAG2AA rules via HTML_CodeSniffer)
npx --yes pa11y@9.1.1 http://localhost:3000
```

For a combined report, run `bash "$HERMES_SKILL_DIR/scripts/audit-a11y.sh" url <url> --json`. It runs both pinned tools, records their status and versions, and fails rather than reporting a clean result when either tool does not produce a valid report.

Review the violations grouped by severity:
- **Critical** — Blocks screen reader access entirely (missing alt text on info images, empty buttons)
- **Serious** — Makes content difficult or impossible to access (low contrast, missing labels)
- **Moderate** — Causes confusion or extra effort (heading level skips, redundant ARIA)
- **Minor** — Cosmetic or progressive enhancement (missing lang attribute on inner elements)

For each violation, verify it's a real issue (not a false positive) by inspecting the element in the browser.

### Phase 3: Manual Testing

Automated checks are a smoke test, not proof of conformance. Complete the audit manually.

**Keyboard audit:**
1. Tab through every interactive element — can you see where you are?
2. Activate buttons and links with Enter/Space
3. Close modals with Escape
4. Navigate lists with Arrow keys
5. Verify no keyboard traps exist

**Screen reader audit (VoiceOver on macOS, NVDA on Windows):**
1. Navigate by heading (VoiceOver: Ctrl+Option+Cmd+H) — is the structure logical?
2. Navigate by landmark (open the rotor with Ctrl+Option+U, choose Landmarks) — are regions labeled?
3. Navigate by link (open the rotor with Ctrl+Option+U, choose Links) — do link descriptions make sense out of context?
4. Interact with a form — are labels announced? Are error messages read?
5. Navigate a dynamic update — are changes announced?

**Visual audit:**
1. Verify color contrast (4.5:1 text, 3:1 large text, 3:1 UI components)
2. Zoom to 200% — does layout break or overlap?
3. Test with forced colors / high contrast mode (Windows High Contrast)
4. Reduce motion (prefers-reduced-motion) — are animations disabled?
5. Test focus visibility — is the focus ring visible against all backgrounds?

### Phase 4: Fix Violations

For each finding, apply the remediation in order:

1. **Fix the root cause** — Replace a `<div>` with `<button>`, add `<label>`, set `alt` text
2. **Add ARIA as bridge** — If native HTML can't express the pattern, add ARIA attributes
3. **Test the fix** — Re-run the automated check and manual verification

Apply the accessibility checklist (`skill_view(name="accessibility-engineering", file_path="references/accessibility-checklist.md")`) as a minimal baseline.

## Hermes Setup Notes

- Needs Node.js with `npx` and a local Chrome or Chromium. Both tools are fetched by `npx` at the pinned versions on first run (network required).
- `pa11y` drives Chrome through Puppeteer. If it reports `Could not find Chrome (ver. …)`, point it at the system browser: `PUPPETEER_EXECUTABLE_PATH=$(command -v google-chrome || command -v chromium)`.
- `@axe-core/cli` ships its own chromedriver, which must match the local Chrome major version. On a mismatch (`This version of ChromeDriver only supports Chrome version …`), fetch a matching driver and pass it in:
  `npx --yes @puppeteer/browsers install chromedriver@"$(google-chrome --version | grep -oE '[0-9.]+$')" --path "$HOME/.cache/a11y"`, then `AXE_CHROMEDRIVER_PATH=<printed path> bash "$HERMES_SKILL_DIR/scripts/audit-a11y.sh" url <url> --json`.
- `SessionNotCreatedError: Chrome instance exited` with a matching driver usually means `TMPDIR` is too long (measured: fails past ~66 characters, as under deep profile or sandbox homes): rerun with a short one, `TMPDIR="$(mktemp -d "$HOME/.a11y-XXXX")"`.
- Local files: serve them first (`python3 -m http.server 8000` in a background `terminal`) and audit the `http://127.0.0.1:8000/...` URL; `file` mode skips axe.
- The script exits 1 and refuses to print a violation total when either tool fails, so a crash never reads as a clean page. Report that as "audit incomplete", never as zero violations.
- For keyboard and visual checks you can drive the page yourself with the browser tools (Tab through it, take screenshots at 200% zoom and inspect them with `vision_analyze`). Screen reader checks still need a human with VoiceOver or NVDA: tag those findings `manual`.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "We'll fix a11y later" | Delayed remediation compounds across components and can create legal liability. Audit before shipping. |
| "Automated tools caught everything" | Automated checks cannot prove keyboard or screen reader behavior. Manual testing is mandatory. |
| "Screen reader users are a tiny minority" | Accessibility is a requirement, not a usage-share calculation. Keyboard navigation, captions, and clear structure also help users across contexts. |
| "This is an internal tool, it doesn't need a11y" | Internal tools are subject to the same regulations. Disabled employees and contractors need access. |
| "I'll just use aria-label everywhere" | ARIA is a bridge, not a foundation. Semantic HTML first. Overusing ARIA creates maintenance burden and can make things worse. |
| "The design system handles accessibility" | Design systems provide accessible primitives. How you compose them determines whether the result is accessible. |
| "The audit passed axe-core, so we're done" | Pass axe-core as a smoke test, then test with screen readers and keyboard. A green lighthouse score is not a clean bill of health. |

## Red Flags

- Elements that look clickable but aren't keyboard-focusable
- Color as the sole indicator of state (red for error, green for success)
- Focus outline removed without a visible replacement
- Missing labels on form inputs
- No heading hierarchy or skipped heading levels
- Empty or unlabeled buttons and links
- Modals that don't trap focus or restore focus on close
- No `alt` text on meaningful images (or `alt=""` missing on decorative ones)
- Touch targets that fail SC 2.5.8's 24x24 CSS px size-or-spacing requirement and do not qualify for an exception (AA; 44x44 is the AAA target, SC 2.5.5)
- Autoplaying video or audio without controls
- Low contrast text (especially on hover, disabled states, placeholders)

## See Also

- `references/accessibility-checklist.md` — Quick reference for WCAG 2.2 AA (load for every audit)
- `references/accessibility-auditor.md` — Report-style audit persona and output format for deep reviews; hand it to `delegate_task` for a clean-context pass

## Verification

After completing the accessibility workflow:

- [ ] Automated audit passes with no Critical or Serious violations
- [ ] All interactive elements are keyboard accessible (Tab, Enter, Escape, Arrow keys)
- [ ] Screen reader navigation by heading, landmark, and link verified with a named browser + AT pair (or recorded as `manual` and untested — never reported as verified)
- [ ] Focus order matches visual order
- [ ] Color contrast passes 4.5:1 text / 3:1 large text / 3:1 UI components
- [ ] All form inputs have associated labels
- [ ] All images have appropriate `alt` text
- [ ] Dynamic content changes are announced (aria-live)
- [ ] Page is usable at 200% zoom with no content loss
- [ ] Touch targets satisfy SC 2.5.8's 24x24 CSS px size-or-spacing requirement or a normative exception (AA); 44x44 is the enhanced AAA target (SC 2.5.5)
