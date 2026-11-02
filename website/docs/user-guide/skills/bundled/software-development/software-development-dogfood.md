---
title: "Dogfood — Exploratory QA of web apps: find bugs, evidence, reports"
sidebar_label: "Dogfood"
description: "Exploratory QA of web apps: find bugs, evidence, reports"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dogfood

Exploratory QA of web apps: find bugs, evidence, reports.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/dogfood` |
| Version | `1.0.0` |
| Author | Teknium (teknium1), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `qa`, `testing`, `browser`, `web`, `dogfood` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Dogfood: Systematic Web Application QA Testing

## Overview

This skill guides you through systematic exploratory QA testing of web applications with `browser_exec`. You will navigate the application, interact with elements, capture evidence of issues, and produce a structured bug report.

## Prerequisites

- The `browser` toolset must be enabled (`browser_exec`). `browser_exec` runs Python with pre-imported helpers (`new_tab`, `goto_url`, `wait_for_load`, `page_info`, `js`, `fill_input`, `click_at_xy`, `capture_screenshot`, `cdp`); browser state persists across calls, Python variables do not.
- A target URL and testing scope from the user

## Inputs

The user provides:
1. **Target URL** — the entry point for testing
2. **Scope** — what areas/features to focus on (or "full site" for comprehensive testing)
3. **Output directory** (optional) — where to save screenshots and the report (default: `./dogfood-output`)

## Workflow

Follow this 5-phase systematic workflow:

### Phase 1: Plan

1. Create the output directory structure:
<!-- ascii-guard-ignore -->
   ```
   {output_dir}/
   ├── screenshots/       # Evidence screenshots
   └── report.md          # Final report (generated in Phase 5)
   ```
<!-- ascii-guard-ignore-end -->
2. Identify the testing scope based on user input.
3. Build a rough sitemap by planning which pages and features to test:
   - Landing/home page
   - Navigation links (header, footer, sidebar)
   - Key user flows (sign up, login, search, checkout, etc.)
   - Forms and interactive elements
   - Edge cases (empty states, error pages, 404s)

### Phase 2: Explore

Batch each step's actions into one `browser_exec` call. For each page or feature in your plan:

1. **Navigate with a console hook installed.** Register the hook on the tab before navigating so errors thrown during page load are captured too:
   ```python
   # Open page with console-error capture
   HOOK = """window.__qa_errors = [];
   const _e = console.error;
   console.error = (...a) => { window.__qa_errors.push(a.map(String).join(' ')); _e(...a); };
   addEventListener('error', e => window.__qa_errors.push('uncaught: ' + e.message));
   addEventListener('unhandledrejection', e => window.__qa_errors.push('rejection: ' + String(e.reason)));"""
   new_tab('about:blank')  # first call only: the hook binds to the current tab
   cdp('Page.addScriptToEvaluateOnNewDocument', source=HOOK)
   goto_url("https://example.com/page")
   wait_for_load()
   print(page_info())
   print(js('window.__qa_errors || []'))
   ```
   Re-register the hook at the start of any later call that navigates. Silent JS errors are high-value findings.

2. **Read the page structure** — list visible interactive elements with their viewport centres (use `cdp('Accessibility.getFullAXTree')` and filter in Python when you need roles/names):
   ```python
   # List interactive elements
   print(js("""(() => [...document.querySelectorAll('a,button,input,select,textarea,[role=button]')]
     .filter(e => e.offsetParent).slice(0, 80).map((e, i) => { const r = e.getBoundingClientRect();
       return {i, tag: e.tagName, text: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 60),
               id: e.id, name: e.name, x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2)}; }))()"""))
   ```

3. **Visually assess the page:** call `capture_screenshot()` and inspect the attached image for layout issues, broken elements, or accessibility concerns. If your model cannot view images, pass the printed path to `vision_analyze`.

4. **Test interactive elements** systematically:
   - Click buttons and links: `js("document.querySelector('#submit').click()")`, or `click_at_xy(x, y)` with a centre from step 2
   - Fill forms: `fill_input('input[name=email]', 'test input')`
   - Test keyboard navigation: `cdp('Input.dispatchKeyEvent', type='keyDown', key='Tab', code='Tab', windowsVirtualKeyCode=9)` followed by the matching `keyUp`, then `js('document.activeElement.outerHTML.slice(0, 200)')` to see where focus landed; same pattern with `key='Enter'`, `code='Enter'`, `windowsVirtualKeyCode=13`
   - Scroll through content: `js('window.scrollBy(0, 800)')`
   - Go back: `js('history.back()')`, then `wait_for_load()`
   - Test form validation with invalid inputs
   - Test empty submissions

5. **After each interaction**, in the same call, check for:
   - Console errors: `print(js('window.__qa_errors || []'))`
   - Visual changes: `capture_screenshot()` and compare with the previous one
   - Expected vs actual behavior (`page_info()`, or `js()` reading the relevant element)

### Phase 3: Collect Evidence

For every issue found:

1. **Take a screenshot** showing the issue: `capture_screenshot()` prints the saved path. Copy it to `{output_dir}/screenshots/` under a descriptive name (e.g. `issue-03-broken-login.png`) — you will reference it in the report.

2. **Record the details**:
   - URL where the issue occurs
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Console errors (if any)
   - Screenshot path

3. **Classify the issue** using the issue taxonomy (see `references/issue-taxonomy.md`):
   - Severity: Critical / High / Medium / Low
   - Category: Functional / Visual / Accessibility / Console / UX / Content

### Phase 4: Categorize

1. Review all collected issues.
2. De-duplicate — merge issues that are the same bug manifesting in different places.
3. Assign final severity and category to each issue.
4. Sort by severity (Critical first, then High, Medium, Low).
5. Count issues by severity and category for the executive summary.

### Phase 5: Report

Generate the final report using the template at `templates/dogfood-report-template.md`.

The report must include:
1. **Executive summary** with total issue count, breakdown by severity, and testing scope
2. **Per-issue sections** with:
   - Issue number and title
   - Severity and category badges
   - URL where observed
   - Description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshot references (use `MEDIA:<screenshot_path>` for inline images)
   - Console errors if relevant
3. **Summary table** of all issues
4. **Testing notes** — what was tested, what was not, any blockers

Save the report to `{output_dir}/report.md`.

## Tools Reference

All browser actions are `browser_exec` calls; these are the helpers and idioms used above.

| Action | `browser_exec` idiom |
|------|---------|
| Go to a URL | `new_tab(url)` once (or `new_tab('about:blank')` before registering the hook), then `goto_url(url)`; follow with `wait_for_load()` |
| Page state | `page_info()` |
| DOM / accessibility tree | `js(...)` extraction, or `cdp('Accessibility.getFullAXTree')` filtered in Python |
| Click | `js("document.querySelector(sel).click()")` or `click_at_xy(x, y)` |
| Type into an input | `fill_input(selector, text)` |
| Scroll | `js('window.scrollBy(0, 800)')` |
| Go back | `js('history.back()')` |
| Press a key | `cdp('Input.dispatchKeyEvent', type='keyDown', key=..., code=..., windowsVirtualKeyCode=...)` + `keyUp` |
| Screenshot | `capture_screenshot()` (image attached to the result; text-only models use `vision_analyze` on the path) |
| JS console errors | hook via `cdp('Page.addScriptToEvaluateOnNewDocument', source=HOOK)`, read with `js('window.__qa_errors || []')` |

## Tips

- **Always read `window.__qa_errors` after navigating and after significant interactions.** Silent JS errors are among the most valuable findings.
- **Batch each sub-procedure into one `browser_exec` call** (navigate, wait, extract, act, check errors) — fewer round-trips, and the hook stays in place.
- **Take a `capture_screenshot()`** when you need to reason about element positions or when DOM extraction is ambiguous; `click_at_xy` takes the centres from the element listing.
- **Test with both valid and invalid inputs** — form validation bugs are common.
- **Scroll through long pages** — content below the fold may have rendering issues.
- **Test navigation flows** — click through multi-step processes end-to-end.
- **Check responsive behavior** by noting any layout issues visible in screenshots.
- **Don't forget edge cases**: empty states, very long text, special characters, rapid clicking.
- When reporting screenshots to the user, include `MEDIA:<screenshot_path>` so they can see the evidence inline.
