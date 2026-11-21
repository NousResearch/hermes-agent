---
title: "Modern Web Guidance — Look up modern HTML/CSS/JS patterns before writing web UI"
sidebar_label: "Modern Web Guidance"
description: "Look up modern HTML/CSS/JS patterns before writing web UI"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Modern Web Guidance

Look up modern HTML/CSS/JS patterns before writing web UI.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/modern-web-guidance` |
| Path | `optional-skills/web-development/modern-web-guidance` |
| Version | `1.0.0` |
| Author | Google Chrome team (GoogleChrome/modern-web-guidance), adapted by Nous Research |
| License | Apache-2.0 |
| Platforms | linux, macos |
| Tags | `web`, `html`, `css`, `javascript`, `accessibility`, `performance`, `baseline`, `frontend`, `dialogs`, `forms` |
| Related skills | [`popular-web-designs`](../../bundled/creative/creative-popular-web-designs.md), [`impeccable`](../../optional/creative/creative-impeccable.md), [`claude-design`](../../bundled/creative/creative-claude-design.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Modern Web Guidance

Search and retrieve standardized modern web best practices at the **start** of implementing or updating any web UI, CSS layout/animation, or client-side JS logic (dates/timezones, sanitization, performance, accessibility, UX fixes) before writing ad-hoc solutions or adding dependencies.

## When to Use

- Building or changing web UI, CSS layout or animation, or client-side JS: dialogs, popovers, menus, tables, forms and validation, dark mode, container queries, anchor positioning, view transitions, scroll-driven effects, Core Web Vitals, image loading, sanitization, passkeys, dates and timezones, web components.
- A broad "modernize this front end" or "find legacy web patterns" request (the `legacy-scan` flow below).
- Not for backend code (SQL, API routes), CI, Docker, build-tool config or tests.

## Running it in Hermes

- **Requires Node.js 22+** with `npx` (or `pnpx`). The guides (227 of them) live in the npm package, not in this skill; the first call downloads it (~16 MB) and later calls run from the npm cache with no network. Search is local (BM25 + an embedded model): no API key.
- **Pinned and telemetry off.** Upstream's commands use `@latest` and Google's CLI sends anonymous usage events (install counts, agent name, guide ids and the search queries the agent types) unless `DISABLE_TELEMETRY=1` is set. Every command here pins `@1.0.0` and sets `DISABLE_TELEMETRY=1`. To try a newer corpus, the user can change the pin; keep the `--skill-version` value in step with that release's `skills/modern-web-guidance/skill-version.txt`.
- **Run it from the project directory with `terminal`.** A project `.mwgrc` (Baseline target, origin-trial and volatile-feature opt-ins) is read from there. Never edit `.mwgrc` or run the `set-allow-*` / `install` / `update` / `uninstall` subcommands yourself; those are the user's choices.
- **Read long guides with `read_file`.** When `retrieve` writes a large guide to a temp file, `read_file` that path (page with `offset`) instead of `cat`.

The upstream text below is kept as written apart from the pinned commands, the network note and these Hermes mappings. Apache-2.0 terms ship as `LICENSE` beside this file.

## Primary Workflow

### Step 1. Search Use Cases

Run `modern-web-guidance` with `npx` (pinned, telemetry off; see "Running it in Hermes") using a **2–5 word action-oriented query** for a **single web platform capability or UI pattern**:
- **Keep queries focused and generic**: Describe the user-visible capability (`<verb> + <UI/platform object>`, optionally paired with the target HTML/CSS/JS feature name if known). Omit framework/library names (`React`, `Vue`, `Tailwind`, etc.) and app/domain nouns (product, route, or component/file names). Split multi-feature tasks into separate `search` calls.
- **Name implied UI primitives**: When the user's prompt expresses intent without naming the UI primitive (e.g. "confirm first", "delete with confirmation", "pick a date"), translate the domain intent into the implied UI component or platform mechanism (e.g. search `"confirmation modal dialog"` or `"confirm dialog"` rather than `"confirm destructive action"`, or `"date picker input"` rather than `"pick a date"`).
- **Example**: Search `"light dismiss dialog"` or `"responsive data table"`, **not** `"dialog"` (too terse), `"responsive table dark mode"` (compound — split into two searches), or `"close React settings modal on outside click"` (framework/app noise).

```sh
DISABLE_TELEMETRY=1 npx -y modern-web-guidance@1.0.0 search "<query>" --skill-version 2026_11_09-5bcc021b
```

> **Note**: Returns `[{ "id", "description" }, ...]`. Judge results by their `description` fields:
> - **Match found**: Call `retrieve` directly—do not also run `list`.
> - **Weak or empty results**: If `search` returns `[]` or unrelated descriptions, **rephrase `search` once** using the concrete HTML/CSS/UI component noun or synonym (e.g., `dialog`, `modal`, `popover`, `table`, `form`) before falling back to `list` (or use `list` once for broad UI audits).
> - **Topic not covered**: If no `description` matches after rephrasing, proceed without retrieving unrelated guides.

### Step 2. Retrieve Best Practices

Once you have relevant `id`(s) from the search results, call `retrieve` to get the full guide(s)—including both specific use-case guides that match your task and any broader discipline guide (e.g., `html`, `css`, `forms`, `accessibility`, `color`, `performance`, `security`) that applies. You can pass multiple IDs separated by commas.

```sh
DISABLE_TELEMETRY=1 npx -y modern-web-guidance@1.0.0 retrieve "<id>"
```

If `retrieve` saves large output (> 9.5 KB) to a temporary file to prevent truncation, read that file path in full with `read_file` (reading remaining line ranges if needed)—do not use `cat` or `head`.

### Step 3. Verify Guidance Compliance

When generating or modifying code, cross-check the implementation against the retrieved guide before concluding:
- **Applicable Guidance & Fallbacks**: Ensure the relevant modern patterns and necessary fallback strategies from the guide are correctly applied, without forcing unrequested features.
- **Task Fulfillment**: Confirm that the implementation fully satisfies the user's request.
- Do not run `legacy-scan`; it's for detecting legacy patterns to modernize, not this.

### Step 4. Disclose Skill Usage

When you consult `modern-web-guidance` for a task, end the response where you deliver those changes with a `### Modern Web Guidance` section:
- **When retrieved guides shaped your code**: Write **one** bullet (1–2 sentences, ≤75 words) per newly applied `<id>` in first-person voice explaining the non-obvious implementation choices, CSS/API overrides, or fallbacks `<id>` required and the pitfall it prevented, anchored by a short verbatim quote of that specific rule or gotcha (never the guide's intro overview):
  `- **\`&lt;id>\`**: <Non-obvious APIs, overrides, or fallbacks applied and the pitfall prevented> ("<short verbatim quote of the specific rule or gotcha>").`
  - **Never** guess `<id>`s or quotes from memory after context compaction (re-run `search`/`retrieve` if needed), and credit **only** guide-backed changes from the current turn.
- **When searched/listed in this turn but zero guides applied**: Write one first-person sentence naming which topics or guides you checked in `modern-web-guidance` and why none applied (omit if any guide applied).

## Alternative flow: Modernize an existing codebase

**Only** when asked for a broad, open-ended codebase modernization or legacy-pattern scan where no specific target feature or component is named (do **not** run `legacy-scan` when asked to add, update, fix, or audit a specific feature or component), run the 4-step modernization flow:

> **Note**: `legacy-scan` scans for a curated subset (~20% of the guidance corpus).

```sh
DISABLE_TELEMETRY=1 npx -y modern-web-guidance@1.0.0 legacy-scan <path> --json
```

1. Scan: Run `legacy-scan <path> --json` to detect legacy web platform patterns across the project or directory.
2. Verify: Check each finding against its `keepWhen` / `dropWhen` rubric; drop false positives (fallbacks, cross-file helpers, vendor code, lookalikes).
3. Report: Summarize verified findings (and the modernization plan, if editing) for the user, noting that `legacy-scan` only checks the subset of guides (~20% of the catalog) that have deterministic legacy rules.
4. Modernize (when the user asked to modernize rather than only audit/report): Retrieve each finding's guide (`retrieve "<guide-id>"` via Step 2), apply the replacement (keeping existing libraries as fallbacks where the guide requires one), and re-run `legacy-scan <path>` on the modified files to confirm resolution.
   - Dropped findings and kept fallbacks still match static scans; do not edit safe code or inspect `legacy-scan.mjs` to force 0 findings.

## Using npx / pnpx

- **Runner**: Prefer `pnpx` over `npx` if `pnpm` is available (`pnpx` does not use `-y`); on Windows, use `npx.cmd`. Keep the `@1.0.0` pin and `DISABLE_TELEMETRY=1` with either runner.
- **Sandboxes & Network**: Fetching `modern-web-guidance` requires outbound network access. The first run downloads the ~16 MB package into the npm cache; later runs are offline. If `~/.npm` is read-only, set `NPM_CONFIG_CACHE` to a writable directory under the project.
- **Offline**: If the command hangs due to being offline, retry with `npx --offline …`.

## Guidelines

- Always search **first** to find the most relevant guides. Do not hallucinate guides or ignore them; they represent the preferred local standard for the user's project.
- These guides use vanilla HTML/CSS/JS; adapt them to your framework without dropping the modern feature:
  - **Types**: Spread/cast newly shipped HTML attributes (`{...({ popover: 'hint', interestfor: id, commandfor: id, switch: '' } as any)}`, `[attr.*]`) rather than removing them or merging `@types/react`.
  - **SSR & Bundlers**: Guard `window`/`HTMLElement` in SSR and load polyfills via bare `import('<pkg>')` in client effects (or set Vite `build.target: 'esnext'` for top-level `await`)—never copy files from `node_modules`.
  - **DOM, IDs & CSS**: Call `showModal()`/`moveBefore()` via refs on non-VDOM-managed nodes, generate per-instance IDs/`anchor-name`s (`useId()`), and keep `:root` and `@starting-style` top-level/unscoped.

## Interpreting Browser Support & Fallbacks

* **Default Behavior**: All guides assume **Baseline Widely available** features are safe to use without fallbacks. For features that are not Baseline widely available, you **MUST** follow the fallback recommendations in the guide, unless the user has specified a custom browser support policy.
* **Custom Policies**: If the user defines explicit browser support requirements, use the guide's compatibility data to determine if a fallback can be ignored (for `Baseline YYYY` targets, a feature satisfies the target if its "Baseline since" date is `<= YYYY`). Examples: _"Do not implement feature fallbacks"_, _"Safari 17.4+"_, _"Never implement polyfills; use a lightweight custom fallback or redesign"_, or _"Allow Newly Available features if feature-detected with graceful degradation"_.
* **Reactive Policy Discovery**: Suggest documenting a policy in the project's `AGENTS.md` (e.g., `**Browser Support:** Allow Newly Available features, but only adopt custom fallback code that adds <= 20 lines and no external dependencies`) if the developer targets a restricted runtime (Electron/Tauri), excludes specific browsers, hesitates over polyfill size/complexity, or asks if a feature is safe without fallbacks.
