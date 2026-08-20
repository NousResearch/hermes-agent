---
title: "Squirrelscan — Audit websites for SEO, performance, and security issues"
sidebar_label: "Squirrelscan"
description: "Audit websites for SEO, performance, and security issues"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Squirrelscan

Audit websites for SEO, performance, and security issues.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/squirrelscan` |
| Path | `optional-skills/web-development/squirrelscan` |
| Version | `0.1.0` |
| Author | squirrelscan contributors, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Web`, `QA`, `SEO`, `Audit`, `CLI` |
| Related skills | [`dogfood`](/docs/user-guide/skills/bundled/software-development/software-development-dogfood) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# squirrelscan

squirrelscan is a website QA/audit CLI: 272 deterministic rules across 21 categories (Core SEO, crawlability, performance, security, accessibility, agent experience, and more). It crawls a site, scores it 0-100 per category, and emits machine-readable reports with exact fixes. The local audit engine is free and works offline; cloud features (browser rendering, publishing) require login and credits. Upstream also ships an optional MCP server (`squirrel mcp`), which this skill does not use.

## When to Use

- The user asks to audit, QA, or health-check a website (SEO, accessibility, performance, security headers, broken links).
- After building or deploying a site, to verify quality before handoff.
- The user wants a machine-readable issue list (JSON/markdown) to drive automated fixes.
- CI-style gating: fail on score or severity thresholds.

## Prerequisites

- Node.js 18+ (npm). The npm package downloads a self-contained platform binary.
- Install locally in a scratch/project dir (package name is `squirrelscan`, but the installed binary is `squirrel`):

```
terminal(command="mkdir -p ~/scratch/squirrelscan && cd ~/scratch/squirrelscan && npm install squirrelscan")
```

## How to Run

Invoke through the Hermes `terminal` tool. The binary lands at `node_modules/.bin/squirrel` — show the full path once, then use it directly:

```
terminal(command="cd ~/scratch/squirrelscan && NO_TELEMETRY=1 ./node_modules/.bin/squirrel audit https://example.com --offline --format json --output report.json -y", timeout=600)
```

Notes:
- The binary is `squirrel`, NOT the package name — there is no `squirrelscan` binary.
- `--offline` skips cloud features, publishing, and telemetry; `NO_TELEMETRY=1` (any value) also disables telemetry and install registration.
- Crawls can take minutes on large sites; cap with `--max-pages` and set a generous `terminal` timeout.
- Read the JSON output afterwards with `read_file` (top-level keys: `meta`, `status`, `score`, `summary`, `issues`, `technologies`).

## Quick Reference

All commands verified against `squirrel --help` / `squirrel audit --help` / `squirrel report --help` (v0.0.85):

- `squirrel audit <URL>` — crawl + run all rules on a URL
- `squirrel audit <URL> --offline -y` — fully offline, no prompts
- `squirrel audit <URL> -m 25` / `--max-pages 25` — cap crawled pages (default coverage `quick` = 25, cap 5000)
- `squirrel audit <URL> --max-depth 2` — cap crawl depth from the seed
- `squirrel audit <URL> -f json -o report.json` — formats: console, text, json, html, markdown, xml, llm
- `squirrel audit <URL> --summary` — score + category breakdown only (console format)
- `squirrel audit <URL> --fail-on "score<90,errors>0"` — exit 2 when a threshold trips (CI gating)
- `squirrel audit <URL> --rule-include ax,perf` / `--rule-exclude images,social` — filter rule categories
- `squirrel audit <URL> -r` / `--refresh` — ignore cache, full re-fetch
- `squirrel audit <URL> -H "Name: Value"` — custom header on every crawl request (repeatable)
- `squirrel report` — view latest stored audit; `squirrel report --list` lists recent audits
- `squirrel report <id-or-domain> -f json --severity error --category core,links` — filtered re-query
- `squirrel report --diff <baseline-id-or-domain>` — compare reports
- `squirrel crawl` — crawl only, no analysis; `squirrel analyze` — run rules on a stored crawl
- `squirrel init` — create squirrel.toml config
- `squirrel self settings set telemetry false` — disable telemetry permanently

## Procedure

1. Install (once) into a scratch dir:

```
mkdir -p ~/scratch/squirrelscan && cd ~/scratch/squirrelscan && npm install squirrelscan
```

2. Run an offline audit with a page cap and JSON output (via `terminal`, timeout 600s):

```
cd ~/scratch/squirrelscan && NO_TELEMETRY=1 ./node_modules/.bin/squirrel audit https://TARGET --offline --max-pages 25 --format json --output report.json -y
```

3. Read the results with `read_file` on `report.json`, or summarize in the console:

```
./node_modules/.bin/squirrel report --summary
```

   JSON shape: `score.overall` (0-100) and `score.grade`, `score.groups` (seo/performance/security/ai with passed/warnings/failed counts), `score.categories` (per-category scores), `summary` (&#123;passed, warnings, failed&#125;), and `issues[]` — each issue has `ruleId`, `name`, `description`, `solution`, `category`, `group`, `severity`, `checks`.

4. For agent-driven fixing, work through `issues[]` sorted by `severity`, applying each `solution`, then re-audit with `--refresh` and diff:

```
./node_modules/.bin/squirrel report --diff TARGET-DOMAIN
```

5. For CI-style gating, add `--fail-on "severity>=error"` and check the exit code (2 = threshold tripped).

## Pitfalls

- **Binary name trap**: the npm package is `squirrelscan` but the binary in `node_modules/.bin/` is `squirrel`. Running the package name fails.
- **Telemetry on by default**: minimal (event name, version, random install ID) but enabled unless you pass `--offline`, set `NO_TELEMETRY=1`, or run `squirrel self settings set telemetry false`. The `report` command prints a telemetry notice on first runs.
- **Cloud upsell paths**: `--publish` is the default when signed in; `--render`, `credits`, `keys`, and `auth` are cloud/account features. Basic audits need no account — stay `--offline` for deterministic local runs.
- **Prompts**: audits may prompt for a project name; pass `-y` and/or `-n <name>` in unattended runs.
- **Crawl time and scope**: default coverage `quick` caps at 25 pages; without `-m` a `full` run can crawl 500+ pages and take a long time. Bot protection (e.g. Cloudflare) can make results incomplete — the tool warns when it detects this.
- **Results are stored**: audits persist to a local database (`~/.squirrel/`); `squirrel report` re-queries them without re-crawling.

## Verification

```
terminal(command="cd ~/scratch/squirrelscan && NO_TELEMETRY=1 ./node_modules/.bin/squirrel audit https://example.com --offline --max-pages 5 --format json --output /tmp/sq-verify.json -y && python3 -c \"import json; d=json.load(open('/tmp/sq-verify.json')); print('score', d['score']['overall'], d['summary'])\"", timeout=600)
```

Expect a line like `score 31 {'passed': 89, 'warnings': 26, 'failed': 5}`.
