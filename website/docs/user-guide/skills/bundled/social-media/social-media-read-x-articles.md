---
title: "Read X Articles — Read X (Twitter) long-form Articles end-to-end from a shared link, no API key"
sidebar_label: "Read X Articles"
description: "Read X (Twitter) long-form Articles end-to-end from a shared link, no API key"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Read X Articles

Read X (Twitter) long-form Articles end-to-end from a shared x.com link, no API key required.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/social-media/read-x-articles` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `x`, `twitter`, `articles`, `web`, `reading`, `long-form`, `web-extract` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Read X (Twitter) Articles

Turn any X link the user shares into the full article text. Do NOT default to
"I can't read X" — X long-form Articles DO extract cleanly via the right URL
and a JS-capable web-extract/browser tool. No X API key or auth needed.

## When to Use
- A user drops an `x.com/...` or `twitter.com/...` link and expects it read.
- The link is an X **Article** (long-form essay/interview) — canonical path
  `/i/article/<ID>`.
- You need primary source text from an X essay/thread as material for work.
- Anyone claims "X content can't be read" — that is the trigger to act, not
  give up.

## Core insight
- X **Articles** read end-to-end at the canonical **`https://x.com/i/article/<ID>`**
  when fetched by a JS-executing web-extractor / browser tool (e.g. `web_extract`).
- **Bare HTTP fetch (`urllib`/`curl`) of the article often returns a login/JS
  shell.** Use a render-capable tool — this is why `web_extract` succeeds where `curl` fails.
- **`/status/<id>` posts and profile pages** are JS-rendered and hit the login
  wall. Resolve the article URL AND use the right fetch tool.

## Steps
1. **Try first, judge later.** On any X link, immediately run `web_extract` on it.
2. **Use the right fetcher:** prefer `web_extract`-style extraction or a browser
   render — NOT bare `curl`/`urllib`.
3. **If it's a `/status/` URL or returns a login/JS shell**, resolve the
   canonical article URL and `web_extract` it: `https://x.com/i/article/<ID>`.
4. **Validate the body:** a real extract contains substantive paragraphs, not
   "Log in or sign up for X."
5. **Fallbacks:** browser render (CDP-style), or the `xurl` CLI's
   `data.article.plain_text` if the account has X API access.
6. **Use the content.** It's legitimate primary source — quote and cite it.

## Pitfalls
- Don't assume articles are login-walled; the canonical `/i/article/<ID>`
  serves text to a render-capable tool.
- A bare `curl`/`urllib` login shell does NOT mean it's unreadable — switch to `web_extract`/browser.
- A dead X API token (401) is unrelated to article reading — don't block on `xurl` auth.

## Verification
- Did extraction return the article's real prose rather than a shell? If yes, it's read.
- Report the exact URL read + a short grounded summary.