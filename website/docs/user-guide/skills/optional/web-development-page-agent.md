---
title: "Page Agent"
sidebar_label: "Page Agent"
description: "Embed alibaba/page-agent into your own web application — a pure-JavaScript in-page GUI agent that ships as a single <script> tag or npm package and lets end-..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Page Agent

Embed alibaba/page-agent into your own web application — a pure-JavaScript in-page GUI agent that ships as a single &lt;script> tag or npm package and lets end-users of your site drive the UI with natural language ("click login, fill username as John"). No Python, no headless browser, no extension required. Use this skill when the user is a web developer who wants to add an AI copilot to their SaaS / admin panel / B2B tool, make a legacy web app accessible via natural language, or evaluate page-agent against a local (Ollama) or cloud (Qwen / OpenAI / OpenRouter) LLM. NOT for server-side browser automation — point those users to Hermes' built-in browser tool instead.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/page-agent` |
| Path | `optional-skills/web-development/page-agent` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `web`, `javascript`, `agent`, `browser`, `gui`, `alibaba`, `embed`, `copilot`, `saas` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# page-agent

alibaba/page-agent (https://github.com/alibaba/page-agent, 17k+ stars, MIT) is an in-page GUI agent written in TypeScript. It lives inside a webpage, reads the DOM as text (no screenshots, no multi-modal LLM), and executes natural-language instructions like "click the login button, then fill username as John" against the current page. Pure client-side — the host site just includes a script and passes an OpenAI-compatible LLM endpoint.

## When to use this skill

Load this skill when a user wants to:

- **Ship an AI copilot inside their own web app** (SaaS, admin panel, B2B tool, ERP, CRM) — "users on my dashboard should be able to type 'create invoice for Acme Corp and email it' instead of clicking through five screens"
- **Modernize a legacy web app** without rewriting the frontend — page-agent drops on top of existing DOM
- **Add accessibility via natural language** — voice / screen-reader users drive the UI by describing what they want
- **Demo or evaluate page-agent** against a local (Ollama) or hosted (Qwen, OpenAI, OpenRouter) LLM
- **Build interactive training / product demos** — let an AI walk a user through "how to submit an expense report" live in the real UI

## When NOT to use this skill

- User wants **Hermes itself to drive a browser** → use Hermes' built-in browser tool (Browserbase / Camofox). page-agent is the *opposite* direction.
- User wants **cross-tab automation without embedding** → use Playwright, browser-use, or the page-agent Chrome extension
- User needs **visual grounding / screenshots** → page-agent is text-DOM only; use a multimodal browser agent instead

## Prerequisites

- Node 22.13+ or 24+, npm 10+ (docs claim 11+ but 10.9 works fine)
- An OpenAI-compatible LLM endpoint: Qwen (DashScope), OpenAI, Ollama, OpenRouter, or anything speaking `/v1/chat/completions`
- Browser with devtools (for debugging)

## Path 1 — 30-second demo via CDN (no install)

Fastest way to see it work. Uses alibaba's free testing LLM proxy — **for evaluation only**, subject to their terms.

Add to any HTML page (or paste into the devtools console as a bookmarklet):

```html
<script src="https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js" crossorigin="true"></script>
```

A panel appears. Type an instruction. Done.

Bookmarklet form (drop into bookmarks bar, click on any page):

```javascript
javascript:(function(){var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js';document.head.appendChild(s);})();
```

## Path 2 — npm install into your own web app (production use)

Inside an existing web project (React / Vue / Svelte / plain):

```bash
npm install page-agent
```

Wire it up with your own LLM endpoint — **never ship the demo CDN to real users**:

```javascript
import { PageAgent } from 'page-agent'

const agent = new PageAgent({
    model: 'qwen3.5-plus',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.LLM_API_KEY,   // never hardcode
    language: 'en-US',
})

// Show the panel for end users:
agent.panel.show()

// Or drive it programmatically:
await agent.execute('Click submit button, then fill username as John')
```

Provider examples (any OpenAI-compatible endpoint works):

| Provider | `baseURL` | `model` |
|----------|-----------|---------|
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen3:14b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.6` |

**Key config fields** (passed to `new PageAgent({...})`):

- `model`, `baseURL`, `apiKey` — LLM connection
- `language` — UI language (`en-US`, `zh-CN`, etc.)
- Allowlist and data-masking hooks exist for locking down what the agent can touch — see https://alibaba.github.io/page-agent/ for the full option list

**Security.** Don't put your `apiKey` in client-side code for a real deployment — proxy LLM calls through your backend and point `baseURL` at your proxy. The demo CDN exists because alibaba runs that proxy for evaluation.

## Path 3 — clone the source repo (contributing, or hacking on it)

Use this when the user wants to modify page-agent itself, test it against arbitrary sites via a local IIFE bundle, or develop the browser extension.

```bash
git clone https://github.com/alibaba/page-agent.git
cd page-agent
npm ci              # exact lockfile install (or `npm i` to allow updates)
```

Create `.env` in the repo root with an LLM endpoint. Example:

```
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

Ollama flavor:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=NA
LLM_MODEL_NAME=qwen3:14b
```

Common commands:

```bash
npm start           # docs/website dev server
npm run build       # build every package
npm run dev:demo    # serve IIFE bundle at http://localhost:5174/page-agent.demo.js
npm run dev:ext     # develop the browser extension (WXT + React)
npm run build:ext   # build the extension
```

**Test on any website** using the local IIFE bundle. Add this bookmarklet:

```javascript
javascript:(function(){var s=document.createElement('script');s.src=`http://localhost:5174/page-agent.demo.js?t=${Math.random()}`;s.onload=()=>console.log('PageAgent ready!');document.head.appendChild(s);})();
```

Then: `npm run dev:demo`, click the bookmarklet on any page, and the local build injects. Auto-rebuilds on save.

**Warning:** your `.env` `LLM_API_KEY` is inlined into the IIFE bundle during dev builds. Don't share the bundle. Don't commit it. Don't paste the URL into Slack. (Verified: grepping the public dev bundle returns the literal values from `.env`.)

## Repo layout (Path 3)

Monorepo with npm workspaces. Key packages:

| Package | Path | Purpose |
|---------|------|---------|
| `page-agent` | `packages/page-agent/` | Main entry with UI panel |
| `@page-agent/core` | `packages/core/` | Core agent logic, no UI |
| `@page-agent/mcp` | `packages/mcp/` | MCP server (beta) |
| — | `packages/llms/` | LLM client |
| — | `packages/page-controller/` | DOM ops + visual feedback |
| — | `packages/ui/` | Panel + i18n |
| — | `packages/extension/` | Chrome/Firefox extension |
| — | `packages/website/` | Docs + landing site |

## Verifying it works

After Path 1 or Path 2:
1. Open the page in a browser with devtools open
2. You should see a floating panel. If not, check the console for errors (most common: CORS on the LLM endpoint, wrong `baseURL`, or a bad API key)
3. Type a simple instruction matching something visible on the page ("click the Login link")
4. Watch the Network tab — you should see a request to your `baseURL`

After Path 3:
1. `npm run dev:demo` prints `Accepting connections at http://localhost:5174`
2. `curl -I http://localhost:5174/page-agent.demo.js` returns `HTTP/1.1 200 OK` with `Content-Type: application/javascript`
3. Click the bookmarklet on any site; panel appears

## Pitfalls

- **Demo CDN in production** — don't. It's rate-limited, uses alibaba's free proxy, and their terms forbid production use.
- **API key exposure** — any key passed to `new PageAgent({apiKey: ...})` ships in your JS bundle. Always proxy through your own backend for real deployments.
- **Non-OpenAI-compatible endpoints** fail silently or with cryptic errors. If your provider needs native Anthropic/Gemini formatting, use an OpenAI-compatibility proxy (LiteLLM, OpenRouter) in front.
- **CSP blocks** — sites with strict Content-Security-Policy may refuse to load the CDN script or disallow inline eval. In that case, self-host from your origin.
- **Restart dev server** after editing `.env` in Path 3 — Vite only reads env at startup.
- **Node version** — the repo declares `^22.13.0 || >=24`. Node 20 will fail `npm ci` with engine errors.
- **npm 10 vs 11** — docs say npm 11+; npm 10.9 actually works fine.

## Reference

- Repo: https://github.com/alibaba/page-agent
- Docs: https://alibaba.github.io/page-agent/
- License: MIT (built on browser-use's DOM processing internals, Copyright 2024 Gregor Zunic)
