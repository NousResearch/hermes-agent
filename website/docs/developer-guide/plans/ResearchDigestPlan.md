# Research Digest MVP Plan

Owner: Sahil Saghir
Operator: KENSEI
Created: Saturday 2026-05-02 22:26 BST
Status: MVP implemented, hardening from first Telegram test feedback

## Decision

Build the Research Digest as the next Morning Brief lane.

Primary delivery stays Telegram. Bring forward the HTML view only as a static local artifact generated from the same digest data. Do not build a full dashboard, public web app, or Workspace panel until the Mailbox Digest, Research Digest, and System Report are stable.

## What other people are doing

### Hermes daily briefing pattern

Source: https://hermes-agent.nousresearch.com/docs/guides/daily-briefing-bot

Useful implementation points:

- Cron jobs run in fresh sessions, so prompts must be self-contained.
- Manual test first, then schedule.
- Delivery can be Telegram or local output under `~/.hermes/cron/output/`.
- The example searches latest AI agent and open-source LLM news, summarizes top stories, includes URLs, and reports source count.
- The guide mentions Firecrawl, but our MVP does not need it because Tavily is available and Hermes `web_search` exists as fallback.

### Existing digest projects worth borrowing from

1. giftedunicorn/ai-news-bot
   URL: https://github.com/giftedunicorn/ai-news-bot
   Useful bits:
   - RSS plus optional web search.
   - LLM summary.
   - HTML email templates.
   - Multi-channel delivery including Telegram, Slack, Discord, email.
   Borrow: HTML rendering pattern and feed-first approach.
   Avoid: extra provider complexity and email delivery for MVP.

2. duanyytop/agents-radar
   URL: https://github.com/duanyytop/agents-radar
   Useful bits:
   - Daily AI ecosystem digest from GitHub, ArXiv, HN, Hugging Face, Product Hunt, Dev.to, Lobsters, official OpenAI/Anthropic sites.
   - Writes Markdown files under dated folders.
   - Publishes issues and has a static web UI for history.
   Borrow: source taxonomy, dated archive folders, static browseable history later.
   Avoid: full bilingual web UI and weekly/monthly reports for MVP.

3. hoangsonww/AI-News-Briefing
   URL: https://github.com/hoangsonww/AI-News-Briefing
   Useful bits:
   - Multi-agent research lanes.
   - Publishes to Notion, Obsidian, Teams, Slack.
   - Supports custom deep-brief prompts.
   Borrow: structured prompt lanes and explicit topic coverage.
   Avoid: Notion/Teams/Slack plumbing and multi-agent complexity on day one.

4. nickzren/ai-news-agent
   URL: https://github.com/nickzren/ai-news-agent
   Useful bits:
   - Lightweight RSS candidate collector.
   - Candidate-only JSON stage before final LLM decision.
   - Duplicate guard before publishing.
   Borrow: collect candidates first, then ask the LLM to cluster, rank, drop noise.
   Avoid: GitHub Issue delivery for MVP.

5. draco-agent/tech-news-digest
   URL: https://github.com/draco-agent/tech-news-digest
   Useful bits:
   - Large curated source list.
   - Quality scoring, dedupe, topic grouping.
   - Tavily or Brave Search for freshness.
   Borrow: scoring heuristics and source overlays.
   Avoid: 168-source sprawl, PDF/email/Discord outputs for MVP.

## Recommended MVP architecture

Pipeline:

1. Collect candidates
   - Tavily CLI search, with `TAVILY_API_KEY` loaded from `~/.hermes/.env`.
   - Hermes `web_search` fallback if Tavily fails.
   - Optional curated RSS/feed list once first manual run proves value.

2. Normalise candidates
   - title
   - URL
   - source
   - published date if available
   - topic lane
   - short raw snippet

3. Dedupe and score
   - Drop duplicates by URL and near-identical headline.
   - Score higher for official source, GitHub release/issue, technical impact, workflow impact, direct relevance to Sahil/KENSEI.
   - Score lower for funding fluff, generic AI hype, listicles, thin reposts.

4. Summarise
   - Max 5 stories.
   - Each story gets headline, why it matters, source link.
   - Include one practical recommendation for Sahil's setup or workflow.
   - Include what was searched and how many candidates were considered.

5. Archive
   - Save full Markdown runbook.
   - Save structured JSON candidates and selected items.
   - Generate static HTML from the selected digest if Option 3-lite is enabled.

6. Deliver
   - Telegram first, under 1,200 characters.
   - Local archive path second.
   - No public exposure.

## Topic lanes

Current lanes after noise hardening:

A. AI News
- Model and provider releases.
- Official changelogs, launch posts, availability changes, model catalogue updates.
- Examples: Gemini, DeepSeek, Kimi, Qwen, GLM, Ollama Cloud, OpenAI, Anthropic.

B. Tool News
- Devtools, integrations, MCP, LangGraph, memory tooling, GitHub releases.
- GitHub root repos only qualify when there is a clear release, version, tag, changelog, or update signal.
- Hacker News can contribute popularity signals, especially points and comments.

C. MyTool
- Hermes, Claude Code, OpenAI Codex, OpenClaw, OpenCode, KENSEI operations.
- Prioritise items that could change cron prompts, skills, model routing, coding-agent workflow, or daily ops.

The digest is daily news and content, not generic research. The right shape is:

- New OpenClaw/Hermes Workspace releases.
- Gemini/DeepSeek/Ollama Cloud model availability.
- Claude Code, Codex, OpenCode, OpenClaw updates.
- LangGraph/MCP/memory tooling releases.
- Popular GitHub repos only when there is a real release, version, changelog, or update signal.

Curated RSS sources now supplement Tavily without new packages:

- Reddit RSS: r/LocalLLaMA, r/ClaudeAI, r/MachineLearning, r/vibecoding, r/Hermes, r/OpenClaw. Reddit requests must send a User-Agent.
- Hacker News RSS: `https://hnrss.org/frontpage` and `https://news.ycombinator.com/rss`.
- Reddit is contributory, not primary. It is scored below official domains and Hacker News.
- Popularity signals from RSS descriptions, such as points and comments, can lift an item but do not override official-source relevance.

Explicitly block or deprioritise:

- PR wires and generic announcement syndication, including openPR and Morningstar carries, unless directly relevant to the KENSEI stack.
- SEO comparison pages, pricing pages, alternatives pages, and affiliate-style `vs` content.
- Generic docs or static user-story pages
- Old guides/tutorials
- Generic funding announcements
- Enterprise AI press releases
- Consumer chatbot drama unless it affects tooling
- Crypto unless directly connected to AI agents or infra
- Random GitHub root repos with only "Latest commit History" noise
- Meme posts and low-signal social filler

## Output format, Telegram

```text
🗞️ Daily AI/Agent News, Saturday 2 May
Window: day or day + week fallback

A. AI News
1. Headline - why it matters. Source

B. Tool News
2. Headline - why it matters. Source

C. MyTool
3. Headline - why it matters. Source

🛠️ Practical recommendation
One concrete thing KENSEI/Sahil should do next.

📊 Searched: Tavily/RSS, N candidates, M sources
📎 Full brief attached below
MEDIA:/absolute/path/to/research-digest.html
```

Rules:

- Keep Telegram under 1,200 characters.
- Use source links, not vague citations.
- Attach the full HTML brief with a `MEDIA:/absolute/path` line for Telegram cron delivery.
- Deliver to `telegram:-1003922682700:23`, the Research topic in Kensei Workspace.
- No markdown tables in Telegram.
- No long commentary.
- If fewer than the minimum useful signal count survive after fallback, send `[SILENT]`, not filler.

## Static HTML artifact, Option 3-lite

Bring this forward only as a renderer, not a product surface.

MVP HTML should be:

- One standalone `.html` file per brief.
- Dark, card-based, mobile-readable.
- Generated from the same JSON/Markdown data as Telegram.
- Stored locally, probably under `~/.hermes/runbooks/research-digest/YYYY-MM-DD.html`.
- Linked from Telegram by path only, not exposed publicly.

Do not build yet:

- Public hosting
- Login/auth
- Searchable history UI
- Workspace panel
- Database-backed archive

## Proposed files

- `~/.hermes/scripts/research_digest_collect.py`
- `~/.hermes/scripts/research_digest_render.py`
- `~/.hermes/runbooks/research-digest/YYYY-MM-DD.json`
- `~/.hermes/runbooks/research-digest/YYYY-MM-DD.md`
- `~/.hermes/runbooks/research-digest/YYYY-MM-DD.html`

Repo docs:

- `/home/kensei/repos/KenseiAgent/docs/ResearchDigestPlan.md`

## Manual run prompt shape

The cron prompt must be self-contained:

```text
Create Sahil's KENSEI Research Digest for today.

Audience: Sahil Saghir, product leader and indie app builder, running Hermes Agent on a VPS.
Focus areas: Hermes Agent, personal AI agents, Claude Code, OpenAI Codex, Ollama Cloud, Kimi, Qwen, GLM, AI devtools, agent frameworks, and workflow improvements relevant to KENSEI.

Use the provided research collection output if available. If not, search the web for recent high-signal updates from the last 24-72 hours.
Select max 5 stories. For each, include headline, source URL, and why it matters. Ignore generic funding, vague AI hype, and duplicate reposts.
End with one practical recommendation for Sahil or KENSEI.
Output a concise Telegram brief first. Also save a full Markdown archive and, if possible, a static HTML artifact.
```

## Build sequence

1. Manual Tavily-backed collector run.
2. Manual digest synthesis from collected candidates.
3. Render Markdown and static HTML artifact.
4. Send test Telegram summary manually.
5. If useful, schedule daily cron at the agreed Research Digest slot.
6. Only after one scheduled run succeeds, consider a morning aggregator.

## Current Tavily state

Verified Saturday 2026-05-02 22:26 BST:

- `TAVILY_API_KEY` exists in `~/.hermes/.env`.
- `tvly search` authenticates successfully when the env file is loaded into the subprocess.
- The normal shell process does not currently expose `TAVILY_API_KEY`, so scripts must explicitly load it or the service environment must be updated.
