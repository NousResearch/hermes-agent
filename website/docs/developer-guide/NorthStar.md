# KENSEI — Single Source of Truth

Owner: Sahil Saghir
Operator: KENSEI, Sahil's Hermes Agent on the VPS
Canonical file: `/home/kensei/repos/KenseiAgent/docs/NorthStar.md`
Created: 2026-04-29
Last verified: Sunday 2026-05-03 20:35 BST
Status: MVP core complete, moving to Next lane

This file is the single source of truth for the KENSEI North Star plan. It replaces the old split between `NorthStar.md`, `NorthStar-Review-20260501.md`, and the duplicate root-level copies under `/home/kensei/`.

If another note, memory, review, or chat conflicts with this file, this file wins.

---

## 1. North Star

KENSEI is Sahil's persistent, self-improving personal AI agent built on Hermes Agent. It runs on the Linux VPS and is accessible through CLI, Telegram, and private browser access via the built-in Hermes dashboard plus Hermes Workspace.

The near-term product is not a science project. The MVP is daily usefulness:

1. A useful mailbox digest across all 7 connected inboxes.
2. A useful research digest with sources and one practical recommendation.
3. A private Command Center that works from browser and phone.
4. Text-only content drafting in the right brand voice.
5. Reliable memory/session recall.
6. No operational fires.

Everything else is post-MVP unless it directly unblocks one of those six outcomes.

---

## 2. Current verified system state

Verified from live commands on Sunday 2026-05-03, with gateway restart and Gmail re-auth completed at 20:11 BST.

### Core runtime

- Hermes Agent: `v0.12.0`, up to date.
- Main model: `openai-codex` / `gpt-5.5` (14-day trial, ends ~12 May 2026).
- Delegation/default workers: `ollama-cloud` / `kimi-k2.6`.
- Approvals: `manual`.
- Cron approvals: `deny`.
- Memory provider: `mem0` plus Hermes session search.
- Memory budgets: `MEMORY.md` 2,200 chars, `USER.md` 4,000 chars.
- CLI toolset is lean and includes `no_mcp`, so normal CLI sessions do not load Google/Outlook MCP schemas by default.
- Tavily CLI is installed and `TAVILY_API_KEY` exists in `~/.hermes/.env`; direct shell sessions do not see it unless the env file is sourced.

### Services

All active and healthy:

- `hermes-gateway.service` — restarted 2026-05-03 19:45, clean state
- `hermes-dashboard.service`
- `hermes-workspace.service`
- `tailscaled.service`

Loopback-only listeners:

- Gateway: `127.0.0.1:8642`
- Hermes dashboard API: `127.0.0.1:9119`
- Hermes Workspace UI: `127.0.0.1:3002`

Dashboard and Workspace smoke checks pass:

- `http://127.0.0.1:9119/api/status`
- `http://127.0.0.1:3002/api/ping`
- `http://127.0.0.1:3002/api/connection-status`

### Connected accounts

Gmail via Google Workspace MCP — all three healthy (token age 0 days, verified 2026-05-03 20:30):

- `saghir.sahil@gmail.com` ✅
- `sahilsaghir.ss9@gmail.com` ✅
- `fusionfirststudios@gmail.com` ✅

Outlook via Softeria MS 365 MCP — all four connected:

- `sahil_ss9@hotmail.com`
- `sahil_saghir@hotmail.co.uk`
- `sahil_ss@outlook.com`
- `matchdaymaestro@outlook.com`

Configured MCP servers:

- `google_workspace`
- `outlook`

Latest auth audit, 2026-05-03 20:30 BST:

- All 3 Gmail accounts: token files present, age 0 days, status healthy.
- All 4 Outlook accounts: token cache present, connected.
- Gateway restart at 19:45 + OAuth state clear + re-auth for two accounts at 20:11 resolved all prior port 8000 / OAuth callback blocks.
- One clean `workspace-mcp` process, one clean `ms-365-mcp-server` process. No orphans.

Not configured yet:

- GitHub MCP
- Postiz MCP/API
- Composio

### Security baseline

- `~/.hermes/` mode: `700`
- `~/.hermes/config.yaml`: `600`
- `~/.hermes/.env`: `600`
- `~/.hermes/SOUL.md`: `600`
- `~/.hermes/memories/MEMORY.md`: `600`
- `~/.hermes/memories/USER.md`: `600`
- Telegram bot restricted to Sahil DM via `gateway.json` allowlist.
- Control plane stays private, loopback/Tailscale/SSH tunnel only.
- No public dashboard exposure.

### Cron jobs

Active cron jobs:

| Job | Schedule | Provider/model | Status | Relevance |
|---|---:|---|---|---|
| `kensei-heartbeat` | every 30m | `ollama-cloud` / `kimi-k2.6` | ok | Operational baseline |
| `system-report-daily` | 07:00 daily | `scripts/system_report.py` (Python) | live-tested 2026-05-03, clean output | Operational pulse |
| `calendar-brief-daily` | 07:30 daily | `ollama-cloud` / `kimi-k2.6` + Google Workspace MCP | live-tested, picked up May 6 events | Morning brief |
| `mailbox-digest-daily` | 08:00 daily | `ollama-cloud` / `kimi-k2.6` + `mailbox-agent` skill | all 7 accounts green, morning-brief format, HTML attachment, Inbox topic 19 | Morning brief |
| `research-digest-daily` | 08:15 daily | `ollama-cloud` / `kimi-k2.6` + `scripts/research_digest.py` | 10 items, day+7d fallback, HTML attachment, 19 unit tests pass, Research topic 23 | Morning brief |
| `memory-curator-run` | 02:00 daily | `ollama-cloud` / `kimi-k2.6` | ok (silent, no user delivery) | Memory maintenance |

Paused (absorbed by `system-report-daily`):

| Job | Status | Notes |
|---|---|---|
| `doctor-daily` | paused | replaced by system-report-daily |
| `token-health-check` | paused | replaced by system-report-daily |
| `Phase1 Gmail Verifier` | paused | replaced by system-report-daily |

### Morning digest lineup

```
07:00  System Report     → DMs (operational pulse: services, tokens, cron, disk, mem)
07:30  Calendar Brief    → Inbox topic 19 (today + next 7 days from Google Calendar)
08:00  Mailbox Digest    → Inbox topic 19 (action items, FYI, noise summary, all 7 inboxes)
08:15  Research Digest   → Research topic 23 (10 AI/agent/devtools signals with HTML brief)
```

All four digests generate dark-themed HTML artifacts with coloured section headers and attach them via MEDIA: tags.

### Mailbox digest reliability

Resolved 2026-05-03. The Gmail MCP port 8000 / OAuth callback blocks were caused by stale `workspace-mcp` process state and orphaned MCP processes. The fix sequence is now documented:

1. Stop gateway: `sudo systemctl stop hermes-gateway.service`
2. Kill any orphan MCP processes on port 8000: `sudo ss -tlnp | grep 8000` then `kill <PID>`
3. Clear OAuth state: write `{}` to `~/.google_workspace_mcp/credentials/oauth_states.json`
4. Only delete account token files if re-auth is actually needed (avoid this if possible)
5. Start gateway: `sudo systemctl start hermes-gateway.service`
6. Verify: `python3 ~/.hermes/scripts/token_health.py` (all accounts healthy, age 0 days)

The `mailbox-agent` skill and cron prompt were patched to:
- Fail closed on evidence conflicts (never claim green without fresh proof)
- Remove the full green account table pattern
- Generate an HTML artifact with coloured section headers
- Route to Inbox topic 19
- Use clean morning-brief format with no cron/log noise

### Research digest

Script-backed at `/home/kensei/repos/KenseiAgent/scripts/research_digest.py`. Uses Tavily search + official RSS feeds (OpenAI, Google AI, GitHub, Hugging Face) with deduplication and scoring. Filters out Reddit discussion sludge, static Ollama catalogue pages, GitHub root repos/PRs/issues, raw HuggingFace file links, stale RSS items, and press-release/SEO junk. Prefers current-day evidence with 7-day fallback. Generates JSON, Markdown, HTML, and Telegram text. 19 unit tests pass.

### Command Center

Hermes Dashboard (port 9119) and Hermes Workspace (port 3002) are functional and accessible via Tailscale from Sahil's laptop and phone. Sahil wants a custom Command Center that feels personal — moved to the custom Command Center lane (Next). Current Workspace remains the operational UI.

---

## 3. Canonical MVP checklist

Legend:

- ✅ Done and verified
- 🔴 Not done
- ⏸️ Deliberately deferred

### MVP outcome checklist

| Outcome | Status | Done when |
|---|---:|---|
| System Report | ✅ | `scripts/system_report.py` consolidates doctor + token health + service/cron status, 07:00 to DMs |
| Calendar Brief | ✅ | Google Workspace MCP reads calendar, today + 7 days, 07:30 to Inbox topic 19 |
| Mailbox digest (all 7 inboxes) | ✅ | All accounts green 2026-05-03 20:30; morning-brief format with HTML attachment |
| Research digest | ✅ | Script-backed, 10 items, day+7d fallback, HTML attachment, 19 tests pass |
| HTML brief artifacts | ✅ | All four digests generate dark-themed HTML with coloured section headers |
| Command Center browser access | ✅ | Dashboard and Workspace reachable locally |
| Command Center laptop/phone access | ✅ | Accessible via Tailscale; custom build deferred to Next lane |
| Content drafting | 🔴 | Three text drafts that are edit-worthy, not rewrite-worthy |
| Memory/session recall | ✅ | Session search and Mem0 recall work in normal use |
| Operational stability | ✅ | 6 active cron jobs, 3 paused; all services green; Gmail auth resolved |

### Foundation checklist

| Area | Status | Notes |
|---|---:|---|
| Hermes installed and current | ✅ | `v0.12.0`, up to date |
| Main provider configured | ✅ | `openai-codex` / `gpt-5.5` (trial ends ~12 May 2026) |
| Worker/delegation provider configured | ✅ | `ollama-cloud` / `kimi-k2.6` |
| Manual approval gates | ✅ | `approvals.mode: manual`, `cron_mode: deny` |
| Memory provider | ✅ | Mem0 plus session search |
| CLI context bloat reduced | ✅ | Explicit CLI toolsets plus `no_mcp` |
| Gmail MCP | ✅ | 3 accounts, all healthy (age 0 days) |
| Outlook MCP | ✅ | 4 accounts connected |
| GitHub token | ✅ | Present, GitHub MCP deferred |
| GitHub MCP | ⏸️ | Post-MVP unless code workflow blocks MVP |
| Postiz | ⏸️ | Post-MVP, drafting first |
| Composio | ⏸️ | Future only if a specific app need appears |
| Firecrawl/Exa | ⏸️ | Tavily/web search is enough for MVP |
| Hindsight | ⏸️ | Mem0/session search is enough for MVP |
| Dashboard API service | ✅ | `127.0.0.1:9119` |
| Workspace UI service | ✅ | `127.0.0.1:3002` |
| Tailscale installed/running | ✅ | Access validated from Sahil's devices |
| Local backups/pre-update backups | 🔴 | Daily local backup not verified |
| Off-VPS backup | ⏸️ | Post-MVP |
| Error-log alerting | 🔴 | Add after digest jobs are proven stable |
| Workspace repo update debt | 🟡 | Local patches exist, repo behind upstream |

---

## 4. Roadmap

### Now

This is the current execution lane.

1. **Content drafting** — build text-only content drafting pipeline.
   - MatchdayMaestro voice
   - Plenishd voice
   - Sahil LinkedIn voice
   - Sahil Twitter voice
   - Output must be edit-worthy, not rewrite-worthy.
   - Use the `brand-voices` skill and individual voice skills for tone.

2. **Verify scheduled runs** — monitor tomorrow's 07:00-08:15 morning lineup.
   - Confirm all four digests deliver on schedule.
   - Check system-report-daily runs cleanly from cron (not just manual trigger).
   - Confirm mailbox digest reads all 7 accounts in scheduled mode.

3. **Add error-log alerting** — basic Telegram ping if any cron goes red.

4. **Verify daily local backup** — create or confirm a simple daily local backup cron.

### Next

Start after the Now lane is boring.

1. **Multi-profile Kanban operating model**, configure KENSEI as orchestrator over specialist Hermes profiles using the native Hermes Kanban layer.
   - Canonical plan: `/home/kensei/repos/KenseiAgent/docs/MultiProfileOperatingModel.md`.
   - Target roster: KENSEI Orchestrator, General Assistant, Research Lead, Coding Lead, Content Lead, Knowledge Librarian, Ops Lead.
   - Use Kanban tasks, parent/child dependencies, comments, shared workspaces, and structured completion metadata as the cross-profile coordination layer.
   - Use Obsidian as the canonical human-readable knowledge layer, not a duplicate project dossier system.
   - Run one real end-to-end pilot before creating custom UI.

2. **Custom Command Center**, paused until Hermes Kanban + dashboard have been exercised.
   - If resumed, build it as a fresh, lightweight KENSEI overlay, not a revival of the scrapped dashboard codebase.
   - Pick best bits from Hermes dashboard, Hermes Workspace, and OpenClaw-style control surfaces only after identifying real gaps.
   - First-class views should be decisions needed, active blockers, profile lanes, latest handoffs, and project/Obsidian links.

3. **Simple operational runbooks**
   - How to check dashboard/workspace
   - How to check cron health
   - How to re-auth Gmail/Outlook
   - Gateway restart + MCP cleanup sequence (already documented above)

4. **Clean Workspace update debt** without losing local service patches.

### Future

Useful, but only after daily value is proven.

1. Polished brief dashboard: HTML history, latest-digest view, Workspace integration
2. GitHub MCP for issue/PR workflows
3. Postiz integration for approved social scheduling
4. Image generation for content
5. Carousel generation
6. Video generation pipeline
7. Per-subagent Hermes profiles if single-profile becomes constraining
8. Better app repo context via project `AGENTS.md` files
9. Weekly Hermes Atlas / skills ecosystem diff
10. Skill usage tracking and quarterly pruning
11. Restore test and disaster recovery drill

### Deferred

Explicitly not MVP.

1. Hindsight memory migration
2. Composio Connect
3. Firecrawl/Exa unless web search becomes inadequate
4. Voice calls, beyond voice memo transcription
5. Public Command Center exposure
6. Email gateway as a messaging interface
7. Discord expansion
8. Wondel/Felo/meta-skill bundles
9. Trading bot or trading automation
10. Autonomous social posting without approval
11. Multi-agent swarm experiments
12. Off-VPS backup if the current focus is daily value rather than disaster recovery

---

## 5. Operating rules

1. Verify live state before updating this file.
2. Keep this file short enough to be useful. Do not turn it into an encyclopedia.
3. If a checklist item has not been verified, mark it 🔴, not ✅.
4. Do not install or integrate Postiz, GitHub MCP, Hindsight, Composio, Firecrawl, Exa, Wondel, or profile architecture while content drafting is still unfinished.
5. Destructive actions, email sending, mailbox mutation, social posting, or public exposure require explicit Sahil approval.
6. Keep dashboards private by default through loopback, SSH tunnel, or Tailscale.
7. Use `hermes dashboard`, not stale `hermes web` language.
8. Skills Hub is CLI-driven. Do not invent YAML blocks like `skills_hub`.
9. MCP tool calls are prefixed as `mcp_<server>_<tool>`.
10. Update this file whenever state changes meaningfully.
11. **New: Gmail MCP fix sequence.** If Gmail reads fail with port 8000 / OAuth callback errors, the fix is: stop gateway, kill orphan MCP on port 8000, clear `oauth_states.json`, start gateway. Do NOT delete account token files unless re-auth is genuinely needed. See section 2 for the full sequence.

---

## 6. Source evaluation and merge decision

Evaluated files on 2026-05-02:

- `/home/kensei/repos/KenseiAgent/docs/NorthStar.md`
- `/home/kensei/repos/KenseiAgent/docs/NorthStar-Review-20260501.md`
- `/home/kensei/NorthStar.md`
- `/home/kensei/NorthStar-Review-20260501.md`

Decision:

- Keep the repo path as canonical: `/home/kensei/repos/KenseiAgent/docs/NorthStar.md`.
- Merge the useful review insights into this file: MVP cut, stale-command warnings, false assumptions, and defer list.
- Remove duplicate root-level files and the separate review file to prevent drift.

Why:

- The repo copy was newer and included the CLI lean-down, Workspace service, cron provider fixes, and mailbox digest probe.
- The root copy was stale.
- The review doc was useful, but as a separate file it kept creating split-brain planning.
- The new roadmap format makes execution clearer: Now, Next, Future, Deferred.

---

## 7. Change log

### 2026-05-03, MVP core complete — System Report, Calendar Brief, Gmail fix, full document refresh

- Built `scripts/system_report.py` — consolidates `hermes doctor`, token health, system stats (disk/mem/uptime), service status, cron health, and Hermes version into one operational pulse. Dark-themed HTML with coloured section headers. Delivers to DMs at 07:00.
- Paused `doctor-daily`, `token-health-check`, and `Phase1 Gmail Verifier` — all absorbed by system-report-daily.
- Built Calendar Brief cron (07:30) — uses Google Workspace MCP to read primary Google Calendar, surfaces today + next 7 days, delivers to Inbox topic 19 with HTML attachment. Live-tested, picked up May 6 events.
- Extended research digest from 5 to 10 items. Fixed `--selected-limit` default and `telegram_text` slice to 10.
- Fixed mailbox digest HTML — added mandatory HTML template to `mailbox-agent` SKILL.md with distinct coloured section headers (red Action, blue Worth Knowing, grey Noise, cyan Health, yellow Next Move).
- Fixed doctor-daily to use bullet points instead of paragraph format.
- Resolved Gmail MCP reliability: gateway restart + OAuth state clear + re-auth for `sahilsaghir.ss9@gmail.com` and `fusionfirststudios@gmail.com` brought all 3 Gmail accounts to healthy (age 0 days).
- Documented Gmail MCP fix sequence in operating rules.
- Updated morning lineup: 07:00 System Report, 07:30 Calendar, 08:00 Mailbox, 08:15 Research. All four deliver via Telegram with HTML attachments.

### 2026-05-03, earlier — manual digest validation and evidence-gate reset

- Manually triggered token health, Gmail verifier, doctor, memory curator, mailbox digest, and research digest jobs.
- Confirmed mailbox digest output was not production-trusted — claimed all Gmail accounts healthy despite fresher verifier failures.
- Updated `mailbox-agent` and cron prompt to fail closed on evidence conflicts, remove full green account table, attach HTML, route to Inbox topic 19.
- Tightened research digest filtering: removed Reddit sludge, static Ollama pages, GitHub root repos/PRs/issues, raw HF file links, stale RSS; 19 tests pass.
- Reworked research digest Telegram output into clean morning-brief layout.

### 2026-05-02, mailbox auth cleanup

- Read-only auth audit across all 7 mailboxes.
- Initial Gmail failures caused by stale duplicate `workspace-mcp` processes and port 8000 ownership.
- Cleaned stale MCP state, cleared OAuth state, restarted gateway. All 7 accounts green. No browser re-auth needed.

### 2026-05-02, single source of truth consolidation

- Rewrote this file as the only canonical KENSEI North Star plan.
- Merged old review doc into checklist, roadmap, source evaluation, and deferred list.
- Marked mailbox partial, research not done, Command Center local done, laptop/phone not done.
- Explicitly deferred Postiz, GitHub MCP, Hindsight, Composio, Firecrawl/Exa, voice calls, public exposure.
