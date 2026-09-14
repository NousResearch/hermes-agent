# KENSEI Multi-Profile Implementation Checklist

Owner: Sahil Saghir
Operator: KENSEI
Created: 2026-05-03 23:15 BST
Status: Profiles created and SOUL.md prompts installed on 2026-05-03
Canonical companion: `/home/kensei/repos/KenseiAgent/docs/MultiProfileOperatingModel.md`

This checklist turns the agreed multi-profile operating model into executable steps. Profile creation is now complete. It still stops before gateway changes, model/provider switching, pilot dispatch, recurring jobs, or external integrations.

---

## 1. Locked decisions

- Keep boring functional profile names.
- Add aliases later when Sahil chooses nicknames.
- Keep `default` as KENSEI.
- General Assistant uses sensitive/admin tools only when task-scoped, including mailbox, calendar, booking, and browser/admin access.
- Research Lead owns all research: AI, agents, social media, Reddit/X scraping, web scraping, technical research, product research, validation, and general/semi-specialised investigations.
- Coding Lead may edit files and run tests. Commits only when explicitly asked. No push without approval.
- Content Lead drafts only for now. Posting/scheduling is a future approval-gated phase.
- Multi-platform posting solution must be free or near-free first. Paid tools must be flagged before recommendation.
- Knowledge Librarian may write to a defined KENSEI area in Obsidian. No git push unless approved or handled by a defined sync job.
- Ops Lead diagnoses freely and proposes fixes. Restarts, destructive actions, public exposure, and security-sensitive changes need approval.
- Ops Lead owns security, DevOps, performance, service health, monitoring, backups, and upkeep.
- First pilot is research -> content -> knowledge.
- Second pilot is a small content drafting pipeline using the same flow, so every profile boundary gets exercised.

---

## 2. Validated Hermes facts

Sources checked:

- Hermes Profiles docs: `https://hermes-agent.nousresearch.com/docs/user-guide/profiles`
- Hermes Profile Commands reference: `https://hermes-agent.nousresearch.com/docs/reference/profile-commands`
- Hermes Kanban tutorial: `https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban-tutorial`
- Dev.to Hermes deep dive: `https://dev.to/truongpx396/hermes-agent-deep-dive-build-your-own-guide-1pcc`
- Live CLI: `hermes kanban --help`, `hermes kanban create --help`, `hermes kanban dispatch --help`

Confirmed facts:

- Profiles are isolated Hermes homes under `~/.hermes/profiles/<name>/`.
- Each profile gets its own `config.yaml`, `.env`, `SOUL.md`, memory, sessions, skills, cron jobs, and state.
- A profile is not a filesystem sandbox. It still runs as the same OS user unless restricted by config and working directory discipline.
- Workspaces are per-task execution locations: `scratch`, `worktree`, or `dir:<path>`.
- `scratch` is the safe default for research, content, and admin tasks.
- `worktree` is the correct default for coding tasks that edit repositories.
- `dir:<path>` is correct when a task must work in a canonical location like the Obsidian vault or KENSEI docs repo.
- Kanban task completion supports `summary` and metadata, which is the correct handoff mechanism between profiles.
- The dashboard and CLI share the same Kanban DB.
- The dispatcher runs through the gateway; `hermes kanban daemon` is deprecated.

Conclusion: our proposed operating model is directionally correct. The biggest risk is not the architecture, it is over-splitting memory/tool access and forgetting that profiles isolate Hermes state, not OS permissions.

---

## 3. Pre-flight checks

Run these before creating profiles. Avoid `hermes status --all` during routine docs/checklist capture because it can print configured API key material into logs.

```bash
hermes profile list
hermes gateway status
hermes kanban stats
hermes kanban assignees
hermes kanban dispatch --dry-run --json
curl -fsS http://127.0.0.1:9119/api/status
curl -fsS http://127.0.0.1:3002/api/ping
```

Expected:

- `default` profile exists and remains active KENSEI.
- Any existing specialist profiles are noted before new creation.
- Kanban commands run cleanly.
- Dashboard and Workspace respond locally.
- No gateway or MCP instability before profile rollout.

Do not continue if gateway, Kanban, dashboard, or Workspace are already unhealthy. Ops Lead should diagnose first.

---

## 4. Profile creation sequence

Use CLI profile creation, not ad-hoc directory copying.

Recommended pattern:

```bash
hermes profile create gojo --clone
hermes profile create remii --clone
hermes profile create octacon --clone
hermes profile create ceecee --clone
hermes profile create light --clone
hermes profile create wesker --clone
```

Why `--clone`:

- It copies core config, `.env`, and `SOUL.md` so model/auth plumbing works immediately.
- It avoids copying sessions and memory from KENSEI.
- It keeps each specialist's learning loop clean from day one.

Do not use `--clone-all` for these profiles. That would copy KENSEI's sessions, state, cron jobs, and memories into specialists. That's memory pollution dressed up as convenience.

After creation:

```bash
hermes profile list
hermes profile show gojo
hermes profile show remii
hermes profile show octacon
hermes profile show ceecee
hermes profile show light
hermes profile show wesker
hermes kanban assignees
```

---

## 5. Alias plan

Keep functional names as canonical profile IDs.

Aliases can be added later:

```bash
hermes profile alias remii --name <nickname>
```

Rules:

- Alias can be fun.
- Profile ID stays boring.
- Docs, Kanban assignees, cron prompts, and runbooks use canonical profile IDs.
- Nicknames are user-interface sugar only.

---

## 6. Model policy

Initial model policy:

| Profile | Provider | Model | Notes |
|---|---|---|---|
| `default` | `openai-codex` | `gpt-5.5` | Temporary orchestrator while trial lasts. Later switch deliberately. |
| `gojo` | `ollama-cloud` | `kimi-k2.6:cloud` | General admin, booking, and secretary execution. |
| `remii` | `ollama-cloud` | `kimi-k2.6:cloud` | Upgrade per task only if research quality is weak. |
| `octacon` | `ollama-cloud` | `qwen3-coder:480b` | Live discovery confirmed availability on 2026-05-03. |
| `ceecee` | `ollama-cloud` | `kimi-k2.6:cloud` | Brand voice skills matter more than raw model size. |
| `light` | `ollama-cloud` | `kimi-k2.6:cloud` | Documentation and synthesis. |
| `wesker` | `ollama-cloud` | `kimi-k2.6:cloud` | Diagnosis, runbooks, monitoring. |

Before setting Qwen:

```bash
cd ~/.hermes/hermes-agent
./venv/bin/python - <<'PY'
from hermes_cli.models import fetch_ollama_cloud_models
for m in fetch_ollama_cloud_models(force_refresh=True):
    if any(x in m.lower() for x in ['kimi','qwen','glm']):
        print(m)
PY
```

Then set profile-specific model/provider only after confirming the IDs.

Implementation status on 2026-05-04:

- [x] `gojo` set to `ollama-cloud/kimi-k2.6:cloud`.
- [x] `remii` set to `ollama-cloud/kimi-k2.6:cloud`.
- [x] `octacon` set to `ollama-cloud/qwen3-coder:480b`.
- [x] `ceecee` set to `ollama-cloud/kimi-k2.6:cloud`.
- [x] `light` set to `ollama-cloud/kimi-k2.6:cloud`.
- [x] `wesker` set to `ollama-cloud/kimi-k2.6:cloud`.
- [x] One smoke prompt passed for every specialist profile.

Critical Kanban fix found during Pilot A:

- `hermes profile create --clone` gives each profile its own `HERMES_HOME`, so Kanban workers initially looked at profile-local `kanban.db` files and could not see tasks created by default KENSEI.
- Fixed by backing up profile-local `kanban.db` files where present and symlinking each specialist profile's `kanban.db` to `/home/kensei/.hermes/kanban.db`.
- Also symlinked each profile's `kanban/` directory to `/home/kensei/.hermes/kanban` so scratch workspaces stay shared.
- Verified with the Pilot A chain: `remii -> ceecee -> light` completed through the shared board.

---

## 7. Prompt files to draft before creation hardening

Create these as repo docs first, then copy into each profile's `SOUL.md` only after review.

```text
docs/profile-prompts/kensei-orchestrator.md
docs/profile-prompts/gojo.md
docs/profile-prompts/remii.md
docs/profile-prompts/octacon.md
docs/profile-prompts/ceecee.md
docs/profile-prompts/light.md
docs/profile-prompts/wesker.md
```

Each prompt must include:

- Identity and scope.
- Non-goals.
- Default tools.
- Task-scoped tools.
- Approval gates.
- Workspace default.
- Handoff metadata schema.
- Escalation rules.
- What to write to memory versus Obsidian.
- What “done” means.

---

## 8. Tool and skill policy

Do not load every tool into every profile. That kills prompt economy and increases risk.

| Profile | Default toolsets | Task-scoped toolsets | Always-relevant skills |
|---|---|---|---|
| `default` | kanban, skills, memory, session search, web, file, terminal as needed | Gmail/Outlook, GitHub, social posting | hermes-agent, hermes-build-plan-validation, kanban-orchestrator |
| `gojo` | memory, session search, clarify, web, file | Gmail/Outlook, calendar, booking platforms, shopping/browser, voice/calls | mailbox-agent, google-workspace when task assigned |
| `remii` | web, search, browser when needed, file, skills | X/Reddit/social APIs, scraping tools, Tavily/Firecrawl/Exa if configured | tavily-dynamic-search, blogwatcher, arxiv, polymarket, ai-seo, programmatic-seo, seo |
| `octacon` | terminal, file, git/GitHub, delegation, skills | browser, MCPs, external deploy tools | systematic-debugging, test-driven-development, github-pr-workflow, requesting-code-review |
| `ceecee` | file, web, memory/session search, skills | social posting APIs, Postiz/direct APIs later | brand-voices, sahil-linkedin-voice, sahil-twitter-voice, plenishd-voice, matchdaymaestro-voice, avoid-ai-writing, humanizer, content-strategy, social-content |
| `light` | file, search, Obsidian, session search | git sync | obsidian, ocr-and-documents, humanizer |
| `wesker` | terminal, file, web, skills | MCP admin, cronjob, system services | hermes-agent, hermes-cron-operations, native-mcp, google-workspace-mcp, code-security, clawsec-suite, 1password |

Sensitive tools are task-scoped unless the profile's whole purpose requires them.

---

## 9. Workspace defaults

Based on Hermes docs and live CLI, use this:

| Profile | Default workspace | Reason |
|---|---|---|
| `gojo` | `scratch` | Admin outputs should be isolated unless assigned to a specific folder. |
| `remii` | `scratch` | Research should produce clean summaries and source lists, not random files in project dirs. |
| `ceecee` | `scratch` | Drafts can be handed off as task metadata or saved deliberately. |
| `octacon` | `worktree` | Code edits need repo isolation and clean diffs. |
| `light` | `dir:/home/kensei/vaults/obsidian-master` only when writing notes, otherwise `scratch` | Vault writes must be explicit. |
| `wesker` | `scratch` for diagnosis, `dir:/home/kensei/repos/KenseiAgent` only for KENSEI docs/scripts tasks | Keeps ops from spraying files into Hermes state. |
| `default` | task-specific | KENSEI should orchestrate, not accumulate artifacts. |

Operational rule:

- Use `scratch` unless there is a named reason not to.
- Use `dir:<path>` only for canonical docs/vault/repo tasks.
- Use `worktree` for code changes.
- Never use `~/.hermes/` as a workspace unless Sahil explicitly asks for Hermes state modification.

---

## 10. Obsidian pattern

Canonical vault path:

```text
/home/kensei/vaults/obsidian-master
```

Create a KENSEI area:

```text
/home/kensei/vaults/obsidian-master/KENSEI/
/home/kensei/vaults/obsidian-master/KENSEI/Operating Model.md
/home/kensei/vaults/obsidian-master/KENSEI/Profile Implementation Checklist.md
/home/kensei/vaults/obsidian-master/KENSEI/Profile Pilot Log.md
```

Rules:

- Knowledge Librarian can write inside `KENSEI/` when assigned.
- Project notes live under project-specific folders later, not random root notes.
- No secrets in Obsidian.
- Git push only by approved command or defined sync job.
- Markdown files in repo remain operational runbooks. Obsidian notes are Sahil-readable knowledge.

---

## 11. Telegram routing and morning digests

### Cron delivery targets (updated 2026-05-04)

All four morning digests now route to the unified Daily Digest in Topic 1:

| Job | Time | Target |
|-----|------|--------|
| system-report-daily | 07:00 | `telegram:-1003922682700:1` |
| calendar-brief-daily | 07:30 | `telegram:-1003922682700:1` |
| mailbox-digest-daily | 08:00 | `telegram:-1003922682700:1` |
| research-digest-daily | 08:15 | `telegram:-1003922682700:1` |

Each digest prepends its section header (🖥️ 📅 📬 🔬). Dark-themed HTML morning-brief style preserved.

### Profile topic mapping (approved 2026-05-04)

| Topic | Purpose | Primary profile |
|-------|---------|-----------------|
| 1 | Daily Digest | All morning briefs |
| 19 | Content Drafts | ceecee |
| 20 | Build Log | octacon |
| 21 | Ops & Alerts | wesker |
| 22 | Research Deep Dives | remii |
| 23 | Archive / Low-priority | any |

### Routing principle

Specialist profiles never push directly to Telegram. They complete Kanban tasks. KENSEI relays relevant output to the correct topic. Cron jobs with explicit delivery targets are the only exception. DM stays for critical alerts and memory curator.

---

## 12. Multi-platform posting research track

This is deliberately separate from Content Lead v1.

Requirement:

- Free or near-free first.
- Multi-platform where possible.
- Approval-gated before publish/schedule.
- Works for LinkedIn, X, Instagram/Facebook where APIs allow.
- Avoid brittle browser automation unless no proper API route exists.

Research questions:

1. What can be done through free official APIs today?
2. Which platforms require app review or business verification?
3. Can Postiz self-hosting solve the problem without subscription cost?
4. Which Postiz MCP/API integrations are maintained enough to trust?
5. What is the cheapest viable path for approval-gated scheduling?
6. Which platforms should remain manual copy/paste because automation is cursed or paid?

Output required from Research Lead later:

```json
{
  "options": [],
  "costs": [],
  "platform_coverage": [],
  "api_limitations": [],
  "setup_complexity": "low|medium|high",
  "risk": [],
  "recommendation": ""
}
```

---

## 13. Pilot A: research -> content -> knowledge

Goal: prove profile routing, handoff metadata, dashboard visibility, and Obsidian capture without touching dangerous systems.

Suggested topic:

```text
Research free/near-free multi-platform social posting options for Sahil's content workflow, then create one LinkedIn/X content draft explaining the decision, then capture the decision trail in Obsidian.
```

Task graph:

```bash
RESEARCH=$(hermes kanban create "Research free multi-platform posting options" \
  --assignee remii \
  --tenant kensei-profile-pilot \
  --workspace scratch \
  --priority 2 \
  --body "Compare free/near-free multi-platform posting options for Sahil. Cover Postiz self-hosting, official APIs, manual copy/paste, and approval-gated scheduling. Return sources, cost, platform coverage, risks, and recommendation." \
  --skill tavily-dynamic-search \
  --json | jq -r .id)

CONTENT=$(hermes kanban create "Draft content from posting-options research" \
  --assignee ceecee \
  --tenant kensei-profile-pilot \
  --workspace scratch \
  --priority 2 \
  --parent "$RESEARCH" \
  --body "Create one short LinkedIn draft and one X thread draft from the research. Draft only. No posting, no scheduling." \
  --skill sahil-linkedin-voice \
  --skill sahil-twitter-voice \
  --json | jq -r .id)

KNOWLEDGE=$(hermes kanban create "Capture posting-options pilot in Obsidian" \
  --assignee light \
  --tenant kensei-profile-pilot \
  --workspace dir:/home/kensei/vaults/obsidian-master \
  --priority 2 \
  --parent "$CONTENT" \
  --body "Create or update KENSEI/Profile Pilot Log.md with the research decision, content outputs, handoff quality, and next improvements. Do not git push." \
  --skill obsidian \
  --json | jq -r .id)
```

Dispatch dry-run first:

```bash
hermes kanban dispatch --dry-run --max 3 --json
```

Then run one dispatch pass or use gateway dispatcher:

```bash
hermes kanban dispatch --max 3 --json
```

Verification:

```bash
hermes kanban show "$RESEARCH"
hermes kanban runs "$RESEARCH"
hermes kanban show "$CONTENT"
hermes kanban runs "$CONTENT"
hermes kanban show "$KNOWLEDGE"
hermes kanban runs "$KNOWLEDGE"
```

---

## 14. Pilot B: small content drafting pipeline

Goal: prove the repeatable content pipeline after the research/content/knowledge pilot.

Suggested pipeline:

1. KENSEI creates brief.
2. Research Lead gathers 3-5 fresh source points.
3. Content Lead drafts:
   - LinkedIn post
   - X thread
   - optional Instagram caption
4. Knowledge Librarian saves:
   - source links
   - final approved drafts
   - lessons learned
5. General Assistant can optionally prepare a send/schedule checklist, but not post.
6. Ops Lead checks whether the pipeline generated any security, credential, or automation concerns.
7. KENSEI reviews quality and decides whether posting automation research should move into build phase.

Success criteria:

- Every profile has a clear reason to exist.
- Handoffs are visible without digging through raw logs.
- No profile needed hidden context from another profile's memory.
- Obsidian note is useful to Sahil, not just machine noise.
- Dashboard/Kanban made the workflow easier, not heavier.

---

## 15. Stop/go gate before building Mission Center

Do not resume custom Mission Center until after both pilots.

Build custom UI only if at least one of these is true:

- Kanban dashboard cannot show the multi-profile state Sahil needs.
- Handoff metadata is hard to inspect.
- Sahil cannot quickly see “what needs my approval”.
- Mobile access is poor even through Tailscale/private access.
- Profile aliases and task lanes are too clunky in practice.

If custom UI is needed, it should be a Sahil-facing overlay on top of Kanban, not a replacement for Kanban.
