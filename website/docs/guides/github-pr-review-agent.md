---
sidebar_position: 10
title: "Tutorial: GitHub PR Review Agent"
description: "Build an automated AI code reviewer that monitors your repos, reviews pull requests, and delivers feedback — hands-free"
---

# Tutorial: Build a GitHub PR Review Agent

**The problem:** Your team opens PRs faster than you can review them. PRs sit for days waiting for eyeballs. Junior devs merge bugs because nobody had time to check. You spend your mornings catching up on diffs instead of building.

**The solution:** An AI agent that watches your repos around the clock, reviews every new PR for bugs, security issues, and code quality, and sends you a summary — so you only spend time on PRs that actually need human judgment.

**What you'll build:**

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│  Cron Timer  │────▶│  Hermes Agent │────▶│  GitHub API  │────▶│  Review to   │
│  (every 2h)  │     │  + gh CLI     │     │  (PR diffs)  │     │  Telegram/   │
│              │     │  + skill      │     │              │     │  Discord/    │
│              │     │  + memory     │     │              │     │  local file  │
└──────────────┘     └───────────────┘     └──────────────┘     └──────────────┘
```

This guide uses **cron jobs** to poll for PRs on a schedule — no server or public endpoint needed. Works behind NAT and firewalls.

:::tip Want real-time reviews instead?
If you have a public endpoint available, check out [Automated GitHub PR Comments with Webhooks](./webhook-github-pr-review.md) — GitHub pushes events to Hermes instantly when PRs are opened or updated.
:::

---

## Prerequisites

- **Hermes Agent installed** — see the [Installation guide](/docs/getting-started/installation)
- **Gateway running** for cron jobs:
  ```bash
  hermes gateway install   # Install as a service
  # or
  hermes gateway           # Run in foreground
  ```
- **GitHub CLI (`gh`) installed and authenticated**:
  ```bash
  # Install
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # Authenticate
  gh auth login
  ```
- **Messaging configured** (optional) — [Telegram](/docs/user-guide/messaging/telegram) or [Discord](/docs/user-guide/messaging/discord)

:::tip No messaging? No problem
Use `deliver: "local"` to save reviews to `~/.hermes/cron/output/`. Great for testing before wiring up notifications.
:::

---

## Step 1: Verify the Setup

Make sure Hermes can access GitHub. Start a chat:

```bash
hermes
```

Test with a simple command:

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

You should see a list of open PRs. If this works, you're ready.

---

## Step 2: Try a Manual Review

Still in the chat, ask Hermes to review a real PR:

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes will:
1. Execute `gh pr diff` to fetch the code changes
2. Read through the entire diff
3. Produce a structured review with specific findings

If you're happy with the quality, time to automate it.

---

## Step 3: Create a Review Skill

A skill gives Hermes consistent review guidelines that persist across sessions and cron runs. Without one, review quality varies.

```bash
mkdir -p ~/.hermes/skills/code-review
```

Create `~/.hermes/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

Verify it loaded — start `hermes` and you should see `code-review` in the skills list at startup.

---

## Step 4: Teach It Your Conventions

This is what makes the reviewer actually useful. Start a session and teach Hermes your team's standards:

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

These memories persist forever — the reviewer will enforce your conventions without being told each time.

---

## Step 5: Create the Automated Cron Job

Now wire it all together. Create a cron job that runs every 2 hours:

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

Verify it's scheduled:

```bash
hermes cron list
```

### Other useful schedules

| Schedule | When |
|----------|------|
| `0 */2 * * *` | Every 2 hours |
| `0 9,13,17 * * 1-5` | Three times a day, weekdays only |
| `0 9 * * 1` | Weekly Monday morning roundup |
| `30m` | Every 30 minutes (high-traffic repos) |

---

## Step 6: Run It On Demand

Don't want to wait for the schedule? Trigger it manually:

```bash
hermes cron run pr-review
```

Or from within a chat session:

```
/cron run pr-review
```

---

## Going Further

### Post Reviews Directly to GitHub

Instead of delivering to Telegram, have the agent comment on the PR itself:

Add this to your cron prompt:

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
Make sure `gh` has a token with `repo` scope. Reviews are posted as whoever `gh` is authenticated as.
:::

### Weekly PR Dashboard

Create a Monday morning overview of all your repos:

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### Multi-Repo Monitoring

Scale up by adding more repos to the prompt. The agent processes them sequentially — no extra setup needed.

---

## Troubleshooting

### "gh: command not found"
The gateway runs in a minimal environment. Ensure `gh` is in the system PATH and restart the gateway.

### Reviews are too generic
1. Add the `code-review` skill (Step 3)
2. Teach Hermes your conventions via memory (Step 4)
3. The more context it has about your stack, the better the reviews

### Cron job doesn't run
```bash
hermes gateway status    # Is the gateway running?
hermes cron list         # Is the job enabled?
```

### Rate limits
GitHub allows 5,000 API requests/hour for authenticated users. Each PR review uses ~3-5 requests (list + diff + optional comments). Even reviewing 100 PRs/day stays well within limits.

---

## What's Next?

- **[Webhook-Based PR Reviews](./webhook-github-pr-review.md)** — get instant reviews when PRs are opened (requires a public endpoint)
- **[Daily Briefing Bot](/docs/guides/daily-briefing-bot)** — combine PR reviews with your morning news digest
- **[Build a Plugin](/docs/guides/build-a-hermes-plugin)** — wrap the review logic into a shareable plugin
- **[Profiles](/docs/user-guide/profiles)** — run a dedicated reviewer profile with its own memory and config
- **[Fallback Providers](/docs/user-guide/features/fallback-providers)** — ensure reviews run even when one provider is down
