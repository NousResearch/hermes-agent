---
sidebar_position: 3
---

# Profile Distributions: Share a Whole Agent

A **profile distribution** packages a complete Hermes agent — personality, skills, cron jobs, MCP connections, config — as a git repository. Anyone with access to the repo can install the whole agent with one command, update it in place, and keep their own memories, sessions, and API keys untouched.

If a [profile](./profiles.md) is a local agent, a distribution is that agent made shareable.

## What this means

Before distributions, sharing a Hermes agent meant sending someone:

1. Your SOUL.md
2. A list of skills to install
3. Your config.yaml, minus the secrets
4. A description of which MCP servers you wired up
5. Any cron jobs you scheduled
6. Instructions for which env vars to set

…and hoping they assembled it correctly. Every version bump or bug fix meant repeating the handoff.

With distributions, all of that lives in one git repo:

```
my-research-agent/
├── distribution.yaml    # manifest: name, version, env-var requirements
├── SOUL.md              # the agent's personality / system prompt
├── config.yaml          # model, temperature, reasoning, tool defaults
├── skills/              # bundled skills that come with the agent
├── cron/                # scheduled tasks the agent runs
└── mcp.json             # MCP servers the agent connects to
```

Recipients run:

```bash
hermes profile install github.com/you/my-research-agent --alias
```

…and they now have the whole agent. They fill in their own API keys (`.env.EXAMPLE` → `.env`), and they can run `my-research-agent chat` or address it through Telegram / Discord / Slack / any gateway platform. When you push a new version, they run `hermes profile update my-research-agent` and pull your changes — their memories and sessions stay put.

## Why git?

We considered tarballs, HTTP archives, a custom format. None of them beat git:

- **Zero build step for authors.** Push to GitHub; consumers install. There's no "pack this, upload that, update the index" loop.
- **Tags, branches, and commits are already the versioning system.** A tag push does for us what "pack + upload a release" does for other tools.
- **Updates are a fetch.** Not a re-download of the whole archive.
- **Transparent.** Users can browse the repo, read diffs between versions, open issues against it, fork it to customize.
- **Private repos work for free.** SSH keys, `git credential` helpers, GitHub CLI stored credentials — whatever auth your terminal is already set up for applies transparently.
- **Reproducibility is a commit SHA.** The same thing pip and npm record.

The tradeoff: recipients need git installed. On any machine running Hermes in 2026, that's already true.

## When should you use a distribution?

Good fits:

- **You're sharing a specialized agent** — a compliance monitor, a code reviewer, a research assistant, a customer-support bot — with a team or with the community.
- **You're deploying the same agent to multiple machines** and don't want to copy files manually each time.
- **You're iterating on an agent** and want recipients to pick up new versions with one command.
- **You're building an agent as a product** — opinionated defaults, curated skills, tuned prompts — that other people should use as a starting point.

Not a fit:

- **You just want to back up a profile on your own machine.** Use [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — that's what those are for.
- **You want to share API keys alongside the agent.** `auth.json` and `.env` are deliberately excluded from distributions. Each installer brings their own credentials.
- **You want to share memories / sessions / conversation history.** Those are user data, not distribution content. Never shipped.

## The lifecycle: author to installer to update

Below is the full end-to-end flow. Pick the side you care about.

---

## For authors: publishing a distribution

### Step 1 — Start from a working profile

Build and refine the agent like any other profile:

```bash
hermes profile create research-bot
research-bot setup                    # configure model, API keys
# Edit ~/.hermes/profiles/research-bot/SOUL.md
# Install skills, wire up MCP servers, schedule cron jobs, etc.
research-bot chat                     # dogfood until it feels right
```

### Step 2 — Add a `distribution.yaml`

Create `~/.hermes/profiles/research-bot/distribution.yaml`:

```yaml
name: research-bot
version: 1.0.0
description: "Autonomous research assistant with arXiv and web tools"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# Tell installers which env vars the agent needs. These are checked against
# the installer's shell and existing .env file so they don't get nagged
# about keys they already have configured.
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key (for model access)"
    required: true
  - name: SERPAPI_KEY
    description: "SerpAPI key for web search"
    required: false
    default: ""
```

That's the whole manifest. Every field except `name` has a sensible default.

### Step 3 — Push to a git repo

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

The repo is now a distribution. Anyone with access can install it.

:::note
The git repo contains **everything in the profile directory except things already excluded from distributions**: `auth.json`, `.env`, `memories/`, `sessions/`, `state.db*`, `logs/`, `workspace/`, `*_cache/`, `local/`. Those stay on your machine. You can also add a `.gitignore` if you want to exclude additional paths.
:::

### Step 4 — Tag versioned releases

Every time the agent reaches a stable point, bump the version and tag:

```bash
# Edit distribution.yaml: version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

Recipients who run `hermes profile update research-bot` will pull the latest.

### What the repo looks like

A complete authored distribution:

```
research-bot/
├── distribution.yaml            # required
├── SOUL.md                      # strongly recommended
├── config.yaml                  # model, provider, tool defaults
├── mcp.json                     # MCP server connections
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # scheduled tasks
└── README.md                    # human-facing description (optional)
```

### Distribution-owned vs user-owned

When an installer updates to a new version, some things get replaced (author's domain) and some things stay put (installer's domain). Defaults:

| Category | Paths | On update |
|---|---|---|
| **Distribution-owned** | `SOUL.md`, `config.yaml`, `mcp.json`, `skills/`, `cron/`, `distribution.yaml` | Replaced from the new clone |
| **Config override** | `config.yaml` | Actually preserved by default — the installer may have tuned model or provider. Pass `--force-config` on update to reset. |
| **User-owned** | `memories/`, `sessions/`, `state.db*`, `auth.json`, `.env`, `logs/`, `workspace/`, `plans/`, `home/`, `*_cache/`, `local/` | Never touched |

You can override the distribution-owned list in the manifest:

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # only my research skills; other installed skills stay
  - cron/digest.json
```

When omitted, the defaults above apply — which is what most distributions want.

---

## For installers: using a distribution

### Install

```bash
hermes profile install github.com/you/research-bot --alias
```

What happens:

1. Clones the repo into a temporary directory.
2. Reads `distribution.yaml`, shows you the manifest (name, version, description, author, required env vars).
3. Checks each required env var against your shell environment and the target profile's existing `.env`. Marks each as `✓ set` or `needs setting` so you know exactly what to configure.
4. Asks for confirmation. Pass `-y` / `--yes` to skip.
5. Copies distribution-owned files into `~/.hermes/profiles/research-bot/` (or wherever the manifest's `name` resolves).
6. Writes `.env.EXAMPLE` with the required keys commented out — copy to `.env` and fill in.
7. With `--alias`, creates a wrapper so you can run `research-bot chat` directly.

### Source types

Any git URL works:

```bash
# GitHub shorthand
hermes profile install github.com/you/research-bot

# Full HTTPS
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# Self-hosted, GitLab, Gitea, Forgejo — any Git host
hermes profile install https://git.example.com/team/research-bot.git

# Private repo using your configured git auth
hermes profile install git@github.com:your-org/internal-bot.git

# Local directory during development (no git push needed)
hermes profile install ~/my-profile-in-progress/
```

### Override the profile name

Two users wanting the same distribution under different profile names:

```bash
# Alice
hermes profile install github.com/acme/support-bot --name support-us --alias
# Bob (same distribution, different local name)
hermes profile install github.com/acme/support-bot --name support-eu --alias
```

### Fill in env vars

After install, the agent's profile contains a `.env.EXAMPLE`:

```
# Environment variables required by this Hermes distribution.
# Copy to `.env` and fill in your own values before running.

# OpenAI API key (for model access)
# (required)
OPENAI_API_KEY=

# SerpAPI key for web search
# (optional)
# SERPAPI_KEY=
```

Copy it:

```bash
cp ~/.hermes/profiles/research-bot/.env.EXAMPLE ~/.hermes/profiles/research-bot/.env
# Edit .env, paste your real keys
```

Required keys that were already in your shell environment (e.g. `OPENAI_API_KEY` exported in your `~/.zshrc`) are marked `✓ set` during install — you don't need to duplicate them in `.env`.

### Check what you installed

```bash
hermes profile info research-bot
```

Shows:

```
Distribution: research-bot
Version:      1.0.0
Description:  Autonomous research assistant with arXiv and web tools
Author:       Your Name
Requires:     Hermes >=0.12.0
Source:       https://github.com/you/research-bot
Installed:    2026-05-08T17:04:32+00:00

Environment variables:
  OPENAI_API_KEY (required) — OpenAI API key (for model access)
  SERPAPI_KEY (optional) — SerpAPI key for web search
```

`hermes profile list` also shows a `Distribution` column so at a glance you can see which of your profiles came from repos and which you hand-built:

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆default         claude-sonnet-4              stopped      —            —
  coder           gpt-5                        stopped      coder        —
  research-bot    claude-opus-4                stopped      research-bot research-bot@1.0.0
  telemetry       claude-sonnet-4              running      telemetry    telemetry@2.3.1
```

### Update

```bash
hermes profile update research-bot
```

What happens:

1. Re-clones the repo from the recorded source URL.
2. Replaces distribution-owned files (SOUL, skills, cron, mcp.json).
3. **Preserves** your `config.yaml` — you may have tuned the model, temperature, or other settings. Pass `--force-config` to overwrite.
4. **Never touches** user data: memories, sessions, auth, `.env`, logs, state.

No re-downloading the whole archive. No stomping your local changes to config. No deleting your conversation history.

### Remove

```bash
hermes profile delete research-bot
```

The delete prompt surfaces distribution info before asking you to confirm:

```
Profile: research-bot
Path:    ~/.hermes/profiles/research-bot
Model:   claude-opus-4 (anthropic)
Skills:  12
Distribution: research-bot@1.0.0
Installed from: https://github.com/you/research-bot

This will permanently delete:
  • All config, API keys, memories, sessions, skills, cron jobs
  • Command alias (~/.local/bin/research-bot)

Type 'research-bot' to confirm:
```

So you never accidentally delete an agent without knowing where it came from or being able to re-install it.

---

## Use cases and patterns

### Personal: sync one agent across machines

You built a research assistant on your laptop. You want the same agent on your workstation.

```bash
# Laptop
cd ~/.hermes/profiles/research-bot
git init && git add . && git commit -m "initial"
git remote add origin git@github.com:you/research-bot.git
git push -u origin main

# Workstation
hermes profile install github.com/you/research-bot --alias
# Fill in .env. Done.
```

Any iteration on the laptop (`git commit && push`) pulls onto the workstation with `hermes profile update research-bot`. Memories stay per-machine — the laptop remembers its own conversations, the workstation remembers its own, they don't collide.

### Team: ship a reviewed internal agent

Your engineering team wants a shared PR-review bot with a specific SOUL, specific skills, and a cron that runs every PR through it.

```bash
# Engineering lead
cd ~/.hermes/profiles/pr-reviewer
# ... build and tune ...
git init && git add . && git commit -m "v1.0 PR reviewer"
git tag v1.0.0
git push -u origin main --tags    # push to your company's internal Git host

# Each engineer
hermes profile install git@github.com:your-org/pr-reviewer.git --alias
# Fill in .env with their own API key (billed to them), .env.EXAMPLE points at what's required
pr-reviewer chat
```

When the lead ships v1.1 (better SOUL, new skill), engineers run `hermes profile update pr-reviewer` and everyone's on the new version within minutes.

### Community: publish a public agent

You built something novel — maybe a "Polymarket trader" or an "academic paper summarizer" or a "Minecraft server ops assistant." You want to share it.

```bash
# You
cd ~/.hermes/profiles/polymarket-trader
# Write a solid README.md at the repo root — GitHub shows it on the repo page
git init && git add . && git commit -m "v1.0"
git tag v1.0.0
# Publish to a public GitHub repo
git remote add origin https://github.com/you/hermes-polymarket-trader.git
git push -u origin main --tags

# Anyone
hermes profile install github.com/you/hermes-polymarket-trader --alias
```

Tweet the install command. People who try it send you issues and PRs. If someone wants to customize, they fork — same git workflow everyone already knows.

### Product: ship an opinionated agent

You built Hermes-on-top — maybe a compliance-monitoring harness, a customer-support stack, a domain-specific research platform. You want to distribute it as a product.

```yaml
# distribution.yaml
name: telemetry-harness
version: 2.3.1
description: "Compliance telemetry harness — monitors and reviews regulated workflows"
hermes_requires: ">=0.13.0"
author: "Acme Compliance Inc."
license: "Commercial"

env_requires:
  - name: ACME_API_KEY
    description: "Your Acme Compliance license key (email support@acme.com)"
    required: true
  - name: OPENAI_API_KEY
    description: "OpenAI API key for model access"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "URL for your Graphiti knowledge graph instance"
    required: false
    default: "http://127.0.0.1:8000/sse"
```

Your customers install via a single command; the install preview tells them exactly which keys to have ready; updates roll out the moment you tag a new release; their compliance data (`memories/`, `sessions/`) never leaves their machine.

### Ephemeral: one-off scripts on shared infra

You're the ops lead. You want a temporary agent that diagnoses a production incident — a canned SOUL with the right tools and MCP connections — and runs on three on-call engineers' laptops for the next week.

```bash
# You
# Build the profile, commit, push a private repo
git push -u origin main

# Each on-call
hermes profile install git@github.com:your-org/incident-2026-q2.git --alias

# Incident resolved — tear it down
hermes profile delete incident-2026-q2
```

The install-delete cycle is cheap enough to be disposable.

---

## Recipes

### Pin to a specific version

:::note
Git ref pinning (`#v1.2.0`) is planned but not in the initial release — install currently tracks the default branch. Track your installed version via `hermes profile info <name>` and hold off on updates until you're ready.
:::

### Check what version you're on vs. latest

```bash
# Your installed version
hermes profile info research-bot | grep Version

# Latest upstream (without installing)
git ls-remote --tags https://github.com/you/research-bot | tail -5
```

### Keep local config customizations through updates

The default update behavior already does this: `config.yaml` is preserved. To be safe, write your local tweaks to a file the distribution doesn't own:

```yaml
# ~/.hermes/profiles/research-bot/local/my-overrides.yaml
# (distribution never touches local/)
```

…and reference it from `config.yaml` or your SOUL as needed.

### Force a clean re-install

```bash
# Nuke and re-install from scratch (loses memories/sessions too)
hermes profile delete research-bot --yes
hermes profile install github.com/you/research-bot --alias

# Update to current main but reset config.yaml to the distribution's default
hermes profile update research-bot --force-config --yes
```

### Fork and customize

The standard git workflow — distributions are just repos:

```bash
# Fork the repo on GitHub, then install your fork
hermes profile install github.com/yourname/forked-research-bot --alias

# Iterate locally in ~/.hermes/profiles/forked-research-bot/
# Edit SOUL.md, commit, push to your fork
# Upstream changes: pull them into your fork the usual way
```

### Test a distribution before pushing

From the author's machine:

```bash
# Install from a local directory (no git push needed)
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test --alias

# Tweak, delete, re-install until it's right
hermes profile delete research-bot-test --yes
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test
```

---

## What's NOT in a distribution (ever)

The installer hard-excludes these paths even if an author accidentally ships them. No config option lets you override this — the safety guard is a regression-tested invariant:

- `auth.json` — OAuth tokens, platform credentials
- `.env` — API keys, secrets
- `memories/` — conversation memory
- `sessions/` — conversation history
- `state.db`, `state.db-shm`, `state.db-wal` — session metadata
- `logs/` — agent and error logs
- `workspace/` — generated working files
- `plans/` — scratch plans
- `home/` — user's home mount in Docker backends
- `*_cache/` — image / audio / document caches
- `local/` — user-reserved customization namespace

When you clone a distribution, these simply aren't there. When you update, they stay put. If you installed the same distribution on five machines, you have five isolated sets of this data — one per machine.

## Security and trust

Profile distributions are unsigned by default. You're trusting:

- **The git host** (GitHub / GitLab / wherever) to serve the bytes the author pushed.
- **The author** to not ship a malicious SOUL, skills, or cron jobs.

Cron jobs from a distribution are **not auto-scheduled** — the installer prints `hermes -p <name> cron list` and you enable them explicitly. SOUL.md and skills ARE active as soon as you start chatting with the profile, so read them before your first run if you're installing from someone you don't know.

Rough analogy: installing a distribution is like installing a browser extension or a VS Code extension. Low friction, high power, trust the source. For internal company distributions, use a private repo and your normal git auth — nothing new to configure.

Future versions may add signing, a lockfile (`.distribution-lock.yaml`) with a resolved commit SHA, and a `--dry-run` flag that prints the diff before applying an update. None of those are shipping yet.

## Under the hood

For implementation details, precise CLI behavior, and all flags, see the [Profile Commands reference](../reference/profile-commands.md#distribution-commands).

The short version:

- `install`, `update`, `info` live inside `hermes profile` — not a parallel command tree.
- The manifest format is YAML with a tiny required schema (`name` only).
- The installer uses your local `git` binary for cloning, so any auth your shell already handles (SSH keys, credential helpers) works transparently.
- After clone, `.git/` is stripped — the installed profile isn't itself a git checkout, avoiding "oh my, I accidentally committed my `.env` to the distribution's git history" traps.
- Reserved profile names (`hermes`, `test`, `tmp`, `root`, `sudo`) are rejected at install time to avoid collisions with common binaries.

## See also

- [Profiles: Running Multiple Agents](./profiles.md) — the base concept
- [Profile Commands reference](../reference/profile-commands.md) — every flag, every option
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — local backup / restore (not distribution)
- [Using SOUL with Hermes](../guides/use-soul-with-hermes.md) — authoring personalities
- [Personality & SOUL](./features/personality.md) — how SOUL fits into the agent
- [Skills catalog](../reference/skills-catalog.md) — skills you can bundle
