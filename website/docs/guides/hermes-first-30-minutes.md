---
sidebar_position: 0
title: "Hermes First 30 Minutes"
description: "An opinionated first-run playbook for installing Hermes, choosing a provider, verifying a working chat, and fixing the most common setup failures"
---

# Hermes First 30 Minutes

This is the guide I wish every new Hermes user had on day one.

If the Quickstart shows you the happy path, this guide gets you to a setup that actually survives real use: install Hermes, choose a provider, verify a working chat, and know exactly what to do when something breaks.

## Who this is for

Use this guide if you are:

- brand new and want the shortest path to a working setup
- switching providers and do not want to lose time to config mistakes
- setting Hermes up for a team, bot, or always-on workflow
- tired of “it installed, but it still does nothing”

## The fastest path

Pick the path that matches your goal.

| Goal | Do this first | Then do this |
|---|---|---|
| I just want Hermes working on my machine | `hermes setup` | Run a real chat and verify it responds |
| I already know my provider | `hermes model` | Save the config, then start chatting |
| I want a bot or always-on setup | `hermes gateway setup` after CLI works | Connect Telegram, Discord, Slack, or another platform |
| I want a local or self-hosted model | `hermes model` → custom endpoint | Verify the endpoint, model name, and context length |
| I want multi-provider fallback | `hermes model` first | Add routing and fallback only after the base chat works |

Rule of thumb:

- if Hermes cannot complete a normal chat, do not add more features yet
- get one clean conversation working first
- only then layer on gateway, cron, skills, voice, or routing

## 1. Install Hermes

Use the standard installer on Linux, macOS, or WSL2:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Windows users should install WSL2 first, then run the command above inside the WSL2 terminal.

After install, reload your shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

## 2. Choose a provider before you touch anything else

The single most important setup step is choosing the provider and model correctly.

Use `hermes model` if you want Hermes to guide you through the choice:

```bash
hermes model
```

Good defaults:

| Situation | Recommended path |
|---|---|
| You want the least friction | Nous Portal or OpenRouter |
| You already have Claude or Codex auth | Anthropic or OpenAI Codex |
| You want local/private inference | Ollama or another custom OpenAI-compatible endpoint |
| You want multi-provider routing | OpenRouter |
| You want a custom GPU server | vLLM, SGLang, LiteLLM, or any OpenAI-compatible endpoint |

For most first-time users, the win is simple:

- use `hermes model`
- choose a provider
- accept the defaults unless you know why you are changing them

If you are not sure, do not manually edit the provider config yet.

## 3. Save settings the Hermes way

Hermes separates secrets from normal config:

- secrets and tokens go in `~/.hermes/.env`
- non-secret settings go in `~/.hermes/config.yaml`

The easiest way to avoid mistakes is to use the CLI instead of editing files by hand.

Examples:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

That gives you three benefits:

- the right value goes to the right file
- the config survives restarts
- you can verify it later with `hermes config get ...`

If you are using a custom endpoint, make sure the base URL and model name match the server exactly.
A surprising number of “Hermes is broken” reports are just endpoint mismatches.

## 4. Run the first real chat

Now start Hermes:

```bash
hermes
```

Use a prompt that is specific and easy to verify.

Good first prompts:

```text
Summarize this repo in 5 bullets and tell me what the main entrypoint is.
```

```text
Check my current directory and tell me what looks like the main project file.
```

```text
Help me set up a clean GitHub PR workflow for this codebase.
```

What success looks like:

- the banner shows your chosen model/provider
- Hermes replies without error
- it can use a tool if needed
- the conversation continues normally for more than one turn

If that works, you are past the hardest part.

## 5. Verify the session lifecycle

Before you move on, make sure resume works:

```bash
hermes --continue
```

That should bring you back to the most recent session.

If it does not, check whether you are in the same profile and whether the session actually saved.
This matters later when you are juggling multiple setups or machines.

## 6. Add the next layer only after the base chat works

Now that the core chat is alive, add the next capability in this order.

### If you want a bot or shared assistant

```bash
hermes gateway setup
```

Then connect Telegram, Discord, Slack, WhatsApp, Signal, Email, or Home Assistant.

### If you want more automation

- `hermes tools` to tune tool access
- `hermes skills` to browse and install reusable workflows
- `hermes gateway` to run the messaging stack
- cron only after your bot or CLI setup is stable

### If you want a more polished runtime

- Docker or another sandboxed backend for untrusted work
- voice mode if you want spoken replies
- browser if you want web workflows
- fallback model or provider routing only after the base provider is stable

## 7. Common failure modes and fixes

These are the problems that waste the most time.

| Symptom | Likely cause | Fix |
|---|---|---|
| Hermes opens but gives empty or broken replies | Provider auth or model selection is wrong | Run `hermes model` again and confirm the provider, model, and auth method |
| The custom endpoint “works” but returns garbage or nothing | Wrong base URL, wrong model name, or not actually OpenAI-compatible | Verify the endpoint in a separate client first, then re-run `hermes model` |
| The gateway starts but nobody can message it | Bot token, allowlist, or platform setup is incomplete | Re-run `hermes gateway setup` and check `hermes gateway status` |
| `hermes --continue` cannot find the old session | You switched profiles or never saved that session | Check `hermes sessions list` and confirm you are in the right Hermes home/profile |
| Hermes says a model is unavailable or falls back oddly | Provider routing or fallback settings are too aggressive | Keep routing off until the base provider is stable |
| `hermes doctor` flags config problems | Config values are missing or stale | Fix the config, then retest a plain chat before adding more features |

## 8. Recovery toolkit

When something feels off, use this order:

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

That sequence gets you from “broken vibes” back to a known state fast.

## 9. What to do next

Once you have one working chat, pick one next step:

- [Configuration](/docs/user-guide/configuration)
- [AI Providers](/docs/integrations/providers)
- [Messaging Gateway](/docs/user-guide/messaging)
- [Tools & Toolsets](/docs/user-guide/features/tools)
- [Skills System](/docs/user-guide/features/skills)
- [Provider Routing](/docs/user-guide/features/provider-routing)
- [Fallback Providers](/docs/user-guide/features/fallback-providers)
- [Sessions](/docs/user-guide/sessions)
- [Tips & Best Practices](/docs/guides/tips)

If you want the ultra-opinionated version of the docs, this is it: get one clean conversation working, then expand from there.
