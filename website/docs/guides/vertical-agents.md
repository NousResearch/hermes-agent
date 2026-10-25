---
sidebar_position: 19
title: "Building Constrained Vertical Agents"
description: "How to build a domain-specific Hermes agent that stays inside its lane — identity, context, workflow, and tool constraints, and where each belongs"
---

# Building Constrained Vertical Agents

A **constrained vertical agent** is a Hermes instance shaped for one domain — support tickets, research, infra checks, whatever — and discouraged from wandering into generalist territory. Done right, it stops reaching for tools it doesn't need, stops answering like a generic assistant, and follows a workflow it was actually handed rather than improvising one each turn.

**The drift you are trying to prevent** is usually gradual:

- It gets *too proactive* — offering to run eleven tools when three would do.
- It gets *too generalist* — answering like the public Hermes persona instead of your domain voice.
- It accumulates *prompt-layer patchwork* — a pile of "don't do X, never do Y" lines that only hold until the next model change.

**A good rule:** the constraints that keep a vertical agent inside its lane should live in *config and skills*, not in the prompt. The prompt's identity slot is for voice, and skills are for workflow — neither is a firewall of "don'ts". This guide shows you which primitive owns which concern, and why that ordering matters.

## The Four-Layer Model

Every vertical agent is built from four independent layers. Put each concern in its proper home and you rarely have to choose between it.

| Concern | Mechanism | Where it lives |
|---|---|---|
| Identity & voice | `SOUL.md` | `$HERMES_HOME/SOUL.md` (per-profile: `~/.hermes/profiles/<name>/SOUL.md`) |
| User & domain context | `USER.md` (+ `MEMORY.md`) | `$HERMES_HOME/memories/USER.md` (per-profile: `…/profiles/<name>/memories/USER.md`) |
| Workflow & procedure | Skills | `~/.hermes/skills/` (per-profile: `…/profiles/<name>/skills/`) |
| Tool & capability constraints | Profiles + `platform_toolsets` + `disabled_toolsets` | `config.yaml` |

**The ordering is deliberate.** Identity is slot #1 in the system prompt; context files are injected at session start; skills are loaded on demand; toolsets are resolved at boot. Because each layer has its own reload cadence, you can edit a skill mid-session without touching the agent's voice, and you can swap a profile's toolsets without rewriting the prompt.

## SOUL.md: Identity Only

`SOUL.md` is who the agent *is* and how it *sounds*. That is its whole job. Nothing else.

**Put in SOUL.md:**

- tone and directness ("be concise, push back when an idea is weak")
- communication style ("lead with the recommendation, then evidence")
- how it relates to uncertainty ("flag speculation separately from evidence")
- what it should avoid stylistically ("no politeness theater, no hype")

**Do NOT put in SOUL.md:**

- domain facts ("the support queue is in Zendesk") → `USER.md`
- when to use which tool or in what order → a skill
- project commands, ports, repo conventions → `AGENTS.md` / `.hermes.md`

**A good rule:** if editing the file changes how the agent *sounds* rather than what it *does*, it belongs in `SOUL.md`.

This is the most common mistake, and it is exactly the mix-up that lands facts in the wrong file. If you put domain facts in `SOUL.md` then wonder why `USER.md` stayed empty, see [Which File Does What?](/user-guide/which-file-does-what). For the full mechanics — first-run seeding, threat scanning, truncation cap, and the new-session reload requirement — see [Use SOUL.md with Hermes](/guides/use-soul-with-hermes) and [Personality & SOUL.md](/user-guide/features/personality#what-should-go-in-soulmd). Voice design itself is covered there; this guide is about *structure*.

## USER.md: User & Domain Context

`USER.md` is the agent's model of who the user is and what domain it is operating in. The agent writes it through the `memory` tool; you can also hand-author it, and you should — seed the domain facts it needs to operate.

**A vertical agent's USER.md typically holds:**

- who the user is and how they want to be addressed
- the user's role and authority level ("can approve deploy to prod")
- domain facts: the ticketing system, the repo layout, the SLA thresholds
- standing expectations ("escalate after 2 hours", "never touch prod on Fridays")

**MEMORY.md vs USER.md** is a standing source of confusion. `USER.md` is the *user profile* — persistent facts about who the user is. `MEMORY.md` is the agent's *own notes* — environment observations, tool quirks, things learned during work. Both load at session init from the `memories/` directory and both are injected as a frozen snapshot, so edits only appear after a new session.

Two config gates control this, both on by default:

```yaml
# ~/.hermes/profiles/<name>/config.yaml
memory:
  memory_enabled: true          # toggles MEMORY.md
  user_profile_enabled: true    # toggles USER.md
```

The default profile lives at `~/.hermes/memories/USER.md`; a named profile lives at `~/.hermes/profiles/<name>/memories/USER.md`. For the full decision table and the rest of the "where does this fact go?" guidance, read [Which File Does What?](/user-guide/which-file-does-what).

## Skills: Workflow, Not Voice

Skills are the right home for *procedure*: when to use which tool, in what order, what evidence a step requires, and what to do when the planned path fails. Because skills load on demand via `skill_view` (they are not in the system prompt until needed), they carry none of the token cost of voice and none of the drift of a prompt firewall.

**A vertical-agent skill should encode:**

- trigger conditions ("when the user references a ticket number or asks about queue status")
- tool order ("check the queue → classify → assign → comment, never reply before classifying")
- evidence requirements ("before assigning, log the triage rationale in a todo step")
- fallback policy ("if the queue API is unreachable, fall back to reading the on-call runbook")

**A skill should NOT encode:**

- how the agent sounds or how to phrase each answer — that is `SOUL.md`
- rigid answer templates — they rot and break when the model changes
- environment facts that change per user — that is `USER.md`

See [Working with Skills](/guides/work-with-skills) for the frontmatter fields and loading model, and [Skills System](/user-guide/features/skills) for per-platform enable/disable (`skills.disabled` globally and `skills.platform_disabled.<platform>` in `config.yaml`, or the `hermes skills config` TUI). Mid-session rescans are possible with `/reload-skills` — it re-reads skills from disk, so it also picks up edits to an existing skill's `SKILL.md`, not just newly added or removed skills. The system prompt is rebuilt on the next session, so the four-layer reload cadence above still applies.

## Tool Pruning Through Native Config

If a vertical agent only needs six tools, give it six tools. Prompt-level constraints ("do not use the browser tool") are the weakest possible fence: they degrade under scale and disappear with a model update. Structural pruning does not.

There are three levers, in order of preference:

1. **A dedicated profile.** `hermes profile create <name> --clone` copies `config.yaml`, `.env`, `SOUL.md`, and `memories/{MEMORY,USER}.md` into `~/.hermes/profiles/<name>/` with isolated sessions, skills, and memories. One profile per domain is the single most effective drift control. Make it the sticky default with `hermes profile use <name>`. See [Profiles](/user-guide/profiles).

2. **Per-platform `platform_toolsets`.** The `hermes tools` interactive wizard writes `platform_toolsets.<platform>` — a per-platform list of toolset names. A support agent on the CLI might keep only `web`, `terminal`, `file`, `skills`, `todo`, `cronjob`; a Discord bot for the same desk might keep only `web`, `file`, `skills`, `todo`. Platform keys include `cli`, `telegram`, `discord`, `whatsapp`, `slack`, `signal`, `homeassistant`, `qqbot`, `yuanbao`, `teams`, `google_chat` — see [Platform-Specific Toolsets](/user-guide/messaging#platform-specific-toolsets) for the full table.

3. **Global `agent.disabled_toolsets`.** When you want one switch that removes a toolset everywhere (across CLI and every gateway platform), list it once:

```yaml
# ~/.hermes/profiles/<name>/config.yaml
agent:
  disabled_toolsets:
    - image_gen      # no image generation anywhere
    - tts            # no text-to-speech anywhere
```

This applies *after* per-platform config, so a toolset here is always removed even if a platform's saved config still lists it. See the [Global Toolset Disable](/user-guide/configuration#global-toolset-disable) section.

Two things to keep in mind:

- **A `platform_toolsets` list is the exact composition, not a floor.** When you list toolsets for a platform, the agent gets precisely those toolsets — that is how the support example below ends up with no vision, no image generation, no browser automation, and no voice. You are not trimming a default; you are declaring the whole surface.
- **`agent.disabled_toolsets` is a subtraction pass.** It removes tools after platform resolution. Disabling a *plain* toolset (`image_gen`, `tts`, `browser`, `vision`, …) removes its tools everywhere. Disabling a *platform bundle* name (`hermes-cli`, `hermes-telegram`, …) is softer: only the bundle's non-core extras are removed, because the terminal/file/web-search essentials shared by every bundle are preserved.
- **Toolset changes land on a new session.** Because they affect the system prompt, they require `/reset` (or a restart) to take effect — they never swap mid-conversation, which keeps prompt caching intact. Don't try to hot-reload toolsets and expect them to bind in the current conversation.

For the full toolset registry and presets (`hermes-cli`, `hermes-telegram`, …), see [Tools & Toolsets](/user-guide/features/tools).

## Helper-First, Fallback-Last

When a vertical agent needs to accomplish something, prefer *existing helpers* over ad hoc glue code: a CLI command, a skill in the catalog, a plugin, or an MCP server. If the job is common enough to have been solved before, the maintained helper is more reliable and easier to swap than a one-off script the agent wrote in a hurry.

Improvise, don't architect. Hand-rolled functions have value — they get a job done when nothing else fits — but treat them as a fallback, not a default. A vertical agent that writes its own tooling on every task is drifting toward generalist again. The test is simple: did a maintained helper exist that you skipped? If so, prefer the helper next time and save the improv to a skill so it stops being improvised.

## Avoiding Generalist Drift

Drift is not a voice problem; it is a structure problem. The remedies are structural, not prompt-level:

- **Constraints live in config and skills, not in the prompt.** A fence made of "don't" sentences rots; a fence made of an empty toolset or a missing skill does not.
- **Evidence requirements live in the skill.** Require a todo step, a logged rationale, or an approval gate before the agent can act — not a plea in SOUL.md.
- **One profile per domain.** Separate state (memory, skills, config, logs) is the cleanest isolation. A single overloaded profile will leak behavior between contexts.
- **Review cadence.** Watch tool usage over a week of real sessions: remove toolsets that were never called, tighten the skill's trigger conditions if it fired on the wrong requests, and keep the domain facts in `USER.md` current as edge cases appear. Iterate on the skill and the config, not on the voice.

## Example: Support Desk Agent

A profile that only handles inbound support tickets.

**1. Isolate it in a profile.**

```bash
hermes profile create support --clone
hermes profile use support
```

**2. `SOUL.md` — voice only.**

```markdown
# Identity
You are a calm, pragmatic support agent for the Acme customer-success team.

# Style
- Lead with the actionable summary, then evidence.
- Be concise; do not restate the ticket.
- Flag uncertainty ("I can't see the logs for that yet") instead of hedging.

# Avoid
- Speculation dressed up as answers.
- Over-explaining obvious steps.
```

**3. `memories/USER.md` — who the user is and the domain.**

```markdown
# User Profile
Name: Dana
Role: Customer-success lead (can approve refunds up to $200)
Communication style: prefers bullet points, minimal preamble.
Timezone: America/Los_Angeles

# Domain
- Ticketing system: Zendesk. Queue is "support-urgent" for SLA < 4h.
- Refund policy: up to $200 with approval; above that, escalate to finance@.
- SLA: acknowledge within 30 min, resolve or escalate within 4 h.
```

**4. The workflow skill.** The agent's procedure lives in `skills/support/support-ticket/SKILL.md`:

```markdown title="~/.hermes/profiles/support/skills/support/support-ticket/SKILL.md"
---
name: support-ticket
description: Triages, assigns, and resolves inbound Zendesk tickets.
version: 1.0.0
metadata:
  hermes:
    tags: [support, zendesk, triage]
---

# Support Ticket

## When to Use
Use this skill when the user references a Zendesk ticket ID, asks about queue
status, or requests a triage/assignment action on an inbound ticket.

## Procedure
1. Read the ticket from Zendesk (`zendesk fetch <id>`).
2. Classify: bug / question / refund request.
3. Check the SLA clock — if < 1h remaining, tag `priority::sla-breach`.
4. Assign to the on-call engineer from the schedule.
5. Comment with the triage rationale and next step.
6. If the user can approve (Dana, up to $200 refunds), offer the action.

## Evidence Requirements
- Log the classification and SLA status in a `todo` step before assigning.
- Do not comment until the ticket has been read and classified.

## Pitfalls
- Do not assume the ticket body matches the user's summary — re-read it.
- Do not edit a ticket you have not loaded first.

## Fallback
If the Zendesk API is unreachable, read the on-call runbook stored in the
profile's `AGENTS.md` and tell the user which step failed.
```

**5. Prune the toolsets for the platform.**

```yaml
# ~/.hermes/profiles/support/config.yaml
platform_toolsets:
  cli: [web, terminal, file, skills, todo, cronjob]

agent:
  disabled_toolsets:
    - image_gen
    - tts
    - browser
```

Now the agent can search the web, run commands, read and edit files, use skills, track steps in todos, and run scheduled checks — and nothing else. No vision, no image generation, no browser automation, no voice.

Because the composition is exact, that list is the whole surface: tools like `delegation` (subagents) and `session_search` (recalling past sessions) are off unless you add them explicitly. A support agent that should hand escalations to a subagent or search prior ticket threads should list them here.

## Example: Research Analyst

A second profile, built the same way, tuned for a different domain. Here is the shape of the difference:

```bash
hermes profile create research --clone
hermes profile use research
```

```yaml
# ~/.hermes/profiles/research/config.yaml
platform_toolsets:
  cli: [web, terminal, file, skills, todo, vision, session_search]

agent:
  disabled_toolsets:
    - image_gen        # not an illustrator
    - tts              # not a speaker
    - cronjob          # not a scheduler
```

The `research` agent keeps `vision` (to read charts in papers) and `session_search` (to recall prior findings) but drops `image_gen`, `tts`, and `cronjob` that the `support` agent had. These lists are illustrative, not a prescription — a research agent that runs scheduled literature scans or recurring searches should keep `cronjob`. Its domain facts go in `memories/USER.md` (the analyst's preferred databases, the standing search queries, the embargo policy), and its procedure lives in a skill like `research-assistant`.

For a complete worked tutorial of a vertical Hermes agent — built profile, skill, and all — see [GitHub PR Review Agent](/guides/github-pr-review-agent). The only thing that changes between domains is the *what goes where* mapping this guide teaches: swap SOUL voice, swap USER.md domain facts, swap the skill, swap the toolsets. The structure is identical.

## Iterating

The four-layer model pays off at iteration time, because each layer reloads on its own schedule:

- **Skill changes** — edit `SKILL.md`, then `/reload-skills` to rescan installed skills mid-session.
- **Config + toolsets changes** — any `config.yaml` edit requires `/reset` to take effect (it rebuilds the system prompt). Do it before a long session, not mid-task.
- **Memory (`USER.md` / `MEMORY.md`)** — new sessions pick up the on-disk snapshot; edits never appear in a running conversation.
- **Voice (`SOUL.md`)** — same rule: new session to feel the change.

**What to watch for:**

- Toolsets that are listed but never called — remove them and `/reset`.
- A skill that fires on requests outside its trigger conditions — narrow the trigger, don't add a "don't fire on X" line to the prompt.
- Domain facts that have gone stale — update `USER.md`; they are cheap to keep current and expensive to get wrong.

Start with a clone of your main profile, trim one layer at a time, and resist the urge to add the missing constraint to the prompt. If a constraint belongs in config or a skill, you already know where it goes.
