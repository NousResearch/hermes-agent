---
sidebar_position: 7
title: "Use SOUL.md with Hermes"
description: "How to use SOUL.md to shape Hermes Agent's default voice, what belongs there, and how it differs from AGENTS.md and /personality"
---

# Use SOUL.md with Hermes

`SOUL.md` is the **primary identity** for your Hermes instance. It's the first thing in the system prompt — it defines who the agent is, how it speaks, and what it avoids.

If you want Hermes to feel like the same assistant every time you talk to it — or if you want to replace the Hermes persona entirely with your own — this is the file to use.

## What SOUL.md is for

Use `SOUL.md` for:
- tone
- personality
- communication style
- how direct or warm Hermes should be
- what Hermes should avoid stylistically
- how Hermes should relate to uncertainty, disagreement, and ambiguity

In short:
- `SOUL.md` is about who Hermes is and how Hermes speaks

## What SOUL.md is not for

Do not use it for:
- repo-specific coding conventions
- file paths
- commands
- service ports
- architecture notes
- project workflow instructions

Those belong in `AGENTS.md`.

A good rule:
- if it should apply everywhere, put it in `SOUL.md`
- if it only belongs to one project, put it in `AGENTS.md`

## Where it lives

Hermes now uses only the global SOUL file for the current instance:

```text
~/.hermes/SOUL.md
```

If you run Hermes with a custom home directory, it becomes:

```text
$HERMES_HOME/SOUL.md
```

## First-run behavior

Hermes automatically seeds a starter `SOUL.md` for you if one does not already exist.

That means most users now begin with a real file they can read and edit immediately.

Important:
- if you already have a `SOUL.md`, Hermes does not overwrite it
- if the file exists but is empty, Hermes adds nothing from it to the prompt

## How Hermes uses it

When Hermes starts a session, it reads `SOUL.md` from `HERMES_HOME`, scans it for prompt-injection patterns, truncates it if needed, and uses it as the **agent identity** — slot #1 in the system prompt. This means SOUL.md completely replaces the built-in default identity text.

If SOUL.md is missing, empty, or cannot be loaded, Hermes falls back to a built-in default identity.

No wrapper language is added around the file. The content itself matters — write the way you want your agent to think and speak.

## A good first edit

If you do nothing else, open the file and change just a few lines so it feels like you.

For example:

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

That alone can noticeably change how Hermes feels.

## Example styles

### 1. Pragmatic engineer

```markdown
You are a pragmatic senior engineer.
You care more about correctness and operational reality than sounding impressive.

## Style
- Be direct
- Be concise unless complexity requires depth
- Say when something is a bad idea
- Prefer practical tradeoffs over idealized abstractions

## Avoid
- Sycophancy
- Hype language
- Overexplaining obvious things
```

### 2. Research partner

```markdown
You are a thoughtful research collaborator.
You are curious, honest about uncertainty, and excited by unusual ideas.

## Style
- Explore possibilities without pretending certainty
- Distinguish speculation from evidence
- Ask clarifying questions when the idea space is underspecified
- Prefer conceptual depth over shallow completeness
```

### 3. Teacher / explainer

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. Tough reviewer

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## What makes a strong SOUL.md?

A strong `SOUL.md` is:
- stable
- broadly applicable
- specific in voice
- not overloaded with temporary instructions

A weak `SOUL.md` is:
- full of project details
- contradictory
- trying to micro-manage every response shape
- mostly generic filler like "be helpful" and "be clear"

Hermes already tries to be helpful and clear. `SOUL.md` should add real personality and style, not restate obvious defaults.

## Suggested structure

You do not need headings, but they help.

A simple structure that works well:

```markdown
# Identity
Who Hermes is.

# Style
How Hermes should sound.

# Avoid
What Hermes should not do.

# Defaults
How Hermes should behave when ambiguity appears.
```

## SOUL.md vs /personality

These are complementary.

Use `SOUL.md` for your durable baseline.
Use `/personality` for temporary mode switches.

Examples:
- your default SOUL is pragmatic and direct
- then for one session you use `/personality teacher`
- later you switch back without changing your base voice file

## SOUL.md vs AGENTS.md

This is the most common mistake.

### Put this in SOUL.md
- “Be direct.”
- “Avoid hype language.”
- “Prefer short answers unless depth helps.”
- “Push back when the user is wrong.”

### Put this in AGENTS.md
- “Use pytest, not unittest.”
- “Frontend lives in `frontend/`.”
- “Never edit migrations directly.”
- “The API runs on port 8000.”

## How to edit it

```bash
nano ~/.hermes/SOUL.md
```

or

```bash
vim ~/.hermes/SOUL.md
```

Then restart Hermes or start a new session.

## A practical workflow

1. Start with the seeded default file
2. Trim anything that does not feel like the voice you want
3. Add 4–8 lines that clearly define tone and defaults
4. Talk to Hermes for a while
5. Adjust based on what still feels off

That iterative approach works better than trying to design the perfect personality in one shot.

## Troubleshooting

### I edited SOUL.md but Hermes still sounds the same

Check:
- you edited `~/.hermes/SOUL.md` or `$HERMES_HOME/SOUL.md`
- not some repo-local `SOUL.md`
- the file is not empty
- your session was restarted after the edit
- a `/personality` overlay is not dominating the result

### Hermes is ignoring parts of my SOUL.md

Possible causes:
- higher-priority instructions are overriding it
- the file includes conflicting guidance
- the file is too long and got truncated
- some of the text resembles prompt-injection content and may be blocked or altered by the scanner

### My SOUL.md became too project-specific

Move project instructions into `AGENTS.md` and keep `SOUL.md` focused on identity and style.

## Related docs

- [Personality & SOUL.md](/docs/user-guide/features/personality)
- [Context Files](/docs/user-guide/features/context-files)
- [Configuration](/docs/user-guide/configuration)
- [Tips & Best Practices](/docs/guides/tips)
