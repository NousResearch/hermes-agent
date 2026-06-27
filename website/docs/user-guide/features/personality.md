---
sidebar_position: 9
title: "Personality & SOUL.md"
description: "Customize Hermes Agent's personality with SOUL.md, built-in personalities, and custom persona definitions"
---

# Personality & SOUL.md

Hermes Agent's personality is fully customizable. `SOUL.md` is the **primary identity** — it's the first thing in the system prompt and defines who the agent is.

- `SOUL.md` — a durable persona file that serves as the agent's identity (slot #1 in the system prompt). A trusted cwd-local soul can override the global profile soul for that session.
- built-in or custom `/personality` presets — session-level system-prompt overlays

If you want to change who Hermes is — or replace it with an entirely different agent persona — edit `SOUL.md`.

## How SOUL.md works now

Hermes seeds a default global `SOUL.md` automatically in:

```text
~/.hermes/SOUL.md
```

More precisely, it uses the current instance's `HERMES_HOME`, so if you run Hermes with a custom home directory, the global fallback is:

```text
$HERMES_HOME/SOUL.md
```

A trusted working directory can also provide a local soul for that session. Hermes only trusts cwd-local soul files when project context discovery is enabled for that run, which is the same boundary used for `AGENTS.md` and other context files. In isolation modes that set `skip_context_files=True` but still request a soul identity, Hermes skips cwd-local soul files and uses only the global `$HERMES_HOME/SOUL.md` fallback.

When local soul discovery is enabled, Hermes checks the current working directory first, then walks parent directories up to the git root if one exists.

Lookup order:

1. `.hermes/soul.md`
2. `.hermes/SOUL.md`
3. `soul.md`
4. `SOUL.md`
5. parent directories, using the same order and stopping at the git root
6. `$HERMES_HOME/SOUL.md`
7. built-in default identity

### Important behavior

- **SOUL.md is the agent's primary identity.** It occupies slot #1 in the system prompt, replacing the hardcoded default identity.
- Hermes creates a starter global `SOUL.md` automatically if one does not exist yet
- Existing user `SOUL.md` files are never overwritten
- A cwd-local soul overrides the global profile soul only when project context discovery is enabled for the session
- If context files are skipped but SOUL identity is still enabled, Hermes uses only `$HERMES_HOME/SOUL.md`
- `/status` shows the active SOUL.md source path when a soul file is loaded, so users can see which identity is in effect
- If no local soul exists, Hermes falls back to `$HERMES_HOME/SOUL.md`
- If the selected `SOUL.md` exists but is empty, or cannot be loaded, Hermes falls back to a built-in default identity
- If `SOUL.md` has content, that content is injected verbatim after security scanning and truncation
- SOUL.md is **not** duplicated in the context files section — it appears only once, as the identity

That makes `SOUL.md` a true identity layer, not just an additive context file.

## Why this design

This keeps personality predictable while supporting per-directory agent homes.

The global `$HERMES_HOME/SOUL.md` remains the stable default for normal use. Local soul files are opt-in: the agent identity changes only when the cwd or one of its parents deliberately contains a `soul.md`/`SOUL.md` file.

That gives you two clear patterns:
- edit `~/.hermes/SOUL.md` to change Hermes' default personality
- add `.hermes/soul.md` inside an agent or workspace directory to give that location its own identity

## Where to edit it

For your global default:

```bash
~/.hermes/SOUL.md
```

If you use a custom home:

```bash
$HERMES_HOME/SOUL.md
```

For a cwd-local agent identity:

```bash
.hermes/soul.md
```

## What should go in SOUL.md?

Use it for durable voice and personality guidance, such as:
- tone
- communication style
- level of directness
- default interaction style
- what to avoid stylistically
- how Hermes should handle uncertainty, disagreement, or ambiguity

Use it less for:
- one-off project instructions
- file paths
- repo conventions
- temporary workflow details

Those belong in `AGENTS.md`, not `SOUL.md`.

## Good SOUL.md content

A good SOUL file is:
- stable across contexts
- broad enough to apply in many conversations
- specific enough to materially shape the voice
- focused on communication and identity, not task-specific instructions

### Example

```markdown
# Personality

You are a pragmatic senior engineer with strong taste.
You optimize for truth, clarity, and usefulness over politeness theater.

## Style
- Be direct without being cold
- Prefer substance over filler
- Push back when something is a bad idea
- Admit uncertainty plainly
- Keep explanations compact unless depth is useful

## What to avoid
- Sycophancy
- Hype language
- Repeating the user's framing if it's wrong
- Overexplaining obvious things

## Technical posture
- Prefer simple systems over clever systems
- Care about operational reality, not idealized architecture
- Treat edge cases as part of the design, not cleanup
```

## What Hermes injects into the prompt

`SOUL.md` content goes directly into slot #1 of the system prompt — the agent identity position. No wrapper language is added around it.

The content goes through:
- prompt-injection scanning
- truncation if it is too large

If the file is empty, whitespace-only, or cannot be read, Hermes falls back to a built-in default identity ("You are Hermes Agent, an intelligent AI assistant created by Nous Research..."). This fallback also applies when `skip_context_files` is set (e.g., in subagent/delegation contexts).

## Security scanning

`SOUL.md` is scanned like other context-bearing files for prompt injection patterns before inclusion.

That means you should still keep it focused on persona/voice rather than trying to sneak in strange meta-instructions.

## SOUL.md vs AGENTS.md

This is the most important distinction.

### SOUL.md
Use for:
- identity
- tone
- style
- communication defaults
- personality-level behavior

### AGENTS.md
Use for:
- project architecture
- coding conventions
- tool preferences
- repo-specific workflows
- commands, ports, paths, deployment notes

A useful rule:
- if it should follow you everywhere, put it in the global `SOUL.md`
- if it is the identity for a specific agent or workspace, use local `.hermes/soul.md`
- if it is project/task instructions rather than identity, put it in `AGENTS.md`

## SOUL.md vs `/personality`

`SOUL.md` is your durable default personality.

`/personality` is a session-level overlay that changes or supplements the current system prompt.

So:
- `SOUL.md` = baseline voice
- `/personality` = temporary mode switch

Examples:
- keep a pragmatic default SOUL, then use `/personality teacher` for a tutoring conversation
- keep a concise SOUL, then use `/personality creative` for brainstorming

## Built-in personalities

Hermes ships with built-in personalities you can switch to with `/personality`.

| Name | Description |
|------|-------------|
| **helpful** | Friendly, general-purpose assistant |
| **concise** | Brief, to-the-point responses |
| **technical** | Detailed, accurate technical expert |
| **creative** | Innovative, outside-the-box thinking |
| **teacher** | Patient educator with clear examples |
| **kawaii** | Cute expressions, sparkles, and enthusiasm ★ |
| **catgirl** | Neko-chan with cat-like expressions, nya~ |
| **pirate** | Captain Hermes, tech-savvy buccaneer |
| **shakespeare** | Bardic prose with dramatic flair |
| **surfer** | Totally chill bro vibes |
| **noir** | Hard-boiled detective narration |
| **uwu** | Maximum cute with uwu-speak |
| **philosopher** | Deep contemplation on every query |
| **hype** | MAXIMUM ENERGY AND ENTHUSIASM!!! |

## Switching personalities with commands

### CLI

```text
/personality
/personality concise
/personality technical
```

### Messaging platforms

```text
/personality teacher
```

These are convenient overlays, but your selected `SOUL.md` still gives Hermes its persistent default personality unless the overlay meaningfully changes it.

## Custom personalities in config

You can also define named custom personalities in `~/.hermes/config.yaml` under `agent.personalities`.

```yaml
agent:
  personalities:
    codereviewer: >
      You are a meticulous code reviewer. Identify bugs, security issues,
      performance concerns, and unclear design choices. Be precise and constructive.
```

Then switch to it with:

```text
/personality codereviewer
```

## Recommended workflow

A strong default setup is:

1. Keep a thoughtful global `SOUL.md` in `~/.hermes/SOUL.md`
2. Optionally add `.hermes/soul.md` inside directories that should behave like their own agent homes
3. Put project instructions in `AGENTS.md`
4. Use `/personality` only when you want a temporary mode shift

That gives you:
- a stable voice
- optional folder-scoped agent identities
- project-specific behavior where it belongs
- temporary control when needed

## How personality interacts with the full prompt

At a high level, the prompt stack includes:
1. **SOUL.md** (agent identity — or built-in fallback if SOUL.md is unavailable)
2. tool-aware behavior guidance
3. memory/user context
4. skills guidance
5. context files (`AGENTS.md`, `.cursorrules`)
6. timestamp
7. platform-specific formatting hints
8. optional system-prompt overlays such as `/personality`

`SOUL.md` is the foundation — everything else builds on top of it.

## Related docs

- [Context Files](/user-guide/features/context-files)
- [Configuration](/user-guide/configuration)
- [Tips & Best Practices](/guides/tips)
- [SOUL.md Guide](/guides/use-soul-with-hermes)

## CLI appearance vs conversational personality

Conversational personality and CLI appearance are separate:

- `SOUL.md`, `agent.system_prompt`, and `/personality` affect how Hermes speaks
- `display.skin` and `/skin` affect how Hermes looks in the terminal

For terminal appearance, see [Skins & Themes](./skins.md).
