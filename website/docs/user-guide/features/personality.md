---
sidebar_position: 9
title: "Personality & SOUL.md"
description: "Customize Hermes Agent's personality — SOUL.md, built-in personalities, and custom persona definitions"
---

# Personality & SOUL.md

Hermes Agent's personality is fully customizable. You can use the built-in personality presets, create a global SOUL.md file, or define your own custom personas in config.yaml.

## SOUL.md — Custom Personality File

SOUL.md is a special context file that defines the agent's personality, tone, and communication style. It's injected into the system prompt at session start.

### Where to Place It

| Location | Scope |
|----------|-------|
| `./SOUL.md` (project directory) | Per-project personality |
| `~/.hermes/SOUL.md` | Global default personality |

The project-level file takes precedence. If no SOUL.md exists in the current directory, Hermes falls back to the global one in `~/.hermes/`.

### How It Affects the System Prompt

When a SOUL.md file is found, it's included in the system prompt with this instruction:

> *"If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it."*

The content appears under a `## SOUL.md` section within the `# Project Context` block of the system prompt.

### Example SOUL.md

```markdown
# Personality

You are a pragmatic senior engineer with strong opinions about code quality.
You prefer simple solutions over complex ones.

## Communication Style
- Be direct and to the point
- Use dry humor sparingly
- When something is a bad idea, say so clearly
- Give concrete recommendations, not vague suggestions

## Code Preferences  
- Favor readability over cleverness
- Prefer explicit over implicit
- Always explain WHY, not just what
- Suggest tests for any non-trivial code

## Pet Peeves
- Unnecessary abstractions
- Comments that restate the code
- Over-engineering for hypothetical future requirements
```

:::tip
SOUL.md is scanned for prompt injection patterns before being loaded. Keep the content focused on personality and communication guidance — avoid instructions that look like system prompt overrides.
:::

## Built-In Personalities

Hermes ships with 14 built-in personalities defined in the CLI config. Switch between them with the `/personality` command.

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

### Examples

**kawaii:**
`You are a kawaii assistant! Use cute expressions and sparkles, be super enthusiastic about everything! Every response should feel warm and adorable desu~!`

**noir:**
> The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?

**pirate:**
> Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!

## Switching Personalities

### CLI: /personality Command

```
/personality            — List all available personalities
/personality kawaii      — Switch to kawaii personality
/personality technical   — Switch to technical personality
```

When you set a personality via `/personality`, it:
1. Sets the system prompt to that personality's text
2. Forces the agent to reinitialize
3. Saves the choice to `agent.system_prompt` in `~/.hermes/config.yaml`

The change persists across sessions until you set a different personality or clear it.

### Gateway: /personality Command

On messaging platforms (Telegram, Discord, etc.), the `/personality` command works the same way:

```
/personality kawaii
```

### Config File

Set a personality directly in config:

```yaml
# In ~/.hermes/config.yaml
agent:
  system_prompt: "You are a concise assistant. Keep responses brief and to the point."
```

Or via environment variable:

```bash
# In ~/.hermes/.env
HERMES_EPHEMERAL_SYSTEM_PROMPT="You are a pragmatic engineer who gives direct answers."
```

:::info
The environment variable `HERMES_EPHEMERAL_SYSTEM_PROMPT` takes precedence over the config file's `agent.system_prompt` value.
:::

## Custom Personalities

### Defining Custom Personalities in Config

Add your own personalities to `~/.hermes/config.yaml` under `agent.personalities`:

```yaml
agent:
  personalities:
    # Built-in personalities are still available
    # Add your own:
    codereviewer: >
      You are a meticulous code reviewer. For every piece of code shown,
      identify potential bugs, performance issues, security vulnerabilities,
      and style improvements. Be thorough but constructive.
    
    mentor: >
      You are a kind, encouraging coding mentor. Break down complex concepts
      into digestible pieces. Celebrate small wins. When the user makes a
      mistake, guide them to the answer rather than giving it directly.
    
    sysadmin: >
      You are an experienced Linux sysadmin. You think in terms of
      infrastructure, reliability, and automation. Always consider
      security implications and prefer battle-tested solutions.
    
    dataengineer: >
      You are a data engineering expert specializing in ETL pipelines,
      data modeling, and analytics infrastructure. You think in SQL
      and prefer dbt for transformations.
```

Then use them with `/personality`:

```
/personality codereviewer
/personality mentor
```

### Using SOUL.md for Project-Specific Personas

For project-specific personalities that don't need to be in your global config, use SOUL.md:

```bash
# Create a project-level personality
cat > ./SOUL.md << 'EOF'
You are assisting with a machine learning research project.

## Tone
- Academic but accessible
- Always cite relevant papers when applicable
- Be precise with mathematical notation
- Prefer PyTorch over TensorFlow

## Workflow
- Suggest experiment tracking (W&B, MLflow) for any training run
- Always ask about compute constraints before suggesting model sizes
- Recommend data validation before training
EOF
```

This personality only applies when running Hermes from that project directory.

## How Personality Interacts with the System Prompt

The system prompt is assembled in layers (from `agent/prompt_builder.py` and `run_agent.py`):

1. **Default identity**: *"You are Hermes Agent, an intelligent AI assistant created by Nous Research..."*
2. **Platform hint**: formatting guidance based on the platform (CLI, Telegram, etc.)
3. **Memory**: MEMORY.md and USER.md contents
4. **Skills index**: available skills listing
5. **Context files**: AGENTS.md, .cursorrules, **SOUL.md** (personality lives here)
6. **Ephemeral system prompt**: `agent.system_prompt` or `HERMES_EPHEMERAL_SYSTEM_PROMPT` (overlaid)
7. **Session context**: platform, user info, connected platforms (gateway only)

:::info
**SOUL.md vs agent.system_prompt**: SOUL.md is part of the "Project Context" section and coexists with the default identity. The `agent.system_prompt` (set via `/personality` or config) is an ephemeral overlay. Both can be active simultaneously — SOUL.md for tone/personality, system_prompt for additional instructions.
:::

## Display Personality (CLI Banner)

The `display.personality` config option controls the CLI's **visual** personality (banner art, spinner messages), independent of the agent's conversational personality:

```yaml
display:
  personality: kawaii  # Affects CLI banner and spinner art
```

This is purely cosmetic and doesn't affect the agent's responses — only the ASCII art and loading messages shown in the terminal.
