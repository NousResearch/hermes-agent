---
sidebar_position: 2
title: "Skills System"
description: "On-demand knowledge documents — progressive disclosure, agent-managed skills, and the Skills Hub"
---

# Skills System

Skills are on-demand knowledge documents the agent can load when needed. They follow a **progressive disclosure** pattern to minimize token usage and are compatible with the [agentskills.io](https://agentskills.io/specification) open standard.

All skills live in **`~/.hermes/skills/`** — a single directory that serves as the source of truth. On fresh install, bundled skills are copied from the repo. Hub-installed and agent-created skills also go here. The agent can modify or delete any skill.

## Using Skills

Every installed skill is automatically available as a slash command:

```bash
# In the CLI or any messaging platform:
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

You can also interact with skills through natural conversation:

```bash
hermes --toolsets skills -q "What skills do you have?"
hermes --toolsets skills -q "Show me the axolotl skill"
```

## Progressive Disclosure

Skills use a token-efficient loading pattern:

```
Level 0: skills_categories()     → ["mlops", "devops"]           (~50 tokens)
Level 1: skills_list(category)   → [{name, description}, ...]   (~3k tokens)
Level 2: skill_view(name)        → Full content + metadata       (varies)
Level 3: skill_view(name, path)  → Specific reference file       (varies)
```

The agent only loads the full skill content when it actually needs it.

## SKILL.md Format

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [python, automation]
    category: devops
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

## Skill Directory Structure

```
~/.hermes/skills/                  # Single source of truth
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   └── assets/                # Supplementary files
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Tracks seeded bundled skills
```

## Agent-Managed Skills (skill_manage tool)

The agent can create, update, and delete its own skills via the `skill_manage` tool. This is the agent's **procedural memory** — when it figures out a non-trivial workflow, it saves the approach as a skill for future reuse.

### When the Agent Creates Skills

- After completing a complex task (5+ tool calls) successfully
- When it hit errors or dead ends and found the working path
- When the user corrected its approach
- When it discovered a non-trivial workflow

### Actions

| Action | Use for | Key params |
|--------|---------|------------|
| `create` | New skill from scratch | `name`, `content` (full SKILL.md), optional `category` |
| `patch` | Targeted fixes (preferred) | `name`, `old_string`, `new_string` |
| `edit` | Major structural rewrites | `name`, `content` (full SKILL.md replacement) |
| `delete` | Remove a skill entirely | `name` |
| `write_file` | Add/update supporting files | `name`, `file_path`, `file_content` |
| `remove_file` | Remove a supporting file | `name`, `file_path` |

:::tip
The `patch` action is preferred for updates — it's more token-efficient than `edit` because only the changed text appears in the tool call.
:::

## Skills Hub

Search, install, and manage skills from online registries:

```bash
hermes skills search kubernetes          # Search all sources
hermes skills install openai/skills/k8s  # Install with security scan
hermes skills inspect openai/skills/k8s  # Preview before installing
hermes skills list --source hub          # List hub-installed skills
hermes skills audit                      # Re-scan all hub skills
hermes skills uninstall k8s              # Remove a hub skill
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json # Export skill config
hermes skills tap add myorg/skills-repo  # Add a custom source
```

All hub-installed skills go through a **security scanner** that checks for data exfiltration, prompt injection, destructive commands, and other threats.

### Trust Levels

| Level | Source | Policy |
|-------|--------|--------|
| `builtin` | Ships with Hermes | Always trusted |
| `trusted` | openai/skills, anthropics/skills | Trusted sources |
| `community` | Everything else | Any findings = blocked unless `--force` |

### Slash Commands (Inside Chat)

All the same commands work with `/skills` prefix:

```
/skills search kubernetes
/skills install openai/skills/skill-creator
/skills list
```
