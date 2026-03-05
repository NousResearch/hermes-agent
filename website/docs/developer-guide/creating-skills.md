---
sidebar_position: 3
title: "Creating Skills"
description: "How to create skills for Hermes Agent — SKILL.md format, guidelines, and publishing"
---

# Creating Skills

Skills are the preferred way to add new capabilities to Hermes Agent. They're easier to create than tools, require no code changes to the agent, and can be shared with the community.

## Should it be a Skill or a Tool?

Make it a **Skill** when:
- The capability can be expressed as instructions + shell commands + existing tools
- It wraps an external CLI or API that the agent can call via `terminal` or `web_extract`
- It doesn't need custom Python integration or API key management baked into the agent
- Examples: arXiv search, git workflows, Docker management, PDF processing, email via CLI tools

Make it a **Tool** when:
- It requires end-to-end integration with API keys, auth flows, or multi-component configuration
- It needs custom processing logic that must execute precisely every time
- It handles binary data, streaming, or real-time events
- Examples: browser automation, TTS, vision analysis

## Skill Directory Structure

Bundled skills live in `skills/` organized by category:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Required: main instructions
│       └── scripts/              # Optional: helper scripts
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

## SKILL.md Format

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

## Skill Guidelines

### No External Dependencies

Prefer stdlib Python, curl, and existing Hermes tools (`web_extract`, `terminal`, `read_file`). If a dependency is needed, document installation steps in the skill.

### Progressive Disclosure

Put the most common workflow first. Edge cases and advanced usage go at the bottom. This keeps token usage low for common tasks.

### Include Helper Scripts

For XML/JSON parsing or complex logic, include helper scripts in `scripts/` — don't expect the LLM to write parsers inline every time.

### Test It

Run the skill and verify the agent follows the instructions correctly:

```bash
hermes --toolsets skills -q "Use the X skill to do Y"
```

## Should the Skill Be Bundled?

Bundled skills (in `skills/`) ship with every Hermes install. They should be **broadly useful to most users**:

- Document handling, web research, common dev workflows, system administration
- Used regularly by a wide range of people

If your skill is specialized (a niche engineering tool, a specific SaaS integration, a game), it's better suited for a **Skills Hub** — upload it to a registry and share it via `hermes skills install`.

## Publishing Skills

### To the Skills Hub

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### To a Custom Repository

Add your repo as a tap:

```bash
hermes skills tap add owner/repo
```

Users can then search and install from your repository.

## Security Scanning

All hub-installed skills go through a security scanner that checks for:

- Data exfiltration patterns
- Prompt injection attempts
- Destructive commands
- Shell injection

Trust levels:
- `builtin` — ships with Hermes (always trusted)
- `trusted` — from openai/skills, anthropics/skills
- `community` — any findings = blocked unless `--force`
