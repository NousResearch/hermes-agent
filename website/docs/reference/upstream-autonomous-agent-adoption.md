---
sidebar_position: 10
title: "Upstream Autonomous-Agent Adoption Notes"
description: "Why Hermes adopted selected autonomous-agent ideas as optional skills instead of core runtime features."
---

# Upstream Autonomous-Agent Adoption Notes

This note records the June 2026 review of several external autonomous-agent projects and the resulting Hermes adoption choices.

## Scope reviewed

- [AgentWrapper/agent-orchestrator](https://github.com/AgentWrapper/agent-orchestrator)
- [paperclipai/paperclip](https://github.com/paperclipai/paperclip)
- [bytedance/deer-flow](https://github.com/bytedance/deer-flow)
- [openai/symphony](https://github.com/openai/symphony)
- `Agent-Analytics/awesome-mu…` as research input only

## Decision summary

Hermes adopted the useful parts as optional autonomous-agent skills and documentation, not as core runtime features.

That keeps Hermes aligned with its existing design goals:

- preserve narrow core scope
- preserve prompt-caching behavior
- prefer plugins, skills, and edge integrations over new resident daemons
- avoid silently absorbing third-party control planes into Hermes core

## Per-project outcome

### Agent Orchestrator

Adopted as an optional skill because the interesting value is the pattern:

- worktree-per-run isolation
- external dashboard/control plane
- CI / PR feedback routed back to the owning agent

Not adopted into core because the daemon, UI, and state model are a separate product.

### Paperclip

Adopted as an optional skill because it offers useful governance ideas:

- budgets
- approvals
- heartbeats
- multi-agent org structure

Not adopted into core because those concepts imply a company-style persistent control plane that would broaden Hermes substantially.

### DeerFlow

Adopted as an optional skill because it is a strong reference for:

- long-horizon agent harnesses
- sandbox boundaries
- separation of operator / worker / memory concerns

Not adopted into core because DeerFlow is an all-in-one runtime platform, not a small extension.

### Symphony

Adopted as an optional skill because it contributes a valuable workflow shape:

- tracker item -> isolated run -> proof-of-work artifact
- spec-first orchestration ideas

Not adopted into core because the repository is an engineering preview and best treated as an external system or design reference.

### Awesome-mu…

Not implemented directly. It is useful as a discovery source, not as a runnable integration target.

## Files added in Hermes

Optional skills:

- `optional-skills/autonomous-ai-agents/agent-orchestrator/SKILL.md`
- `optional-skills/autonomous-ai-agents/paperclip/SKILL.md`
- `optional-skills/autonomous-ai-agents/deer-flow/SKILL.md`
- `optional-skills/autonomous-ai-agents/symphony/SKILL.md`

Generated docs:

- `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-agent-orchestrator.md`
- `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-paperclip.md`
- `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-deer-flow.md`
- `website/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-symphony.md`

Reference surfaces:

- `website/docs/reference/optional-skills-catalog.md`
- `website/sidebars.ts`

## Verification used

- regenerated skill docs with `python3 website/scripts/generate-skill-docs.py`
- confirmed expected files exist
- ran `git diff --check`

## Why this shape fits Hermes

These additions make the upstream ideas discoverable and usable without forcing Hermes to become:

- a resident orchestration daemon
- a dashboard-first control plane
- a duplicate governance engine
- a second long-horizon runtime living inside the core agent

In short: Hermes learned from these projects without becoming them.
