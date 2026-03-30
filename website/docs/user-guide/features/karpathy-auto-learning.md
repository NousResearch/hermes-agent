---
title: Karpathy Auto-Learning
description: Staged post-task learning for Hermes with conservative promotion into memory and skills.
sidebar_label: Karpathy Auto-Learning
sidebar_position: 9
---

# Karpathy Auto-Learning

Karpathy auto-learning is a conservative staged-learning loop for Hermes. After tool-heavy work, Hermes can run a background review, propose durable learnings, store them locally as candidates, and only promote strong candidates into built-in memory or skills when promotion rules allow it.

## Why it exists

Hermes already has:
- built-in memory for compact durable facts
- skills for reusable procedures
- session search for transcript recall

Karpathy auto-learning adds a staging layer between "the model noticed something" and "persist it forever".

That keeps the system safer and more reviewable:
- stage first
- promote second
- keep the feature disabled by default

## Default behavior

The feature is disabled by default.

Config defaults:

```yaml
auto_learning:
  enabled: false
  review_interval: 10
  min_tool_iterations: 4
  candidate_char_limit: 12000
  candidate_max_entries: 200
  promotion_threshold: 0.80
  auto_promote_memory: true
  auto_promote_skills: false
  store_path: ""
  debug: false
```

Notes:
- `store_path: ""` means use the default path under `~/.hermes/auto_learning/`
- memory auto-promotion is allowed by default
- skill auto-promotion is off by default

## CLI control surface

```bash
hermes autolearning status
hermes autolearning enable
hermes autolearning disable
hermes autolearning list
hermes autolearning list --status candidate
hermes autolearning promote <candidate-id>
hermes autolearning reject <candidate-id>
```

## Candidate storage

Candidates are stored locally in a JSONL file under Hermes home by default:

```text
~/.hermes/auto_learning/candidates.jsonl
```

Each entry tracks staged state such as:
- id
- created_at
- status
- category
- summary
- confidence
- evidence
- fingerprint

Statuses:
- `candidate`
- `promoted`
- `rejected`
- `superseded`

## Promotion model

Memory candidates:
- can auto-promote when confidence meets threshold
- reuse Hermes' existing `memory` tool path

Skill candidates:
- are staged by default
- only auto-promote when `auto_promote_skills: true`
- reuse Hermes' existing `skill_manage` validation path

## Safety model

This feature is designed to stay conservative:
- disabled by default
- no direct prompt-memory bloat
- local staging before durable writes
- uses existing memory/skill write surfaces instead of bypassing validation

## Recommended usage

For most users:
1. enable it
2. let it stage candidates
3. inspect candidates through the CLI
4. keep skill auto-promotion off unless you trust the workflow
