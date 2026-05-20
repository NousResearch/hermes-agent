---
title: Kanban Control Cockpit
description: Visual fleet, approval, and handshake overview for Hermes Kanban.
---

# Kanban Control Cockpit

The Kanban Control Cockpit is the visual command layer for Hermes multi-agent work. It sits at the top of the Kanban dashboard and summarizes the current board, profile roster, active worker attempts, diagnostics, approval queue, and handshake phase distribution.

## What It Shows

- **Handshake phases**: maps each card's Kanban status into an operator-facing lifecycle: discovered, bound to Kanban, accepted, in progress, waiting for approval, peer review, blocked, and complete.
- **Approval queue**: highlights blocked or review cards that look like they need human approval for destructive actions, production changes, spending, secrets, external writes, or similar gates.
- **Fleet roster**: shows installed profiles, ready/blocked work per profile, and currently active runs.
- **Worker evidence**: the task drawer summarizes the latest run summary, changed files, tests run, verification notes, and last error when worker metadata includes those fields.

## Handshake Flow

Hermes treats Kanban as the durable coordination ledger:

```text
Intent -> canonical path verified -> board selected -> assignee selected -> lease claimed -> approval gates declared -> work starts -> evidence submitted -> peer review if needed -> done or blocked
```

The cockpit does not create a second state machine. It reads existing Kanban tables and renders a safer operator view over the same state used by the CLI, gateway, dispatcher, and workers.

## Approval Gates

The approval queue is intentionally conservative. A card enters the queue when it is blocked or in review and its title, body, result, latest run summary, or last failure mentions a sensitive approval trigger such as destructive deletion, production deploy, credentials, spending, external writes, or owner approval.

Workers should make approval needs explicit in the card body or completion/block summary. Clear approval language makes the cockpit useful and keeps ordinary engineering blockers out of the owner queue.

## Worker Evidence Contract

When a worker completes or blocks a task, it should include a concise summary and metadata shaped like this:

```json
{
  "changed_files": ["web/src/App.tsx", "tests/test_app.py"],
  "tests_run": "pytest tests/test_app.py -v",
  "verification": "Dashboard loaded locally and the card opened without console errors",
  "evidence": {
    "changed_files": ["web/src/App.tsx"],
    "tests_run": "npm run build",
    "verification": "Vite build passed"
  }
}
```

The top-level fields are preferred. The nested `evidence` object is also recognized for workers that already group handoff metadata under one key.

## Operating Pattern

Open the dashboard with:

```bash
hermes dashboard
```

Use the cockpit from top to bottom:

1. Check the board name and total active task count.
2. Scan approvals and diagnostics first.
3. Scan handshake phases for bottlenecks.
4. Check profile load before assigning more work.
5. Open tasks from the cockpit and review worker evidence before marking work complete.
