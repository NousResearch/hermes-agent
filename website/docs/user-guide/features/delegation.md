---
sidebar_position: 7
title: "Subagent Delegation"
description: "Spawn isolated child agents for parallel workstreams with delegate_task"
---

# Subagent Delegation

The `delegate_task` tool spawns child AIAgent instances with isolated context, restricted toolsets, and their own terminal sessions. Each child gets a fresh conversation and works independently — only its final summary enters the parent's context.

## Single Task

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"]
)
```

## Parallel Batch

Up to 3 concurrent subagents:

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## Key Properties

- Each subagent gets its **own terminal session** (separate from the parent)
- **No nested delegation** — children cannot delegate further (no grandchildren)
- Subagents **cannot** call: `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code`
- **Interrupt propagation** — interrupting the parent interrupts all active children
- Only the final summary enters the parent's context, keeping token usage efficient

## Configuration

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  default_toolsets: ["terminal", "file", "web"]  # Default toolsets
```

## When to Use Delegation

Delegation is most useful when:

- You have **independent workstreams** that can run in parallel
- A subtask needs a **clean context** (e.g., debugging a long error trace without polluting the main conversation)
- You want to **fan out** research across multiple topics and collect summaries

:::tip
The agent handles delegation automatically based on the task complexity. You don't need to explicitly ask it to delegate — it will do so when it makes sense.
:::
