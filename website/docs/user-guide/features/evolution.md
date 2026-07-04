---
sidebar_position: 4
title: "Evolution Engine"
description: "Autonomous evaluation and self-improvement — define tasks, benchmark performance, and let the agent close the loop from failure to fix"
---

# Evolution Engine

The Evolution Engine (HAEE) gives Hermes the ability to **evaluate its own work, diagnose failures, and improve autonomously**. It closes the loop that every other agent leaves open: you define what "success" looks like, Hermes attempts the task, and the engine measures, analyzes, and fixes failures — without you having to re-explain or re-prompt.

It is built on research from HarnessX (ICLR 2026), the SIA 3-agent loop, and the Darwin Gödel Machine. The engine is **disabled by default** — enable it when you want systematic evaluation and autonomous improvement.

## What it does

```
You define a task     →   Agent attempts it      →   Engine evaluates
(success criteria)        (trajectory captured)       (score + pass/fail)
                                                          │
                              ┌───────────────────────────┤
                              │ PASS                      │ FAIL
                              ▼                           ▼
                      Baseline recorded          Root cause analyzed
                      (seesaw constraint)        Fix proposed + gated
                                                 Agent retries with fix
```

## Quick Start

### 1. Enable the engine

```bash
hermes evolution enable
```

### 2. Define a task

Create a YAML file describing what success looks like:

```yaml title="my-task.yaml"
name: fix-login-bug
description: "Fix the login redirect loop and verify tests pass"
domain: software-development
complexity: 5
success_criteria:
  - type: test_pass
    command: "pytest tests/test_auth.py -v"
    weight: 1.0
timeout_seconds: 300
max_turns: 20
```

```bash
hermes evolution define-task my-task.yaml
```

### 3. Run it

```bash
# Single run
hermes evolution run fix-login-bug

# Multiple iterations for confidence
hermes evolution run fix-login-bug --iterations 5

# Full benchmark suite
hermes evolution benchmark
```

### 4. Check results

```bash
hermes evolution status
hermes evolution history fix-login-bug --verbose
```

That's it. When tasks fail, the engine analyzes why and proposes fixes. You approve them (or let safe ones auto-apply) and the agent retries.

## Success Criteria Types

You can mix and match five criterion types to define what "done" means:

| Type | What it checks | Example |
|------|---------------|---------|
| `test_pass` | Shell command exits 0 | `"pytest tests/ -q"` |
| `file_exists` | File was created | `"/tmp/report.md"` |
| `content_match` | Regex matches in file | `"Fixed:"` in CHANGES.md |
| `command_output` | Command output contains string | `"Total Revenue"` in stdout |
| `llm_judge` | AI evaluates quality | Rubric: "Report must include pricing and features" |

### Weighted composite scoring

Each criterion has a `weight` (default 1.0). The final score is:

```
score = sum(criterion_score × weight) / sum(weights)
```

The task **passes only when ALL criteria pass** (logical AND). A task with 3 criteria at equal weight that fails 1 criterion scores 67% but is marked FAIL — the engine will analyze and propose fixes.

### Example: multi-criteria task

```yaml
name: deploy-and-verify
description: "Deploy the app and verify it's healthy"
success_criteria:
  - type: test_pass
    command: "docker-compose up -d"
    weight: 0.2
  - type: command_output
    command: "curl -s http://localhost:8080/health"
    expected_output: "ok"
    weight: 0.3
  - type: content_match
    path: "/var/log/app.log"
    pattern: "started successfully"
    weight: 0.3
  - type: file_exists
    path: "/tmp/deploy_checkpoint.txt"
    weight: 0.2
timeout_seconds: 120
max_turns: 15
```

## How Failure Analysis Works

When a task fails, the engine uses a **two-tier analysis**:

### Tier 1: Deterministic rules (instant, always runs)

The engine checks for common failure patterns:
- **Timeout** — task exceeded time or turn limits
- **Premature completion** — agent said "done" but criteria failed
- **Loop detection** — agent repeated the same tool calls without progress
- **Execution errors** — tool calls failed (permissions, missing dependencies, network)
- **Missing tools** — agent needed a capability that doesn't exist

### Tier 2: LLM deep analysis (configurable, uses auxiliary model)

The full trajectory is sent to a separate model (default: DeepSeek) which returns structured findings with:
- Specific failure category and confidence score
- Evidence excerpts from the trace
- Implicated tools and steps
- Suggested fix category (skill, tool, prompt, memory, or environment)

## Improvement Types

The engine can propose six kinds of fixes:

| Action | What it creates | Requires approval? |
|--------|----------------|-------------------|
| `skill_create` | New SKILL.md with procedural knowledge | No (auto-approved) |
| `skill_patch` | Targeted edit to existing skill | No (auto-approved) |
| `tool_create` | New Python tool registered in the tool ecosystem | Yes |
| `tool_modify` | Code-level patch to existing tool | Yes |
| `prompt_modify` | Strategy change in agent guidance | Yes |
| `memory_update` | Persistent knowledge in MEMORY.md | Yes |

Approval requirements are configurable in `config.yaml` under `evolution.safety`.

## Safety: The Five Gates

Every improvement proposal passes through five deterministic checks before it can be applied. **No LLM judgment is trusted for safety decisions** — this is the key insight from HarnessX.

| Gate | What it checks | Failure consequence |
|------|---------------|-------------------|
| **Manifest completeness** | Proposal declares what it changes and why | REJECT — cannot proceed |
| **Content validation** | Valid SKILL.md frontmatter, valid Python syntax | REJECT — cannot proceed |
| **Smoke test** | Code compiles and imports successfully | REJECT — cannot proceed |
| **Size limits** | Skills ≤ 15KB, descriptions ≤ 500 chars | NEEDS_REVIEW — warns but may proceed |
| **Seesaw constraint** | No previously-solved task now fails | NEEDS_REVIEW — warns but may proceed |

The **seesaw constraint** is the most important gate. It prevents "catastrophic forgetting" — where fixing bug A silently breaks bug B. Every time a task succeeds, its score is recorded as a regression baseline. Before any new change is applied, the engine checks that all baselines still pass.

## Harness Variants

Sometimes a fix genuinely helps some tasks but hurts others. Rather than rejecting it outright, the engine can **fork a new harness variant** — an independent configuration snapshot. Tasks that benefit from the change route to the new variant; tasks that would regress stay on the old one.

```bash
hermes evolution variants
```

Shows active variants, their success rates, and which tasks they handle best.

## CLI Reference

```bash
hermes evolution status              # Engine status, config, recent runs
hermes evolution define-task <file>  # Load task from YAML
hermes evolution list-tasks          # List all defined tasks
hermes evolution run <task>          # Run a task with evolution tracking
hermes evolution benchmark [task]    # Run full benchmark suite
hermes evolution history [task]      # Evolution run history with --verbose
hermes evolution variants            # Harness variant statistics
hermes evolution enable              # Turn on the engine
hermes evolution disable             # Turn off the engine
```

### Run options

```bash
# Run 10 iterations for statistical confidence
hermes evolution run my-task --iterations 10

# Verbose mode shows per-criterion results
hermes evolution run my-task --verbose

# Set working directory for evaluation commands
hermes evolution run my-task --cwd /path/to/project

# Benchmark a specific task
hermes evolution benchmark my-task

# Benchmark everything
hermes evolution benchmark
```

## Configuration

```yaml
evolution:
  enabled: false              # Master switch (default: off)
  mode: on_failure            # When to run: on_failure | continuous | manual
  max_iterations: 5           # Max improvement attempts per task

  regression_gate:
    enabled: true
    max_regression_tasks: 20  # Max prior tasks to check for regression

  safety:
    require_approval_for: [tool_create, tool_modify, prompt_modify]
    auto_approve: [skill_create, skill_patch]

  trace_retention_days: 90
  max_trace_size_bytes: 10485760  # 10MB

  # Optional: route evolution to a cheaper model
  auxiliary_provider: deepseek
  auxiliary_model: deepseek-chat
```

## Architecture

The engine is built into the agent runtime — not an external tool. It follows the same patterns as the MemoryManager and plugin system:

```
agent/evolution/
├── evolution_manager.py     # Central orchestrator
├── task_definition.py       # YAML task model + 5 criterion types
├── trajectory_collector.py   # Captures every step during agent execution
├── evaluator.py             # Scores tasks against success criteria
├── failure_analyzer.py      # 2-tier root cause analysis (rules + LLM)
├── improvement_proposer.py  # Generates concrete fixes
├── regression_gate.py       # 5 deterministic safety gates
├── harness_variants.py      # Variant isolation for conflicting fixes
├── evolution_store.py       # SQLite persistence for history + baselines
├── auxiliary_llm.py         # Separate LLM client for analysis/proposals
├── evolution_hooks.py       # Lifecycle hooks (pre/post LLM, tool calls)
└── evolution_tools.py       # Model-facing tools (define_task, status, etc.)
```

**Zero overhead when disabled.** The hooks are no-ops when `evolution.enabled: false`.

## Use Cases

### As a non-technical user
You never see the engine. You just notice Hermes getting better at tasks over time. When it fails at something, it learns and succeeds next time.

### As a developer
Define tasks for your project's quality gates. When the agent fixes a bug, the engine verifies tests pass, CHANGES.md is updated, and no regressions were introduced — before you see the result.

### As a researcher
Benchmark models systematically. Track improvement over weeks. Know exactly which domains and complexity levels need work.

```bash
# Compare two models
hermes model openrouter:claude-sonnet-4
hermes evolution benchmark

hermes model openrouter:qwen-3-5b  
hermes evolution benchmark
```

### As a team lead
Define organization-wide quality tasks. Every agent interaction is measured against them. The evolution history shows you exactly which capabilities are improving and which need attention.

## Relationship to Curator and Skills

The Evolution Engine works **with** the existing self-improvement systems, not against them:

| System | What it does | When it runs |
|--------|-------------|-------------|
| **Skills** | Creates SKILL.md from complex tasks | After 5+ tool calls |
| **Curator** | Archives stale skills, consolidates duplicates | Weekly (idle-gated) |
| **Evolution Engine** | Evaluates, analyzes failures, proposes fixes, gates safety | Configurable (on_failure, continuous, manual) |

The Evolution Engine can create skills that the Curator later maintains. It can also propose tool code and prompt changes that go beyond what the skill system handles.

## Privacy

Evolution traces are stored locally in `~/.hermes/evolution/traces/`. No data leaves your machine unless you configure an external auxiliary model (like DeepSeek) for analysis. You can use a local model via Ollama:

```yaml
evolution:
  auxiliary_provider: ollama
  auxiliary_model: llama3.2
```
