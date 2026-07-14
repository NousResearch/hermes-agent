---
sidebar_position: 4
title: "Evolution Engine"
description: "Autonomous evaluation and self-improvement — define tasks, benchmark performance, and let the agent close the loop from failure to fix"
---

# Evolution Engine

The Evolution Engine (HAEE) gives Hermes the ability to **evaluate its own work, diagnose failures, and improve autonomously**. It closes the loop that every other agent leaves open: you define what "success" looks like, Hermes attempts the task, and the engine measures, analyzes, and fixes failures — without you having to re-explain or re-prompt.

It is built on research from HarnessX (ICLR 2026), SIA, Darwin Gödel Machine, and HyperAgents (Meta, ICLR 2026). The engine is **disabled by default**. Enable it once — then use Hermes normally. HAEE watches, learns, and improves silently.

**No separate API key needed.** HAEE uses your existing Hermes model through the standard auxiliary client — the same one used for context compression and session search.

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

## Pre-built Tasks

HAEE ships with 10 ready-to-use tasks across 4 domains. Run immediately:

```bash
hermes evolution benchmark
```

| Task | Domain | Complexity | What It Checks |
|------|--------|:----------:|----------------|
| bug-fix-verify | software-dev | 5 | Tests pass + CHANGES.md updated + patch file exists |
| code-review | software-dev | 6 | Security issues found + recommendations given |
| deploy-verify | devops | 4 | Deploy command succeeds + health check passes |
| data-pipeline | data-science | 5 | ETL script runs + output file exists + functions defined |
| api-endpoint | software-dev | 5 | Tests pass + endpoint documented |
| security-audit | security | 7 | Vulnerabilities found + remediations documented |
| document-generation | software-dev | 4 | Docs file exists + overview + examples + API reference |
| refactor-module | software-dev | 7 | Tests pass (no regressions) + refactor notes exist |
| dependency-update | devops | 3 | Tests pass after update + update report exists |
| config-migration | devops | 4 | Config validates + migration logged |

Tasks use `/tmp/` paths and simple commands — no project setup needed. Edit criteria to match your project paths. Benchmark takes <1 second for all 10 tasks.

## How Failure Analysis Works

When a task fails, the engine uses a **two-tier analysis**:

### Tier 1: Deterministic rules (instant, always runs)

The engine checks for common failure patterns AND generates real fixes:

| Failure | Tier 1 Action | What's Produced |
|---------|--------------|-----------------|
| Premature completion | Creates `verify-before-complete` skill | 5-section SKILL.md with real verification procedures |
| Missing tool | Creates `workaround-{task}` skill | Alternative approaches using existing tools |
| Loop detected | Creates `detect-and-break-loops` skill | Pattern recognition + escape strategies |
| Execution error | Creates `troubleshoot-{tool}` skill | Tool-specific diagnosis with common causes |
| Timeout | Creates `time-efficient-{task}` skill | Prioritization + budget management strategies |

Every generated skill includes: YAML frontmatter, "When to Use", "How to Run", "Procedure", "Pitfalls", and "Verification" sections. These are **real, functional skills** — not placeholder templates.

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

## Silent Operation (Auto-Trigger)

You don't need to run commands. HAEE watches your conversations through the auto-trigger:

1. You chat normally: "fix the login bug"
2. Observer records tool sequences across sessions
3. After 3+ similar sessions, a cluster forms
4. When agent forgets verification, HAEE auto-creates a skill
5. Skills evolve recursively — each generation gets smarter
6. For code-level issues, HAEE asks before creating a PR branch

**5 failure types auto-detected** during normal chat:

| Failure Type | What It Detects | HAEE Action |
|-------------|----------------|-------------|
| Missing verification | Agent did work (write/patch) but didn't verify (terminal/read) | Auto-creates `verify-before-complete` skill |
| User correction | User said "no", "wrong", "forgot", "actually..." | Auto-creates targeted troubleshooting skill |
| Loop detected | Same tool called 3+ times consecutively | Auto-creates `detect-and-break-loops` skill |
| Missing output | Agent did work but no files were created | Routes to PR proposer for code-level fix |
| Silent session | No user feedback on a high-confidence cluster | Flags for verification review |

### Nudge Level

Control how intrusive HAEE is:

```yaml
evolution:
  nudge_level: notify   # silent | notify | approve | off
```

| Level | Skills | PR Branches |
|-------|--------|-------------|
| `silent` | Apply silently | Apply silently |
| `notify` (default) | Apply + tell user | **Ask first** |
| `approve` | Apply + tell user | **Ask first** |
| `off` | Nothing | Nothing |

PR branches are never created without permission unless `nudge_level: silent`.

## Recursive Skill Evolution

Skills don't just get created — they evolve. When a skill fails to prevent an issue:

```
Gen 0 (277 bytes): Basic "verify before complete"
Gen 1 (607 bytes): + Concrete verification commands (`echo $?`)
Gen 2 (964 bytes): + Anti-loop guidance + targeted prevention
```

Each generation adds specific procedures from real failure data. Skills get smarter, not just longer. Every patch passes through the 5 safety gates.

## PR Proposer (HyperAgents)

When HAEE detects a code-level issue (tool keeps failing the same way):

1. LLM generates a code fix
2. Staged evaluation validates it (compile → benchmark)
3. Random candidate selection preserves diversity
4. Evolution lineage tracked across generations
5. User approves: `/evolution approve-pr <tool>`
6. Git branch created with full PR body, diff, and safety report

Paths excluded from modification: benchmarks/, tests/, evaluation code — prevents reward hacking.

## CLI Reference

```bash
hermes evolution status              # Rich dashboard: runs, improvement, clusters, PRs
hermes evolution define-task <file>  # Load task from YAML
hermes evolution list-tasks          # List all defined tasks
hermes evolution run <task>          # Run with --iterations N, --verbose
hermes evolution benchmark [task]    # Full benchmark suite (10 pre-built tasks)
hermes evolution history [task]      # Run history with --verbose
hermes evolution variants            # Harness variant statistics
hermes evolution suggest-tasks       # Auto-discover tasks from your usage
hermes evolution improvement         # Statistical proof (Wilcoxon p-value, Cohen's d)
hermes evolution export --all        # Export Atropos training data
hermes evolution pr-status           # Pending code fix proposals
hermes evolution approve-pr <tool>   # Approve and create PR branch
hermes evolution enable | disable    # Toggle engine
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
├── auxiliary_llm.py         # LLM client with sync wrappers (analyze_sync, propose_sync, judge_sync)
├── evolution_hooks.py       # Lifecycle hooks — transport-agnostic (OpenAI/Anthropic/dict)
└── evolution_tools.py       # Model-facing tools (define_task, status, etc.)

# Main API entry point — drives the full evolution loop:
run = manager.run_full_cycle(task, executor)
# executor executes the agent and populates the trajectory
# Engine handles: evaluate → analyze → propose → gate → apply → retry
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
