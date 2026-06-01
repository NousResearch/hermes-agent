# Goal-Driven Development with Hermes Agent

## Overview

Hermes Agent now includes a **production-grade Human-in-the-Loop (HITL) harness** that implements the research-backed GOAL.md specification for autonomous coding agent evaluation. This enables your agent to:

- Define measurable fitness functions (GOAL.md)
- Enforce **monotonic improvement** — never accept regressions
- Score code using **dual scoring** (primary metrics + instrument trust)
- Run **agentic metrics** (Task Completion, Tool Correctness, Step Efficiency, etc.)
- Log every iteration to **iterations.jsonl** for auditability
- Operate in **Continuous / Converge / Supervised** modes via cron
- Receive **production telemetry feedback** for post-deploy learning

## Quick Start

### 1. Enable the Harness

The harness is enabled by default in `~/.hermes/config.yaml`:

```yaml
hitl:
  enabled: true
  dual_scoring: true
  include_deepeval: true
  state_locking: true
  max_iterations_without_improvement: 5
  iterations_log_path: "~/.hermes/iterations.jsonl"
```

### 2. Create a GOAL.md

```python
from agent.hitl_harness import HITLHarness, FitnessFunctionWizard

wizard = FitnessFunctionWizard()
goal = wizard.create_goal_from_template(
    template="production_readiness",
    name="Auth module hardening",
    session_id="auth-hardening-001"
)
```

Or manually write a `GOAL.md` (JSON format) to `~/.hermes/goals/{session_id}/GOAL.md`:

```json
{
  "id": "auth-goal-1",
  "name": "Auth module hardening",
  "description": "Improve auth module production readiness",
  "properties": [
    {"name": "test_pass_rate", "type": "threshold", "target": 1.0},
    {"name": "security_scan_pass", "type": "boolean"},
    {"name": "error_rate", "type": "threshold", "target": 0.01}
  ],
  "composite_weights": {
    "test_pass_rate": 0.5,
    "security_scan_pass": 0.3,
    "error_rate": 0.2
  }
}
```

### 3. Run with the Harness

The harness activates automatically when `hitl.enabled: true`. Every iteration:

1. Evaluates the current state against the GOAL.md
2. Computes dual scores (primary + instrument trust)
3. Applies State Locking — only keeps changes that improve the combined score
4. Logs to `iterations.jsonl`
5. Halts after `max_iterations_without_improvement` stalls

## Modes

### Converge Mode (default)

Run until a target score is reached or improvement stalls:

```python
from agent.hitl_harness.continuous_mode import run_continuous_goal

result = run_continuous_goal(
    goal_path="~/.hermes/goals/auth-hardening-001/GOAL.md",
    mode="converge",
    target_score=0.95,
    max_iterations=50
)
```

### Continuous Mode

Run indefinitely (or up to max_iterations), reporting progress:

```python
result = run_continuous_goal(
    goal_path="...",
    mode="continuous",
    max_iterations=100
)
```

### Supervised Mode

Pauses every 5 iterations for human review:

```python
result = run_continuous_goal(
    goal_path="...",
    mode="supervised",
    max_iterations=100
)
```

## Cron Integration

Schedule autonomous improvement jobs:

```python
from cron.jobs import create_job

job = create_job(
    prompt="Improve test coverage for auth module",
    schedule="0 2 * * *",  # 2 AM daily
    mode="converge",
    name="Auth coverage improvement"
)
```

The scheduler will:
- Auto-enable the HITL harness when `mode` is set
- Run the agent with dual scoring active
- Deliver a markdown report on completion

## Production Telemetry Feedback

Connect post-deploy signals back into the fitness function:

```python
from agent.hitl_harness import TelemetryFeedback

feedback = TelemetryFeedback(agent)
feedback.on_deployment(session_id, commit_hash="abc123")

# Later, when an incident occurs:
feedback.on_incident(session_id, severity="high", description="Auth timeout spike")

# Enrich the goal with production metrics:
feedback.enrich_fitness_goal(goal)

# Get a health report:
print(feedback.generate_feedback_report())
```

The telemetry system tracks:
- Error rates
- P99 latency
- Incident counts
- Rollback flags
- Custom metrics from the observability plugin

## iterations.jsonl Format

Every iteration produces a structured log entry:

```json
{
  "timestamp": "2026-06-01T14:32:10Z",
  "session_id": "auth-hardening-001",
  "iteration": 7,
  "primary_score": 0.72,
  "instrument_trust_score": 0.85,
  "combined_score": 0.759,
  "action": "keep",
  "locked_constraints": ["test_pass_rate"],
  "explanation": "Iteration 7: combined score improved from 0.7450 to 0.7590 (+0.0140). Keeping change."
}
```

Query history:

```python
from agent.hitl_harness import HITLHarness

harness = agent.hitl_harness
history = harness.iterations_logger.get_history(limit=50)
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  FitnessGoal    │────▶│  FitnessEngine  │────▶│  HITLHarness    │
│  (GOAL.md)      │     │  (dual scoring) │     │  (orchestrator) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │         ┌─────────────────┐                   │
         │         │  DeepevalAdapter│◀──────────────────┤
         │         │  (agentic metrics)│                 │
         │         └─────────────────┘                   │
         │                                               │
         │         ┌─────────────────┐                   │
         └────────▶│ TelemetryFeedback│◀─────────────────┘
                   │  (prod signals)  │
                   └─────────────────┘
```

## Configuration Reference

```yaml
hitl:
  enabled: true                    # Master switch
  dual_scoring: true               # Enable primary + instrument trust scores
  include_deepeval: true           # Enable agentic metrics
  state_locking: true              # Enforce monotonic improvement
  max_iterations_without_improvement: 5
  iterations_log_path: "~/.hermes/iterations.jsonl"
  autonomy_level: 1                # L0–L4 graduated autonomy
  multi_agent_verification: true
  observability_layer: 1
  fitness_wizard: true
  confidence_threshold_high: 0.80
  confidence_threshold_low: 0.40
```


## SWE-bench Integration

The harness includes native SWE-bench (real GitHub issue resolution) support via `SWEBenchAdapter`.

### Running SWE-bench Evaluation

```python
from agent.hitl_harness import SWEBenchAdapter

adapter = SWEBenchAdapter(
    repo_path="/path/to/repo",
    dataset="princeton-nlp/SWE-bench_Lite",
    env_type="docker",  # or "local" / "modal"
)

# Evaluate a single issue
result = adapter.evaluate_issue(
    issue_id="django-1234",
    issue_description="Fix the bug where...",
    harness=agent.hitl_harness,
)
print(f"Resolved: {result.resolved}")

# Evaluate a corpus
results = adapter.evaluate_corpus([
    {"issue_id": "issue-1", "description": "Fix bug A"},
    {"issue_id": "issue-2", "description": "Fix bug B"},
], harness=agent.hitl_harness)

print(f"Resolution rate: {results['resolution_rate']:.2%}")
```

### Using SWE-bench in Fitness Functions

The `swe_bench_resolution` template includes SWE-bench resolution rate as a primary metric:

```json
{
  "properties": [
    {"name": "swe_bench_resolution_rate", "type": "threshold", "target": 0.5},
    {"name": "test_pass_rate", "type": "threshold", "target": 1.0},
    {"name": "patch_quality", "type": "threshold", "target": 0.8}
  ],
  "composite_weights": {
    "swe_bench_resolution_rate": 0.5,
    "test_pass_rate": 0.3,
    "patch_quality": 0.2
  }
}
```

Create it via the wizard:

```python
from agent.hitl_harness import FitnessFunctionWizard

wizard = FitnessFunctionWizard()
goal = wizard.create_goal_from_template(
    template="swe_bench_resolution",
    name="Fix Django auth issues",
    session_id="django-auth-swe"
)
```

### Integration with mini_swe_runner.py

The existing `mini_swe_runner.py` provides a standalone SWE-bench runner. The `SWEBenchAdapter` bridges it into the HITL harness so that SWE-bench results feed into the fitness function and iterations.jsonl.

## Troubleshooting

**Harness not initializing?**
- Check `hitl.enabled: true` in config.yaml
- Verify no import errors in logs
- Ensure `~/.hermes/goals/` directory exists

**iterations.jsonl not growing?**
- Verify the goal is loaded: `harness.fitness_wizard.current_goal is not None`
- Check that `on_iteration_end` is being called (requires active conversation loop)

**Low instrument trust scores?**
- The deepeval metrics need a trajectory with tool_calls and messages
- Ensure the agent is actually using tools during the iteration

## Further Reading

- `PROPOSED.md` — Original design document
- `agent/hitl_harness/` — Source code
- `agent/hitl_harness/README.md` — Component reference
