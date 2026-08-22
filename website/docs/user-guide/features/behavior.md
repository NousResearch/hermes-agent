# Behavioral Analysis (`/behavior`)

Understand **how** you build with AI, not just how much. `/behavior` analyzes your session history and produces a behavioral profile: 5-axis numeric scores and 23 personality-driven insight cards.

This is the qualitative complement to `/insights` (which shows quantitative stats like tokens, cost, and tool usage).

## Enable

Behavioral analysis is opt-in. Add to your `config.yaml`:

```yaml
behavior:
  enabled: true
  model: null  # optional: use a cheaper model for LLM cards (null = your current model)
```

## Usage

```
/behavior              # last 30 days
/behavior 7            # last 7 days
/behavior --days 90    # last 90 days
/behavior --source cli # CLI sessions only
```

Or from the command line:

```bash
hermes behavior
hermes behavior --days 7 --source telegram
```

## What You Get

### 5-Axis Behavioral Scores

Each axis is scored 1-10 with a one-line rationale:

| Axis | What It Measures |
|------|------------------|
| Execution Leverage | How much you get done per prompt. Do you delegate or micro-manage? |
| Steering | How often you course-correct mid-task vs let the agent run. |
| Engineering Quality | Do you verify, test, and review, or ship and pray? |
| Product Thinking | Do you plan features, prioritize, and think about user value? |
| Planning | Do you plan before acting or jump straight in? |

### 23 Insight Cards

**LLM-powered narrative cards (4):**

- **Archetype** — "The Orchestrator", "The Architect", "The Speed Runner", etc.
- **Agent relationship** — "Like a design partner", "Like a power tool", etc.
- **Growth edge** — Recommendation targeting your lowest-scoring axis
- **Biggest crash out** — Your worst crash-out message with context

**Deterministic signal-based cards (19):**

- **Prompt style** — Length distribution, percentage under 10 words
- **Go-to prompts** — Your most-repeated short prompts
- **Politeness** — Thank-yous and pleases count
- **Crash outs** — ALL CAPS messages and frustration spikes
- **Model preference** — Which models you reach for most
- **Productivity timing** — Night owl, early bird, or afternoon grinder
- **Shipping timing** — Which day of week you push most
- **Agent parallelism** — Max subagents run in parallel
- **Longest agent run** — Your longest single session
- **Cryptic prompt** — Short, typo-filled, late-night messages
- **Planning habits** — What percentage of sessions start with a plan
- **Skill mastery** — Which skills you load most and how often
- **Memory hygiene** — How actively you manage and reference memory
- **Autonomy level** — How much you let agents run vs steering every step
- **Cross-session memory** — How often you reference prior sessions
- **Tool orchestration** — Diversity and combination of tools used
- **Model effectiveness** — Which models produce your best outcomes
- **Skill ROI** — Which skills correlate with longer sessions
- **Session abandonment** — How often sessions end mid-task

## Score Persistence

Every `/behavior` run stores results in your `state.db`. This enables future features like trend tracking ("your Steering score improved from 4 to 7 this month").

## Privacy

- All analysis runs locally. No data leaves your machine.
- The LLM call uses your configured provider (same as every other agent call).
- Only aggregated signal counts and bounded excerpts (top prompts, crash-out messages) are sent to the LLM, with credentials scrubbed.
- No raw session transcripts are sent to the LLM.
- No telemetry, no third-party uploads.

## Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `behavior.enabled` | bool | `false` | Opt-in gate |
| `behavior.model` | string \| null | `null` | Model override for LLM cards (null = current model) |