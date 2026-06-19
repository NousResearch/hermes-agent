# 012 - Workflow Progress Ledger Observation

## Context

We looked at OpenCode's todo tool description and compared it with the current
investment-assistant workflow experience.

The useful idea was not the todo tool itself. The useful idea was that a
structured progress list can make an LLM-led multi-step session easier to
audit, explain, resume, and interrupt.

This is an observation note, not an implementation decision.

## What We Noticed

OpenCode's todo tool has several design choices that map well to agentic
workflows:

- It creates tasks proactively for non-trivial multi-step work.
- It keeps one item `in_progress` at a time.
- It marks an item `completed` only after the work is actually done and
  verified.
- It records blockers or follow-ups instead of hiding partial progress.
- It preserves user-provided instructions.

Those rules convert implicit agent activity into an explicit, user-visible
progress surface.

## Why This Matters For The Investment Assistant

Our investment-assistant workflow has multiple layers of state:

- Hermes conversation state
- workflow state machine state
- PydanticAI agent/tool execution state
- persisted artifacts
- user HITL confirmation or revision state

When the user asks "where are we?", raw internal states such as
`BUILDING_UNBIASED_CANDIDATE_POOL` are technically accurate but not useful.

What the user actually wants to know is closer to:

- Did the agent understand my constraints?
- Did it split the investment theme into layers?
- Did it read the Futu screener catalog?
- Did it run storage, optical, power, cloud, software, and chip probes?
- Did it find or miss important candidates like `US.SNDK`, `US.WDC`,
  `US.COHR`, `US.LITE`, and `US.MRVL`?
- Is a missing step required now, or merely a future quality upgrade?
- What is blocked by Futu rate limits, stale data, missing SEC context, or
  unsupported option data?

The current state machine can say which step is active, but it does not
naturally explain progress at this granularity.

## Design Insight

A state machine and a progress ledger answer different questions.

The state machine answers:

- What transitions are allowed?
- What action should run next?
- Where can HITL be inserted?
- How does the workflow resume?

A progress ledger would answer:

- What has actually been done?
- What artifact proves it?
- What is currently being worked on?
- What is blocked or incomplete?
- What can be explained to the user in plain language?

This suggests a useful separation:

- State machine = control plane
- Artifacts = evidence plane
- Progress ledger = explanation plane

## Example From The AI Discovery Experiment

The trace itself showed a clear progress sequence:

1. User constraints were normalized:
   AI theme, US market, high risk, required `QQQ`, `SOXX`, `NVDA`, cash reserve
   5%.
2. The agent read Futu screener catalog files:
   market quote, valuation, financial, technical, dividend, analysis, options,
   concept plates, industry plates, and stock-field enums.
3. The agent built AI theme layers:
   compute semis, semiconductor equipment, memory/storage, optical networking,
   cloud, data-center power/cooling, and software/security.
4. The agent executed Futu probes:
   storage, optical, AI chips, semiconductors, cloud, IDC, electrical equipment,
   power producers, software, and cybersecurity.
5. The agent surfaced important bottleneck candidates:
   `US.SNDK`, `US.WDC`, `US.COHR`, `US.LITE`, `US.MRVL`.
6. Calibration was identified as useful but not required for the first
   discovery-only version.

This sequence is much easier for a user to understand than a raw workflow state
name.

## Possible Future Direction

If we choose to implement this later, we might persist a lightweight progress
record next to workflow artifacts. It would not replace the state machine.

It could simply help Hermes answer status questions in user language:

> We have completed theme-layer discovery and Futu catalog inspection. The
> current discovery run has produced candidates for storage and optical
> networking, including SNDK, WDC, COHR, LITE, and MRVL. Filter calibration is
> not part of V1, but it is recorded as a future quality upgrade.

This should remain optional until we decide it materially improves HITL,
resume, and user-facing workflow clarity.

## Open Questions

- Is this worth implementing, or can artifact summaries solve most of the same
  problem?
- Should this be an investment-assistant-specific concept or a generic Hermes
  workflow concept?
- Should user-facing progress be authored by deterministic summaries, an LLM,
  or both?
- How much detail is useful before it becomes noisy?
- Would this overlap with existing Hermes session/task/todo abstractions?

## Status

Observation recorded. No implementation commitment.
