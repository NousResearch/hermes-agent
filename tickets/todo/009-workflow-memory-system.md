# 009 - Workflow Memory System

## Goal

Build a structured memory system for the investment assistant workflow so
multi-agent steps share intent, constraints, guidance, intermediate artifacts,
tool traces, and human decisions without relying on chat context.

## Background

The current MVP workflow is moving toward multiple PydanticAI agents:

- Theme Discovery Agent
- Filter Calibration Agent
- Candidate Triage Agent, including Futu lightweight enrichment and coarse
  screening
- Deep Research Agent, including structured financial numbers and selective
  filings/events research

Thesis synthesis and portfolio-map architecture are intentionally paused until
the discovery, triage, and deep-research artifacts are stable.

These agents should not pass large natural-language conversations to each
other. They need a recoverable, auditable artifact memory layer that can support
Hermes routing, HITL confirmation, retries, and later Slack/Feishu usage.

## Scope

- Add workflow-level memory artifacts:
  - `normalized_user_intent`
  - `investment_constraints`
  - `workflow_guidance`
  - `assumptions_and_open_questions`
  - `human_decisions`
- Add artifact memory conventions:
  - each artifact records `artifact_id`, `session_id`, `type`, `version`,
    `producer`, `input_artifact_ids`, `payload`, `summary`, `freshness`, and
    `warnings`
  - each agent reads only the artifacts it needs
  - each generated artifact records its upstream dependencies
- Add run/tool trace storage:
  - `agent_runs`
  - `tool_calls`
  - calibration trials and rejected trials
- Define per-agent read contracts:
  - Discovery reads `normalized_user_intent`, `investment_constraints`,
    and discovery guidance.
  - Calibration reads discovery artifacts, constraints, and calibration
    guidance.
  - Candidate Triage reads discovery artifacts and Futu lightweight enrichment
    artifacts, then writes shortlist/watchlist/defer/reject queues.
  - Deep Research reads triage queues and structured financial context, then
    decides which candidates need filings, SEC data, earnings, events, or news.
  - Ensure workflow resumes by `session_id` and current state rather than by
  Hermes chat memory.

## Non-Goals

- Do not use long free-text "general memory" as a hidden prompt.
- Do not let memory artifacts inject new investment conclusions.
- Do not use memory to preserve stale market facts as authoritative data.
- Do not let future downstream portfolio-map agents read raw unbounded tool
  output.

## Acceptance Criteria

- A workflow session can be resumed from SQLite using `session_id`.
- User intent, constraints, workflow guidance, assumptions, and human decisions
  are stored as separate typed artifacts.
- Each agent invocation records the artifact ids it read and wrote.
- Tool calls are persisted with arguments, timing, result summary, warnings,
  and error details.
- Calibration trials record filters, returned counts, sample symbols, selected
  filter, rejected trials, and selection reason.
- Future downstream portfolio-map agents are prevented by validation from
  introducing symbols not present in curated candidate artifacts.
- Tests prove that workflow output can be reconstructed from artifacts without
  relying on prior Hermes conversation context.

## Notes

This ticket is about memory architecture, not recommendation logic. The memory
system should support AI-authored investment reasoning, but deterministic code
must still own persistence, freshness checks, dependency tracking, and
validation.
