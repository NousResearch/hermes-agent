---
title: "One Three One Rule — Structured decision-making framework for technical proposals and trade-off analysis"
sidebar_label: "One Three One Rule"
description: "Structured decision-making framework for technical proposals and trade-off analysis"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# One Three One Rule

Structured decision-making framework for technical proposals and trade-off analysis. When the user faces a choice between multiple approaches (architecture decisions, tool selection, refactoring strategies, migration paths), this skill produces a 1-3-1 format: one clear problem statement, three distinct options with pros/cons, and one concrete recommendation with definition of done and implementation plan. Use when the user asks for a "1-3-1", says "give me options", or needs help choosing between competing approaches.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/communication/one-three-one-rule` |
| Path | `optional-skills/communication/one-three-one-rule` |
| Version | `1.0.0` |
| Author | Willard Moore |
| License | MIT |
| Tags | `communication`, `decision-making`, `proposals`, `trade-offs` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# 1-3-1 Communication Rule

Structured decision-making format for when a task has multiple viable approaches and the user needs a clear recommendation. Produces a concise problem framing, three options with trade-offs, and an actionable plan for the recommended path.

## When to Use

- The user explicitly asks for a "1-3-1" response.
- The user says "give me options" or "what are my choices" for a technical decision.
- A task has multiple viable approaches with meaningful trade-offs (architecture, tooling, migration strategy).
- The user needs a proposal they can forward to a team or stakeholder.

Do NOT use for simple questions with one obvious answer, debugging sessions, or tasks where the user has already decided on an approach.

## Procedure

1. **Problem** (one sentence)
   - State the core decision or desired outcome in a single concise sentence.
   - Focus on the *what*, not the *how* — no implementation details, tool names, or specific technologies.
   - Keep it tight. If you need "and", you're describing two problems.

2. **Options** (exactly three)
   - Present three distinct, viable approaches labeled A, B, C.
   - Each option gets a brief description, pros, and cons.
   - Options should represent genuinely different strategies, not minor variations of the same approach.

3. **Recommendation** (one option)
   - State which option you recommend and why, based on the user's context and priorities.
   - Be direct — this is your professional judgment, not a hedge.

4. **Definition of Done**
   - List the specific success criteria for the recommended option.
   - These are concrete, verifiable outcomes — not vague aspirations.
   - If the user picks a different option, revise this section to match.

5. **Implementation Plan**
   - Concrete steps to execute the recommended option.
   - Include specific commands, tools, or actions where applicable.
   - If the user picks a different option, revise this section to match.

## Verification

- Response has exactly one Problem sentence.
- Response has exactly three Options (A, B, C) with pros and cons for each.
- Response has a single Recommendation that picks one option with reasoning.
- Definition of Done and Implementation Plan align with the recommended option.
- If the user selects a different option, Recommendation, DoD, and Implementation Plan update accordingly.

## Example

User: "Give me a 1-3-1 for adding retry logic to our API client."

Agent:

**Problem:** API calls to external services fail intermittently under load and need graceful retry handling.

**Options:**

- **Option A: Exponential backoff with jitter in a shared utility.**
  Pros: Simple, well-understood pattern; single implementation shared across all callers.
  Cons: No circuit-breaking; callers still block during retries.

- **Option B: Retry library (tenacity) as a decorator on each endpoint call.**
  Pros: Configurable per-endpoint; built-in support for circuit breakers and rate limits.
  Cons: Adds a dependency; decorator pattern can obscure control flow.

- **Option C: Queue-based async retry with dead letter handling.**
  Pros: Non-blocking; failed calls are tracked and replayable; scales well.
  Cons: Significant complexity increase; requires a queue backend; overkill for low-volume services.

**Recommendation:** Option A. The call volume doesn't justify queue infrastructure, and a shared utility with exponential backoff + jitter covers the failure mode cleanly without new dependencies.

**Definition of Done:**
- `api_retry()` utility handles transient HTTP errors (429, 502, 503, 504).
- Retries up to 3 times with exponential backoff and jitter.
- All external API calls route through the retry utility.
- Logs each retry attempt with endpoint and attempt number.
- Tests cover retry exhaustion, immediate success, and transient-then-success scenarios.

**Implementation Plan:**
1. Create `utils/api_retry.py` with configurable max retries, base delay, and retryable status codes.
2. Add jitter using `random.uniform(0, base_delay)` to prevent thundering herd.
3. Wrap existing API calls in `api_client.py` with the retry utility.
4. Add unit tests mocking HTTP responses for each retry scenario.
5. Verify under load with a simple stress test against a flaky endpoint mock.
