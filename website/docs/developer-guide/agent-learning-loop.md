---
title: "Agent Learning Loop"
sidebar_label: "Agent Learning Loop"
---

# Agent Learning Loop

Hermes can save trajectories for training data and debugging. This page describes the missing next layer: how to curate those trajectories into reviewed learning artifacts without turning Hermes into an unsafe self-mutating runtime.

Use this guide when building a sidecar, plugin, or external pipeline that reads Hermes trajectories and produces:

- memory proposals
- skill proposals
- supervised fine-tuning (SFT) records
- preference-tuning (DPO) records
- evaluation reports

The recommended posture is review-first: Hermes may generate evidence and proposals, but importing those proposals into live memory, skills, or configuration should remain an explicit reviewed step.

## Why this is separate from core runtime

A learning loop touches sensitive data: user messages, tool outputs, file paths, environment names, logs, and sometimes credentials accidentally printed by external tools. It can also change future agent behavior. For that reason, the first integration boundary should be a sidecar/export workflow, not automatic writes into `~/.hermes`.

Keep these systems separate:

| System | Responsibility |
|---|---|
| Hermes runtime | Execute the current task, persist sessions, save trajectories when enabled |
| Learning sidecar | Read completed traces, score them, create proposals, export datasets |
| Reviewer | Approve/reject memory and skill proposals |
| Training stack | Fine-tune or preference-train models from curated datasets |

## Recommended pipeline

```text
Hermes trajectory JSONL
  -> normalize / validate
  -> evaluate
  -> distill proposals
  -> review queue
  -> approved exports
  -> optional import or training
```

A minimal local directory layout:

```text
.agent-learning/
  learning.db
  approved/
    memory/*.md
    skill/*.md
exports/
  sft.jsonl
  dpo.jsonl
```

The sidecar may live in a project directory, a plugin repo, or a separate package. It should not write directly to global Hermes state unless the user explicitly asks to import reviewed outputs.

## Inputs: Hermes trajectories

Hermes trajectories are documented in [Trajectory Format](/developer-guide/trajectory-format). At minimum, each JSONL record includes:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."},
    {"from": "tool", "value": "..."}
  ],
  "timestamp": "2026-06-05T00:00:00",
  "model": "provider/model",
  "completed": true
}
```

Batch trajectories may also include `tool_stats`, `tool_error_counts`, `metadata`, and `partial` fields. Preserve those fields as provenance; they are useful for filtering and evaluation.

## Evaluation signals

Start with deterministic checks before adding LLM judges.

Positive signals:

- `completed` is true
- final assistant answer exists
- successful tool calls or test output are present
- user confirmation exists in the trace
- no unresolved tool errors remain

Negative signals:

- `completed` is false or `partial` is true
- traceback, exception, failed command, or timeout without recovery
- user corrections such as "wrong", "no", "actually", or "don't do that"
- assistant claims success without tool evidence
- tool output contains secrets or private data

Example evaluation record:

```json
{
  "trace_id": "abc123",
  "score": 82,
  "tags": ["completed", "verified", "user-confirmed"],
  "notes": ["Tests passed and final answer summarized real output."]
}
```

## Distilling memory proposals

Only propose memory for durable facts that are likely to be useful in future sessions.

Good memory candidates:

- stable user preferences
- recurring project conventions
- environment details that are hard to rediscover
- workflow constraints that prevent repeated mistakes

Do not propose memory for:

- PR numbers, issue numbers, or commit SHAs
- temporary task progress
- raw logs or large transcripts
- credentials, tokens, cookies, or API keys
- facts likely to go stale within a week

Memory proposal shape:

```json
{
  "kind": "memory",
  "title": "Verifier timing preference",
  "content": "User prefers verifier subagents only at final pre-push/readiness gates.",
  "reason": "User explicitly corrected verifier timing.",
  "source_trace_id": "abc123",
  "status": "pending"
}
```

Use declarative language in memory content. Avoid imperative wording that may be re-read as a system instruction later.

## Distilling skill proposals

Create skill proposals when a trace contains a reusable procedure, especially if the agent overcame a non-obvious error.

A useful skill proposal includes:

- when to use it
- exact commands or tool sequence
- pitfalls encountered
- verification steps
- safety boundaries

Skill proposal shape:

```json
{
  "kind": "skill",
  "title": "Safe trajectory curation workflow",
  "content": "# Safe Trajectory Curation\n\n## When to use...",
  "reason": "Trace contains reusable trace-evaluate-review-export workflow.",
  "source_trace_id": "abc123",
  "status": "pending"
}
```

Do not write generated skill proposals directly into `skills/` or `optional-skills/`. Keep them in a review queue first.

## Exporting SFT records

SFT records should come from successful, high-quality traces. A simple export shape is:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Recommended filters:

- include only completed traces by default
- exclude traces with unresolved tool failures
- exclude traces containing unredacted secrets or third-party private data
- preserve metadata separately so bad records can be traced and removed later

Do not train on failed traces as if they were success examples.

## Exporting DPO records

DPO records require a real preference contrast:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Valid sources:

- user rejected an earlier answer and accepted a corrected answer
- a reviewer compared two candidate responses
- an automated judge produced a clear chosen/rejected pair and the pair was reviewed

Invalid sources:

- invented rejected responses
- traces where the chosen output was never verified
- private messages or credentials used as preference examples

## Privacy and security checklist

Before exporting or sharing learning artifacts:

- [ ] No `.env` files, API keys, tokens, cookies, SSH keys, or OAuth JSON are included
- [ ] Tool outputs are scanned for credential-like strings
- [ ] Private third-party messages are removed unless consent exists
- [ ] Local file paths are sanitized when not relevant
- [ ] Failed traces are labeled as failures, not success data
- [ ] Memory and skill proposals remain pending until reviewed
- [ ] Approved exports include source/provenance metadata
- [ ] Generated `.agent-learning/` and `exports/` directories are gitignored

## Importing approved artifacts back into Hermes

Importing is separate from exporting.

Safe import flow:

1. Review the proposal content.
2. Check that it is durable, non-secret, and scoped correctly.
3. Apply through the normal Hermes memory or skill mechanisms.
4. Verify the updated behavior in a new session.
5. Keep a rollback path.

Avoid background jobs that automatically ingest every session and mutate live Hermes state. A bad memory or skill can degrade future behavior across unrelated tasks.

## Relationship to optional skills

The optional `agent-learning-loop` skill packages this workflow for agents to follow during implementation work. This developer guide explains the same architecture from the maintainer/integrator perspective.

The intended contribution path is incremental:

1. document the review-first learning loop
2. provide optional skill guidance
3. build sidecar/plugin implementations outside core
4. move stable, dependency-light interfaces into core only after real usage proves the shape
