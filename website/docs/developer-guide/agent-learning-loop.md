---
title: "Agent Learning Loop"
sidebar_label: "Agent Learning Loop"
---

# Agent Learning Loop

Hermes can save trajectories for training data and debugging. This guide covers
the next layer: curating those trajectories into reviewed learning artifacts
without turning Hermes into an unsafe self-mutating runtime.

A sidecar, plugin, or external pipeline can produce memory proposals, skill
proposals, SFT records, DPO records, and evaluation reports. Importing a
proposal into live memory, skills, or configuration remains an explicit,
reviewed step.

## Why this is separate from core runtime

Learning loops touch user messages, tool output, file paths, logs, and sometimes
credentials accidentally printed by external tools. They can also change future
agent behavior. Start with a sidecar/export boundary, not automatic writes into
the selected profile's `$HERMES_HOME` state.

Hermes resolves state through `HERMES_HOME`; the default home and each named or
custom profile have separate state. Never assume a fixed home directory when
reading from or importing into a live profile.

| System | Responsibility |
|---|---|
| Hermes runtime | Execute tasks, persist sessions, save enabled trajectories |
| Learning sidecar | Read traces, score them, create proposals, export datasets |
| Reviewer | Approve or reject memory, skill, and dataset proposals |
| Training stack | Train models from curated datasets |

## Recommended pipeline

```text
Hermes trajectory JSONL
  -> normalize and validate
  -> evaluate
  -> distill proposals
  -> review queue
  -> approved exports
  -> optional import or training
```

Keep sidecar state project-local:

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

The sidecar should not write into `$HERMES_HOME` unless the user explicitly
requests import of reviewed output into the selected profile.

## Inputs: Hermes trajectories

Hermes trajectories are documented in [Trajectory Format](/developer-guide/trajectory-format).
Each JSONL record contains a conversation plus metadata such as timestamp,
model, and completion state. Batch trajectories can also include `tool_stats`,
`tool_error_counts`, `metadata`, and `partial`. Preserve these as provenance.

## Evaluation signals

Start with deterministic checks before adding LLM judges. Positive evidence
includes completed traces, final answers, passing tests, successful tool calls,
and user confirmation. Negative evidence includes partial traces, unresolved
errors, user corrections, unsupported success claims, and exposed secrets.

Store evidence with the score:

```json
{
  "trace_id": "abc123",
  "score": 82,
  "tags": ["completed", "verified", "user-confirmed"],
  "notes": ["Tests passed and the final answer cited the output."]
}
```

## Distilling proposals

Memory proposals should capture durable facts: stable preferences, recurring
project conventions, hard-to-rediscover environment details, and workflow
constraints. Exclude temporary task progress, raw logs, credentials, commit
IDs, and facts likely to go stale soon. Use declarative language so stored
content cannot be confused with a system instruction.

Skill proposals should describe a reusable procedure, including when to use it,
the required sequence, known pitfalls, verification, and safety boundaries.
Keep generated proposals in the review queue; do not write them directly into
`$HERMES_HOME/skills` or repository skill directories.

## Exporting datasets

SFT records should come from verified, high-quality traces:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

DPO records require a real preference contrast:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Valid preference sources include a rejected answer and its accepted correction,
or two candidates reviewed against the same rubric. Do not invent rejected
responses or train on failed traces as if they were successful.

## Privacy and security checklist

- [ ] No `.env`, API keys, tokens, cookies, SSH keys, or OAuth JSON included.
- [ ] Tool output scanned for credential-like strings and sensitive PII.
- [ ] Private third-party messages removed unless consent exists.
- [ ] Irrelevant local file paths sanitized.
- [ ] Failed traces labeled as failures.
- [ ] Proposals remain pending until reviewed.
- [ ] Approved exports retain provenance.
- [ ] Generated sidecar and export directories are ignored by version control.

## Importing approved artifacts

Importing is separate from exporting:

1. Select the intended profile and resolve its `$HERMES_HOME`.
2. Review the proposal for durability, secrets, and correct scope.
3. Apply it through normal Hermes memory or skill mechanisms.
4. Verify behavior in a new session.
5. Preserve a rollback path.

Avoid background jobs that mutate live Hermes state from every session. A bad
memory or skill can degrade unrelated future work.

## Relationship to optional skills

The optional `agent-learning-loop` skill packages this architecture for agents
performing implementation work. It intentionally ships no sidecar executable;
the implementation belongs in a plugin or external project until real usage
stabilizes the interface.
