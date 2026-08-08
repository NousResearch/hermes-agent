---
title: Evaluation tasks
sidebar_label: Evaluation tasks
---

# Evaluation tasks

Hermes can turn real session traces into sanitized, review-required evaluation
tasks. The task corpus is separate from memory and skills:

- **memory** records durable facts
- **skills** record reusable procedures
- **evaluation tasks** prove that an agent still satisfies a behavior contract

The feature is a CLI surface, not a model tool, so it adds no tool-schema cost
to ordinary conversations.

## Mine a candidate from a session

```bash
hermes evals mine SESSION_ID_OR_PREFIX
```

The default output is:

```text
$HERMES_HOME/evals/candidates/trace-<semantic-digest>.yaml
```

Use an explicit path when the task belongs to a version-controlled corpus:

```bash
hermes evals mine SESSION_ID \
  --output evals/candidates/source-reading-001.yaml
```

Mining performs forced secret and PII redaction regardless of the normal
display-redaction setting. Home/workspace paths are replaced, raw session IDs
and tool-output excerpts are omitted, and provenance uses a SHA-256 digest.
Candidate files are written with mode `0600`, writes are atomic, existing files
are not replaced by default, and symlink targets are refused.

A mined task always has `status: candidate`. Hermes records likely signals such
as failed tool results and corrective user turns, but it does not approve the
task or invent task-specific ground truth. Review the instruction and signals,
replace the generic checks with meaningful checks, add applicable skills, then
set `status: approved`.

Session traces can still contain personal or commercially sensitive prose that
is outside the known PII and credential patterns. Automated redaction is not
complete anonymization. Review every candidate before committing or sharing it.

## Validate a task or corpus

```bash
hermes evals validate evals/candidates/task.yaml
hermes evals validate evals/approved/ --ready
```

Normal validation accepts structurally valid candidates and prints their
warnings. `--ready` exits nonzero unless every task is approved, sanitized, and
has at least one success criterion.

The portable schema is installed at:

```text
hermes_cli/schemas/hermes.eval-task.v1.schema.json
```

## Task format

```yaml
schema_version: 1
id: source-reading-001
status: approved
instruction: Read the supplied source before answering.

source:
  kind: manual
  sanitized: true

environment:
  allowed_tools: [web_extract, session_search]

success:
  deterministic:
    - type: tool_called
      name: web_extract
    - type: final_response_excludes
      value: "I could not open the link"
  judged:
    - Every factual claim is supported by the supplied source.

forbidden:
  - Treat session history as proof of current external contents.

skills:
  - social-media/x-link-resolution-fallback
```

Supported deterministic checks in version 1:

| Check | Required field | Meaning |
|---|---|---|
| `tool_called` | `name` | The recorded run called the named tool |
| `tool_succeeded` | `name` | A named tool result reports `success: true` or `exit_code: 0` |
| `final_response_contains` | `value` | Final response contains the value, case-insensitively |
| `final_response_excludes` | `value` | Final response omits the value, case-insensitively |

Scoring also applies an implicit policy check: every recorded tool call must be
listed in `environment.allowed_tools`. Any call outside that list fails the run.

## Score a recorded run

A runner or external harness can emit this JSON artifact:

```json
{
  "task_id": "source-reading-001",
  "final_response": "The source says ...",
  "tool_calls": [
    {
      "name": "web_extract",
      "result": {"success": true}
    }
  ]
}
```

Score it with:

```bash
hermes evals score task.yaml run.json --output result.json
```

Exit codes:

- `0`: every deterministic check passed and no qualitative judge is required
- `1`: at least one deterministic check failed
- `2`: invalid input or task/run mismatch
- `3`: deterministic checks passed, but a separate evaluator must assess the
  `judged` criteria or verify a `forbidden` behavior constraint

Hermes intentionally does not self-grade qualitative criteria. `forbidden`
rules are surfaced to that evaluator rather than silently ignored. Feed the
criteria and recorded evidence to a fresh evaluator, preferably after all
deterministic checks pass.

## Recommended lifecycle

1. Mine candidates from corrections, failed tools, or incomplete verification.
2. Curate the smallest reproducible behavior contract.
3. Remove irrelevant or sensitive trace material.
4. Add deterministic checks wherever possible.
5. Keep gold labels and verifier details hidden from the candidate agent.
6. Bind the task to relevant skills through `skills`.
7. Run the corpus before promoting skill, prompt, harness, or model changes.
8. Keep generation and evaluation in separate contexts.

The candidate compiler is intentionally conservative. It is a provenance and
curation aid, not an automatic benchmark factory.
