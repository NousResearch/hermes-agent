---
sidebar_position: 6
title: "Cory Control Plane Worker"
description: "Run Hermes as the Cory request-interpretation worker behind a Cory control plane."
---

# Cory Control Plane Worker

Hermes can run as the execution runtime behind Cory's request-interpretation loop.

This integration is intentionally thin:

- the **control plane** owns request intake, workflow state, approval, and the canonical runtime contract
- **Hermes** owns LLM execution, prompt assembly, retries, and model/provider configuration
- the **Cory worker** is the bridge that claims jobs, runs Hermes, validates output, and writes results back

## What ships in Hermes

This repository now includes:

- `hermes-cory-worker` — a long-running worker process for request interpretation jobs
- `cory_runtime/` — the integration package
- `skills/cory/` — Cory-specific interpretation skills used to build deterministic runtime prompts

The worker currently targets **request interpretation** only. It does not yet execute downstream delivery workflows, review loops, or shipping.

## Runtime flow

```text
control plane
  -> POST /api/internal/request-interpretation-jobs/claim
  -> returns request + prior state + Cory Hermes harness

hermes-cory-worker
  -> builds a deterministic system prompt from the harness + Cory skills
  -> runs Hermes AIAgent
  -> validates the JSON output against the harness contract
  -> POST /complete or /fail back to the control plane
```

## Environment

Required:

- `CORY_CONTROL_PLANE_BASE_URL` or `CONTROL_PLANE_BASE_URL`
- `CORY_CONTROL_PLANE_INTERNAL_API_TOKEN` or `CONTROL_PLANE_INTERNAL_API_TOKEN`

Optional:

- `HERMES_CORY_MODEL`
- `HERMES_CORY_PROVIDER`
- `HERMES_CORY_POLL_INTERVAL_SECONDS`
- `HERMES_CORY_MAX_BACKOFF_SECONDS`
- `HERMES_CORY_MAX_COMPLETION_ATTEMPTS`
- `HERMES_CORY_REQUEST_TIMEOUT_SECONDS`

## Usage

Run continuously:

```bash
hermes-cory-worker
```

Run one job and exit:

```bash
hermes-cory-worker --once
```

Override model or provider for the worker profile:

```bash
hermes-cory-worker --model anthropic/claude-sonnet-4 --provider anthropic
```

## Prompting model

The worker does **not** wait for the model to discover Cory skills dynamically.

Instead it:

1. takes the control-plane harness
2. loads the mapped skill documents from `skills/cory/`
3. assembles a deterministic system prompt
4. asks Hermes to return a single JSON object

This is deliberate. Background workers need stronger contract reliability than an interactive chat session.

## Safety and contract behavior

- The worker runs Hermes with `enabled_toolsets=[]` for interpretation. This step is analysis-only.
- The worker skips Hermes memory and project-context injection so the control-plane harness stays authoritative.
- `interpretationStatus=failed` is never sent through `/complete`; runtime failures go through `/fail`.
- If Hermes returns invalid JSON, the worker retries with a repair prompt before failing the job.

## Current boundary

This is the **first runtime bridge**, not the full Cory product.

What it solves now:

- control-plane jobs can be executed by Hermes
- prompt, skills, and output validation live in versioned code
- deployment can start on a single VPS as a long-running worker

What remains for later phases:

- clarification application loops
- approval-aware downstream planning
- coding-agent dispatch and PR workflow execution
- KM writeback and richer artifact generation
