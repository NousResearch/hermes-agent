---
sidebar_position: 8
title: "Gemini delegation routing"
description: "Route bounded leaf work through subscription-backed Gemini while Hermes keeps tools and final authority"
---

# Gemini delegation routing

Gemini delegation routing gives bounded, output-only leaf work to `gemini-3.8-flash-low` through the subscription-backed `agy` CLI. The Sol or Astra parent keeps every Hermes tool, the final judgment, and all side-effect authority.

:::caution Current activation state
The routing code is opt-in and its upstream defaults are disabled. This page does not activate the lane, change profile configuration, restart Hermes, install a cron job, or make a model call. No live activation has occurred as part of this implementation work.
:::

## When Hermes uses Gemini

The router uses a short denylist rather than a workload taxonomy. An enabled profile sends a leaf task to Gemini when the useful result can come back as text or schema-constrained JSON and the supplied data is allowed on the subscription lane.

Examples include rewriting text, extracting records, drafting documentation or code for review, and analyzing supplied source excerpts. The word `deploy` or any other word in the prompt does not change the route by itself. There is no prose classifier.

Three optional task fields control routing:

```json
{
  "route": "auto",
  "data_classification": "standard",
  "output_contract": "text"
}
```

- `route` accepts `auto`, `gemini`, or `sol`.
- Only `data_classification: standard` is eligible for Gemini. `restricted`, `sensitive`, `local-only`, `secret`, `ambiguous`, and any unknown classification fails closed to Sol.
- `output_contract` accepts `text` or `json`.

Hermes routes to Sol or Astra when any hard exclusion applies:

- `role: orchestrator`, because Gemini only handles leaf work;
- `route: sol`;
- `data_classification: restricted`;
- Gemini routing is disabled for the active profile.

`route: gemini` requests the Gemini lane but cannot override these exclusions. If the request conflicts with a data or authority boundary, Hermes records the denial and uses Sol instead.

## Execution and authority boundaries

The Gemini worker is output-only. It receives the normalized goal and context, a fixed worker contract, and an optional JSON schema. It receives no Hermes tools, profile memory, SOUL, parent transcript, environment secrets, or project directory.

Hermes runs `agy` in a new private temporary directory with plan mode, sandboxing, slash commands disabled, JSON output, and a bounded timeout. It does not pass the parent working directory, extra directories, continuation state, or `--dangerously-skip-permissions`. The prompt is sent on stdin instead of the process list.

The worker uses the authenticated Antigravity subscription. There is no Gemini API-key fallback and no paid Gemini API path in this lane. If `agy` is missing, authentication has expired, its response is invalid, a tool action is blocked, or the request times out, Hermes records the failed attempt and falls back once to the configured delegation model.

A Gemini response is a draft or analysis result, not proof that the work is correct or safe. The Sol or Astra parent must still:

- verify material factual claims against the supplied sources;
- run deterministic checks before applying code;
- make the final judgment;
- request human approval for gated side effects;
- perform any approved action with the parent's own tools.

Gemini never approves, executes, or verifies a live side effect.

## Configuration

The shipped configuration is inert:

```yaml
# ~/.hermes/config.yaml
delegation:
  gemini_routing:
    enabled: false
    profiles: []
    default_route: gemini
    default_data_classification: restricted
    command: agy
    model: gemini-3.8-flash-low
    effort: low
    timeout_seconds: 120
    max_input_bytes: 262144
    max_output_bytes: 131072
    fallback_to_delegation_model: true
    receipt_db: routing/gemini-routing.sqlite3
    review:
      enabled: false
      timezone: America/Los_Angeles
      sample_size: 5
      not_before_local: "00:15"
      review_provider: openai-codex
      review_model: gpt-5.6-sol
      alert_target: slack:C0AEMP1AG0H
      alert_workspace_id: T_APPROVED_WORKSPACE
```

Routing starts only when `enabled: true` and the active profile appears in `profiles`. The conservative `default_data_classification: restricted` also remains in force until an operator explicitly changes it for an approved profile. Existing `delegation.provider` and `delegation.model` settings stay in place as the fallback.

Configuration changes take effect in a fresh CLI or gateway session because the delegation tool schema is cached for the session.

### Actions that need separate approval

Creating or reviewing this code and documentation does not activate anything. Each of these live operations needs separate approval:

1. Change profile configuration to enable routing, allow a profile, or change its default data classification.
2. Restart an affected CLI or gateway process.
3. Run the live `agy` wiring canary or an acceptance delegation. These calls use subscription quota.
4. Copy `scripts/gemini_daily_review.py` into `$HERMES_HOME/scripts/`.
5. Create, enable, run, pause, or remove the hourly no-agent cron job.
6. Send a real Slack alert canary.

Do not treat approval to merge documentation as approval for any item in this list.

## Privacy and local receipts

Restricted data is never sent to Gemini and is not stored in the routing receipt database. Mark the task `data_classification: restricted` or use `route: sol` whenever the subscription lane is not allowed to process the payload.

Every started Gemini attempt receives one profile-local receipt, including failures and fallbacks. The default database is:

```text
$HERMES_HOME/routing/gemini-routing.sqlite3
```

The database file and SQLite sidecars use mode `0600`; its parent directory uses mode `0700`. Attempt receipts are metadata-only from the first write. They record route and model identity, statuses, timestamps, usage, SHA-256 digests, byte counts, and duration. The database never stores prompt or context text, Gemini or fallback model output, provider envelopes, conversation IDs, or provider error text. Startup migration strips those fields from existing receipt rows while retaining available hashes and byte counts. When Gemini falls back, the same receipt also records the terminal Sol provider, model, and status.

Raw task text and model output do not go to the receipt database, ordinary gateway logs, or a Slack alert. Logs use metadata such as receipt IDs, status, byte counts, and duration. Slack alerts contain counts, the routing day, pipeline status, a batch ID, and the local receipt path.

Do not delete the receipt database during rollback. It is the evidence needed to inspect routing and review failures.

## Daily review

The review is postdeployment operational assurance, not a model-evaluation program. The approved schedule contract is one hourly no-agent cron tick. On each tick, the reviewer must stay idle until 00:15 local time, then process the previous `America/Los_Angeles` day once. Runtime validation rejects a different timezone or a configured sample size other than five. The job is not installed by this documentation change.

For each routing day, the reviewer:

1. Builds the cohort from every Gemini attempt whose Antigravity process started, including successful, failed, timed-out, malformed, and fallback-triggering attempts.
2. Selects `min(5, eligible_count)` by sorting receipt IDs on `HMAC-SHA256(sample_seed, receipt_id)` with receipt ID as the tie-breaker. This gives five random tasks selected deterministically and replayably when at least five are eligible, all tasks when fewer than five are eligible, and an empty sample when no tasks are eligible.
3. Persists the random seed and selected receipt IDs before review. A retry reuses the same sample.
4. Makes one separate, isolated `openai-codex/gpt-5.6-sol` call for each selected item. Each call has no Hermes tools, memory, SOUL, sibling samples, model fallback, or raw task and response bodies. It checks receipt integrity and execution status; it must mark semantic correctness unreviewable when content would be required.
5. Stores each bounded verdict, its digest, and the final batch status in the local receipt database. Reviewer prose and provider error text are not persisted.

An incomplete review, invalid reviewer response, provider mismatch, timeout, missing receipt, or worker failure fails closed. Recurring failures create evidence and an alert; they do not automatically disable routing.

### Failure-only delivery

A passing batch makes no Slack API call and the cron wrapper leaves empty stdout. A zero-sample pass is silent too.

A sampled-task failure or pipeline failure produces one aggregate Slack alert for the batch. It never sends one alert per sample and never includes prompt or response text. The configured sender must resolve the approved Slack workspace and exact channel before posting. An unknown delivery result is reconciled by the stable batch ID before any retry.

Use local delivery for the no-agent cron job. The review script sends the failure-only alert itself and keeps stdout empty, which prevents Cron from wrapping and sending a duplicate message.

## What this review is not

Routing does not require a prelaunch model bake-off. The launch contract has these explicit exclusions:

- no shadow program;
- no class qualification;
- no benchmark threshold;
- no 200-case or 500-case cohort;
- no predeployment comparative evaluation that blocks launch.

Ordinary tests use fakes to verify routing, sandboxing, fallback, receipts, sampling, review isolation, and alert behavior. A separately approved live canary checks only authentication, model selection, the response envelope, and sandbox wiring. It does not measure Gemini against Sol.

## Rollback

After an approved activation, set `delegation.gemini_routing.enabled` to `false` and restart the affected runtime. Normal Sol delegation remains configured. Pause the review cron job only if the operator also wants reviews to stop; disabling routing does not erase existing receipts or silently remove the job.

See [Subagent delegation](./delegation.md) for the parent and child execution model, and [Scheduled tasks](./cron.md) for no-agent cron behavior.
