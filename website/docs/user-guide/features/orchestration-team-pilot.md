---
title: Controlled team workflow pilot
sidebar_label: Team workflow pilot
---

# Controlled team workflow pilot

A model parent completed the eight checks in a bounded team workflow using two user-configured provider profiles. The run exercised running-worker guidance, a separate review, correction on the original retained worker, a second review and execution of a dependent task after acceptance. This is evidence for these tested routes and this scenario, not a model recommendation or a claim about every provider.

The runtime was `c7fb32110494054fe0ed06bb8267b7794dce8c34`. The successful runner/fixture candidate was `f1c1b890ff20ded4086731d242fa10118bfbb705`; it adds evaluation code over that runtime. Evidence was collected on September 9, 2026 UTC. The [sanitized reports](/evidence/orchestration-team-pilot-c7fb3211.json) preserve both the failed first version and the passing revision.

## What the parent did

1. Discovered the two configured worker profiles and created two dependent Kanban tasks.
2. Started the first worker and sent guidance while its exact run was active.
3. Submitted the implementation for review through a different profile.
4. Requested one correction on the original worker, preserving earlier context.
5. Collected and acknowledged the second review, then accepted the first task.
6. Started the dependent task only after acceptance and acknowledged its successful worker result.

All five worker runs succeeded and their completions were acknowledged. The dependent task's worker completed; that task was **not separately accepted**. This distinction is intentional: a successful worker result does not itself mark a Kanban task done.

## Routes and receipts

| Role | Selected route | Thinking level |
| --- | --- | --- |
| Parent | OpenAI-Codex / gpt-6-astra | high |
| Implementation and retained correction | OpenAI-Codex / gpt-5.6-luna | xhigh |
| Review and dependent execution | ZAI / glm-5.3 | high |

These are the existing authorized accounts used for this pilot. Users can create different profiles and choose other supported providers, models and efforts. No automatic model/interface registry entry was enabled.

Each worker's resolved profile settings matched its transmitted provider, model and effort. The parent receipt also records observed wire settings. Provider-reported identity is recorded separately; it is not independent verification of the model that executed the request. Explicit route-override fields can be null when the parent selected a named profile instead.

## Both samples count

| Sample | Parent turn cap | Time cap | Observed duration | Checks passed | Tool errors |
| --- | --- | --- | --- | --- | --- |
| Version 1 | 24 | 360 seconds | 221.383 seconds | 5 of 8 | 1 |
| Version 2 | 32 | 360 seconds | 301.136 seconds | 8 of 8 | 0 |

The first version reached its turn cap after accepting the first task, leaving the dependent task ready but unstarted. The requested sequence needs 27 public tool operations when performed one at a time. The parent also made one rejected guidance call, then corrected it.

Version two gave the same scenario a 32-turn cap and explicitly described the required guidance target array. It kept the same eight assertions, providers, models, efforts, six-minute limit, two concurrent worker limit and three-iteration child limit. The original failed result remains in the evidence bundle; it is not relabeled as passing.

## What this does and does not prove

The passing sample demonstrates the tested model parent's use of the public styled team and worker tools. Workers had no tools. No live Bot or room recipients participated. It does not prove arbitrary native or MCP tool execution, process-level crash recovery, external delivery guarantees, or all provider combinations; those need their own acceptance evidence.

The runtime's nine focused service/store tests and independent review cover their recorded boundaries. They use synthetic execution, policy and monitoring fixtures; they are not live transport-only tests. Hosted validation is tracked with the PR and is separate from the live sample.

The report observed no duplicate owned run identities or unauthorized actions. That does not establish exactly-once external effects. Cost is partial: included-account cost is recorded, while ZAI billed cost and complete end-to-end cost remain unknown. Token counters retain their reported categories and should not be treated as a normalized cross-provider price comparison.

This pilot does not establish statistical reliability, installation, release, customer readiness or native Codex/Claude harness equivalence. The automatic interface qualification registry remains empty.
