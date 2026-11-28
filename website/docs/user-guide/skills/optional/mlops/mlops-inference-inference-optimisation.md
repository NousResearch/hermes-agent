---
title: "Inference Optimisation — Hill-climb local LLM serving speed, upstream-maintained"
sidebar_label: "Inference Optimisation"
description: "Hill-climb local LLM serving speed, upstream-maintained"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Inference Optimisation

Hill-climb local LLM serving speed, upstream-maintained.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/inference-optimisation` |
| Path | `optional-skills/mlops/inference/inference-optimisation` |
| Version | `1.0.0` |
| Author | Sybil Solutions contributors (sybil-solutions/inference-optimisation-skill) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `inference`, `llm-serving`, `benchmark`, `tok-s`, `speculative-decoding`, `kv-cache`, `vllm`, `sglang`, `llama-cpp`, `exllama`, `mlx` |
| Related skills | [`llama-cpp`](../../optional/mlops/mlops-inference-llama-cpp.md), [`serving-llms-vllm`](../../optional/mlops/mlops-inference-serving-llms-vllm.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Inference Optimisation (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)
> (MIT). `hermes skills install official/mlops/inference/inference-optimisation`
> pulls the current `skills/inference-optimisation/` tree live from that repo
> (quarantined and scanned like any hub install), so this directory holds only
> the catalog metadata and never goes stale.

The skill makes the agent speed up a local model server with evidence instead
of flag-guessing: measure the baseline, compute the theoretical ceiling
(bytes per token ÷ measured bandwidth for decode, FLOPs ÷ ~50% peak for
prefill), change ONE thing, A/B it between two controls, keep or reject it in a
ledger, and only call a config "best" after a long-context needle gate and a
concurrent soak. It covers single-user and concurrent decode, speculative
decoding (MTP / EAGLE / draft model / n-gram), prefill, KV sizing, TTFT and
startup across vLLM, SGLang, llama.cpp, ExLlama/TabbyAPI, MLX and TensorRT-LLM.

## When to Use

- "Make my local model faster", "tune vLLM / llama.cpp / SGLang", "what's the
  max tok/s this GPU can do", "is MTP / speculative decoding worth it here".
- Publishing a tuned serving recipe that others can reproduce.

Not for choosing or installing an engine from scratch: use `llama-cpp` or
`serving-llms-vllm` first, then come here to tune it.

## What the install contains

| Path | Use |
|---|---|
| `SKILL.md` | the loop, ground rules, lever order, plateau and reporting rules |
| `references/ceilings.md`, `methodology.md`, `levers.md`, `gotchas.md`, `publishing.md` | load with `skill_view(name="inference-optimisation", file_path=...)` when the step needs it |
| `scripts/ceiling.py` | stdlib ceiling calculator from a small JSON of measured hardware + model sizes |
| `scripts/bench.py` | stdlib benchmark for any OpenAI-compatible server; appends to `ledger.jsonl` |
| `scripts/queue.sh`, `scripts/mem.sh` | flock-serialized GPU queue; container memory probe for the watchdog |

## Hermes notes

- Scripts live in the installed skill dir: run them as
  `python3 "$HERMES_SKILL_DIR/scripts/bench.py" run configs/<name>.json` from
  the experiment directory (the one holding `configs/` and `ledger.jsonl`).
- `bench.py` needs no `launch`/`stop` keys to benchmark an already-running
  server; with them it starts and stops the server itself. It only talks to the
  `base_url` you give it; there is no telemetry.
- `queue.sh` defaults its lock to the system temp dir; set
  `QUEUE_LOCK="$HOME/.cache/inference-hillclimb.lock"` (or any shared path) so
  every caller uses the same lock.
- Long queues: upstream detaches them with `setsid nohup … &`. In Hermes, run
  the queue with `terminal(background=true, notify_on_complete=true)` instead,
  so the result comes back to the conversation.
- `mem.sh` reads other users' `/proc/<pid>/fdinfo` on unified-memory hosts and
  may need root; ask the user before using sudo, never embed a password.
- Ground rule 4 already forbids `pkill -f` patterns that can match the
  agent's own shell: stop servers by PID.

## Verification

After install: `python3 "$HERMES_SKILL_DIR/scripts/ceiling.py" "$HERMES_SKILL_DIR/examples/ceiling-ornith-b390.json"`
prints decode, prefill and KV ceiling tables; `bench.py show ledger.jsonl`
after a run lists every recorded experiment.
