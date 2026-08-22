---
title: "Trajectory Debugger — Analyze session trajectories, cache hit rates, and tokens"
sidebar_label: "Trajectory Debugger"
description: "Analyze session trajectories, cache hit rates, and tokens"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Trajectory Debugger

Analyze session trajectories, cache hit rates, and tokens.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/trajectory-debugger` |
| Version | `0.1.0` |
| Author | Thamer (taljeri), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Telemetry`, `Trajectory`, `Prompt-Caching`, `Debugging`, `Observability` |
| Related skills | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Trajectory Debugger

Inspect session execution transcripts (`transcript.jsonl`), monitor prompt cache efficiency, identify cache-busting prefix mutations, and profile tool performance bottlenecks.

Zero external dependencies: uses Python standard library to parse and audit trajectory logs.

## When to Use

- "Why did this session consume so many tokens?"
- "Analyze prompt cache hit rates for this run"
- "Detect where prompt cache was invalidated in the conversation"
- "Audit tool execution error rates and latency bottlenecks"
- "Inspect step-by-step model reasoning and tool payloads"

Don't use for:
- Live Python / Node.js process debugging (use `python-debugpy` or `node-inspect-debugger`)
- System prompt authoring rules (use `hermes-agent-skill-authoring`)

## Prerequisites

- Standard Python 3.9+ runtime.
- Path to a conversation transcript or trajectory JSONL file (`transcript.jsonl`).

## How to Run

Execute through the `terminal` tool:

```bash
# Full trajectory summary analysis
python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py analyze path/to/transcript.jsonl

# Inspect prompt cache hit rate and detect cache busts
python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py cache path/to/transcript.jsonl --threshold 85 --json

# Tool usage and error rate audit
python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py tools path/to/transcript.jsonl

# Inspect a specific step payload
python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py turn path/to/transcript.jsonl --turn 5
```

## Quick Reference

| Task | Command |
|---|---|
| Summary audit | `python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py analyze <transcript.jsonl> [--json]` |
| Cache efficiency | `python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py cache <transcript.jsonl> [--threshold N]` |
| Tool error rates | `python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py tools <transcript.jsonl> [--json]` |
| Inspect step turn | `python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py turn <transcript.jsonl> --turn N` |

## Procedure

### 1. Audit Overall Token Consumption
1. Run `debug_trajectory.py analyze <transcript.jsonl>`.
2. Check `overall_hit_rate_pct`: A healthy session should maintain 90%+ prompt cache hit rates.
3. Compare total prompt tokens against cached tokens to calculate cost efficiency.

### 2. Locate Cache-Busting Turns
1. Run `debug_trajectory.py cache <transcript.jsonl> --threshold 80`.
2. Review any detected drops where hit rate plummeted.
3. Investigate the identified step index using `debug_trajectory.py turn <transcript.jsonl> --turn <step_index>` to determine if mid-conversation mutation occurred.

### 3. Diagnose Tool Failures
1. Run `debug_trajectory.py tools <transcript.jsonl>`.
2. Identify tools with high error counts or frequent retries.
3. Check if parameter parsing errors or timeouts contributed to excessive token burn.

## Pitfalls

- **Empty usage records:** Mocked or synthetic runs without API token metrics will report 0% cache hits.
- **Provider differences:** Anthropic reports `cache_read_input_tokens`, while OpenAI/DeepSeek report `prompt_cache_hit_tokens`. The debugger automatically normalizes both.

## Verification

Run verification against a sample transcript:
```bash
python3 skills/software-development/trajectory-debugger/scripts/debug_trajectory.py --help
```
