---
title: "Sandbox Runner — Run untrusted code in disposable isolated sandboxes"
sidebar_label: "Sandbox Runner"
description: "Run untrusted code in disposable isolated sandboxes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sandbox Runner

Run untrusted code in disposable isolated sandboxes.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/sandbox-runner` |
| Version | `0.1.0` |
| Author | Thamer (taljeri), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Sandbox`, `Docker`, `Security`, `Isolation`, `DevOps` |
| Related skills | [`sdlc-review`](/docs/user-guide/skills/bundled/devops/devops-sdlc-review), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Sandbox Runner

Execute untrusted code, external benchmark suites, third-party repositories, and destructive scripts inside ephemeral throwaway sandboxes. Inspired by DeepSeek Harness's minimal evaluation sandboxes, this skill prevents unverified code from mutating the host environment.

Zero external dependencies: uses Docker when available, or falls back to temporary isolated workspaces with strict execution timeouts and automated cleanup.

## When to Use

- "Run this untrusted Python script in a disposable sandbox"
- "Execute benchmark evaluation in an isolated environment"
- "Test a script without touching my local filesystem"
- "Install and test experimental dependencies safely"

Don't use for:
- Long-running server daemons that must persist across sessions
- Basic local file edits that already run safely via Hermes tools

## Prerequisites

- Standard Python 3.9+ runtime.
- Optional: Docker daemon running for hardware-isolated container execution (falls back to isolated temp workspace if Docker is absent).

## How to Run

Execute commands via the `terminal` tool using the bundled runner:

```bash
# Run inline command in isolated sandbox with network disabled
python3 skills/devops/sandbox-runner/scripts/sandbox.py run "python3 -c 'import math; print(math.pi)'" --network none --json

# Execute a standalone script file inside a Python container
python3 skills/devops/sandbox-runner/scripts/sandbox.py exec-file path/to/script.py --image python:3.11-slim

# Check available sandbox engines
python3 skills/devops/sandbox-runner/scripts/sandbox.py check
```

## Quick Reference

| Task | Command |
|---|---|
| Run command in sandbox | `python3 skills/devops/sandbox-runner/scripts/sandbox.py run "<cmd>" [--image IMG] [--mount DIR] [--json]` |
| Execute script file | `python3 skills/devops/sandbox-runner/scripts/sandbox.py exec-file <script> [--image IMG] [--timeout SEC]` |
| Check engine status | `python3 skills/devops/sandbox-runner/scripts/sandbox.py check [--json]` |
| Prune containers | `python3 skills/devops/sandbox-runner/scripts/sandbox.py prune` |

## Procedure

### 1. Check Engine Availability
1. Run `sandbox.py check` to inspect whether Docker container isolation is active or local workspace isolation will be used.

### 2. Configure Isolation Constraints
1. Select appropriate container image (default: `python:3.11-slim` or custom node/golang images).
2. Set network policy: Use `--network none` for strict air-gapped execution of untrusted scripts.
3. Set timeout limits (default: 120 seconds) to prevent infinite loops.

### 3. Execute and Inspect Output
1. Run the target command or script file via `sandbox.py run` or `sandbox.py exec-file`.
2. Inspect JSON output containing `exit_code`, `stdout`, `stderr`, and `elapsed_seconds`.
3. Auto-cleanup tears down containers and temporary workspaces automatically upon completion.

## Pitfalls

- **State persistence:** Containers and isolated workspaces are ephemeral and destroyed after each run. Mount a workspace explicitly via `--mount <dir>` if outputs must be written back.
- **Root permissions:** Avoid executing commands with root privileges when mounting host directories.

## Verification

Verify the sandbox runner by executing a self-test:
```bash
python3 skills/devops/sandbox-runner/scripts/sandbox.py run "echo 'sandbox verified'"
```
