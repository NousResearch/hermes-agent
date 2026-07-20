---
title: "Kubernetes Readonly — Inspect clusters with read-only kubectl helpers"
sidebar_label: "Kubernetes Readonly"
description: "Inspect clusters with read-only kubectl helpers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kubernetes Readonly

Inspect clusters with read-only kubectl helpers.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/kubernetes-readonly` |
| Version | `1.1.0` |
| Author | xyiy001, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `kubernetes`, `k8s`, `kubectl`, `devops`, `observability`, `metrics` |
| Related skills | `webhook-subscriptions` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Skill

JSON-in / JSON-out helper that runs an allowlisted, read-only `kubectl` argv list.
It never uses `shell=True` and never issues mutating verbs.

## When to Use

- Need cluster visibility (`get`, `describe`, `explain`, `top`, version/info) without changing state.
- Pair with incident analysis where mutating workflows are out of scope.

## Prerequisites

- `kubectl` installed and available on `PATH`.
- A valid kubeconfig for the target cluster.
- Prefer invoking the helper via `terminal` from the skill directory.

## How to Run

```bash
python scripts/k8s_readonly.py <<'EOF'
{"op": "get", "resource": "pods", "namespace": "kube-system", "output": "json"}
EOF
```

## Quick Reference

| `op` | kubectl behavior |
|------|------------------|
| `version` | `kubectl version -o json` |
| `cluster_info` | `kubectl cluster-info` |
| `api_resources` | `kubectl api-resources -o wide` (+ optional `api_group`) |
| `explain` | `kubectl explain` (+ optional `recursive`) |
| `get` | `kubectl get` (`output`: `json` / `yaml` / `wide` / `name` / `default`) |
| `describe` | `kubectl describe` |
| `top_pods` / `top_nodes` | `kubectl top pods` / `kubectl top nodes` |

## Procedure

1. Build a single JSON object matching the Pydantic models in `scripts/k8s_models.py`.
2. Pipe it to `scripts/k8s_readonly.py`.
3. Read the JSON result: `ok`, `argv`, `returncode`, `stdout`, `stderr`, `truncated`.
4. If `truncated` is true, stdout/stderr were capped at 2 MiB per stream during streaming capture.

## Pitfalls

- Do not ask this skill to `apply`, `delete`, `create`, `patch`, `exec`, or `port-forward`.
- Oversized API output sets `"truncated": true`; re-query with a narrower selector.
- Timeouts return `"error": "kubectl_timeout"`; raise the cluster/network health question with the user.

## Verification

- Missing kubectl: `{"ok": false, "error": "kubectl_not_found", ...}`.
- Successful `version` returns `ok: true` and JSON in `stdout` when the cluster is reachable.
