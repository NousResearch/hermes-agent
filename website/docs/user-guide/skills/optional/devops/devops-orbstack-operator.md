---
title: "Orbstack Operator — Operate OrbStack containers, machines, and Kubernetes"
sidebar_label: "Orbstack Operator"
description: "Operate OrbStack containers, machines, and Kubernetes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Orbstack Operator

Operate OrbStack containers, machines, and Kubernetes.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/orbstack-operator` |
| Path | `optional-skills/devops/orbstack-operator` |
| Version | `0.1.0` |
| Author | Thomas Oertel (tomraider4720), Hermes Agent |
| License | MIT |
| Platforms | macos |
| Tags | `OrbStack`, `Docker`, `Linux`, `Kubernetes`, `macOS`, `DevOps` |
| Related skills | [`docker-management`](../../optional/devops/devops-docker-management.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# OrbStack Operator Skill

Operate and troubleshoot OrbStack's Docker engine, Linux machines, and local Kubernetes cluster from the command line. Prefer inspection and reversible changes; this skill does not replace application-specific Docker or Kubernetes runbooks.

Official reference: [OrbStack documentation](https://docs.orbstack.dev/).

## When to Use

- Inspect, start, stop, configure, or troubleshoot OrbStack.
- Create, size, clone, export, import, or operate an OrbStack Linux machine.
- Operate OrbStack's Docker context, engine, networking, or volumes.
- Start, stop, inspect, or troubleshoot OrbStack Kubernetes.
- Diagnose `.orb.local`, port-forwarding, VPN, proxy, architecture, or file-sharing behavior.

Don't use for Docker-only work that is independent of OrbStack; load `docker-management` instead. Don't use OrbStack commands against a remote Linux host: OrbStack runs on macOS.

## Prerequisites

- macOS with OrbStack installed. If absent, install with `terminal(command="brew install orbstack")` after the user approves installing software.
- `orb` available on `PATH`; OrbStack also exposes binaries under `~/.orbstack/bin`.
- `docker` for container operations and `kubectl` for Kubernetes operations.
- The user must approve destructive actions and changes that expose services to the LAN.

No API key is required. Login is needed only for licensed OrbStack features.

## How to Run

Use `terminal` for every OrbStack, Docker, and Kubernetes CLI invocation. Start with read-only discovery:

```text
terminal(command="orb version")
terminal(command="orb status")
terminal(command="orb list")
terminal(command="docker context show")
terminal(command="docker info")
```

Use `orb <subcommand> --help` immediately before an unfamiliar or version-sensitive operation. OrbStack's short `orb` command runs Linux commands when its first argument is not a recognized management subcommand, so never guess a subcommand.

## Quick Reference

| Task | `terminal` command |
|---|---|
| Start OrbStack | `orb start` |
| Stop OrbStack | `orb stop` |
| Show status | `orb status` |
| Show live resource use | `orb top` |
| List machines | `orb list` |
| Machine details | `orb info MACHINE` |
| Create machine | `orb create ubuntu:24.04 MACHINE` |
| Run in machine | `orb -m MACHINE COMMAND` |
| Run as root | `orb -m MACHINE -u root COMMAND` |
| Machine boot logs | `orb logs MACHINE` |
| Docker engine logs | `orb logs docker` |
| Restart Docker engine | `orb restart docker` |
| Start Kubernetes | `orb start k8s` |
| Stop Kubernetes | `orb stop k8s` |
| Show configuration | `orb config show` |
| Docker volume backup | `orb docker volume export VOLUME BACKUP.tar.zst` |

Authoritative command references: [CLI](https://docs.orbstack.dev/machines/commands), [headless operation](https://docs.orbstack.dev/headless), [Linux machines](https://docs.orbstack.dev/machines/), [Docker](https://docs.orbstack.dev/docker/), and [Kubernetes](https://docs.orbstack.dev/kubernetes/).

## Procedure

### 1. Inventory before changing state

Run read-only checks with `terminal`:

```text
orb version
orb status
orb list
orb config show
docker context show
docker info
docker ps --all
```

If Kubernetes is in scope, also run:

```text
kubectl config current-context
kubectl get nodes
kubectl get pods --all-namespaces
```

Redact credentials embedded in proxy URLs before reporting configuration. **Complete when** the active Docker context, OrbStack status, relevant machines, and affected workload are identified.

### 2. Bound the requested change

Classify the target as one of: OrbStack service, Linux machine, Docker engine/workload, Kubernetes cluster/workload, network, or storage. Record the current value before any `orb config set`, and state whether the operation restarts a service, interrupts workloads, exposes a port, or deletes data.

Treat these as destructive and require explicit confirmation immediately before execution:

- `orb reset`
- `orb delete docker`, `orb delete k8s`, or `orb delete MACHINE`
- replacing an existing machine during import
- deleting Docker volumes, images, containers, or Kubernetes persistent data

**Complete when** the target, expected impact, rollback path, and confirmation requirement are explicit.

### 3. Operate Linux machines

Inspect the exact local syntax first with `terminal(command="orb create --help")` or the relevant subcommand help.

```text
orb create --memory 4G --cpus 2 --disk 64G ubuntu:24.04 dev
orb info dev
orb -m dev uname -a
orb -m dev -u root id
orb logs dev
```

Use `orb push` and `orb pull` for explicit transfers. macOS files are mounted at `/mnt/mac`; other machines are under `/mnt/machines`. For untrusted workloads, prefer `orb create --isolated` and add only required `--mount` paths; add `--isolate-network` when host and peer access are unnecessary.

Before machine deletion, offer `terminal(command="orb export MACHINE MACHINE.tar.zst")` if the machine contains user data. **Complete when** `orb info MACHINE` reports the intended state and a command inside the machine succeeds.

### 4. Operate the Docker engine

Confirm the active context is `orbstack` before changing workloads:

```text
docker context show
docker context inspect orbstack
docker info
```

Use ordinary Docker and Compose commands for workloads. Use OrbStack extensions for engine lifecycle, debugging, and volume backup:

```text
orb restart docker
orb logs docker
orb debug CONTAINER
orb docker volume clone SOURCE DESTINATION
orb docker volume export VOLUME BACKUP.tar.zst
orb docker volume import BACKUP.tar.zst
```

For a distroless or read-only container, prefer `orb debug CONTAINER` over modifying the image merely to add diagnostics. **Complete when** `docker info` succeeds, the intended containers report the expected state, and the relevant health check or endpoint responds.

### 5. Operate Kubernetes

Verify the current context before any mutation; do not assume `kubectl` targets OrbStack.

```text
orb start k8s
kubectl config current-context
kubectl cluster-info
kubectl get nodes
kubectl get pods --all-namespaces
```

OrbStack Kubernetes shares the container image store. For local images, avoid `:latest` or set `imagePullPolicy: IfNotPresent`. Services and Pod IPs are reachable from macOS; LoadBalancer and Ingress names use `*.k8s.orb.local`.

`orb delete k8s` deletes the cluster and requires explicit confirmation. **Complete when** the current context is proven, the node is Ready, and affected workloads are Ready or their failure is explained from events and logs.

### 6. Diagnose networking and file performance

Use the topology-specific hostname:

| From | To | Hostname or route |
|---|---|---|
| Container | macOS | `host.docker.internal` |
| Linux machine | macOS | `host.orb.internal` |
| Linux machine | forwarded Docker port | `docker.orb.internal` |
| macOS | container | `CONTAINER.orb.local`, published port, or container IP |
| macOS | Linux machine | `MACHINE.orb.local`, `localhost:PORT`, or machine IP |

Docker published ports are reachable from the LAN by default unless `docker.expose_ports_to_lan` is disabled. Machine services listening on `0.0.0.0` or `::` may likewise reach the LAN unless `machines.expose_ports_to_lan` is disabled. Obtain approval before enabling either exposure.

Use named Docker volumes for heavy container I/O; bind mounts traverse macOS file sharing. For subnet conflicts, inspect OrbStack networking and VPN routes before editing Docker address pools. **Complete when** connectivity is tested from both relevant endpoints and exposure is no broader than requested.

### 7. Change configuration safely

Read the existing value, set one key, restart only the affected component if required, then read it back:

```text
orb config get KEY
orb config set KEY VALUE
orb config get KEY
```

Use documented keys from [Settings](https://docs.orbstack.dev/settings). Edit Docker engine JSON through `terminal(command="orb config docker")` only in an interactive session; for automated edits, use `read_file` and `patch` on the user's config after inspecting the whole file, then `terminal(command="orb restart docker")`.

Never expose the Docker API as unauthenticated TCP. A listener such as `tcp://0.0.0.0:2375` grants control of containers and access to macOS data; use SSH or mutually authenticated TLS. **Complete when** the key reads back correctly and the affected service passes its health checks.

### 8. Troubleshoot from evidence

Collect the narrowest relevant evidence:

```text
orb status
orb logs MACHINE
orb logs docker
docker inspect OBJECT
docker logs --tail 200 CONTAINER
kubectl describe RESOURCE NAME
kubectl logs --tail=200 POD
```

Use `terminal(command="orb report")` only when preparing a support bundle and tell the user what will be collected before sharing it. Consult `web_extract` against the official docs for version-sensitive behavior rather than guessing.

**Complete when** the failing layer is isolated, the smallest corrective action is applied, and the original failing check passes.

## Pitfalls

- `orb` is overloaded: it opens a shell, runs a Linux command, or invokes a management subcommand. Use `--help` and an explicit `-m MACHINE -- COMMAND` for automation.
- Docker work uses `docker`, not `orb`; `orb` manages OrbStack and adds a few extensions.
- The active Docker or Kubernetes context may point elsewhere. Verify it before mutation.
- `localhost` from a Linux machine does not reach macOS; use `host.orb.internal`. Docker host networking is different and can share `localhost`.
- Forwarded Docker and machine ports may be LAN-accessible. Treat exposure changes as security-sensitive.
- `orb reset`, `orb delete docker`, and `orb delete k8s` destroy data. A successful backup command is not enough; verify the archive exists before deletion.
- Apple Silicon can run `amd64` workloads through Rosetta, but architecture mismatches may still affect packages and debugging. Set architecture deliberately.
- Kubernetes images tagged `:latest` are pulled by default, which can bypass a local build.
- Editing `~/.orbstack/config/docker.json` requires an engine restart and valid Docker daemon JSON.
- Nested KVM virtualization is unavailable on Apple Silicon.

## Verification

After an operation, run the checks that prove the requested outcome rather than relying on a zero exit code:

- OrbStack: `orb status`
- Machine: `orb info MACHINE` and `orb -m MACHINE COMMAND`
- Docker engine: `docker info`
- Container or Compose service: state, health, logs, and an application-level request
- Kubernetes: current context, Ready node, workload rollout status, events, and an endpoint request
- Configuration: `orb config get KEY` plus the affected service check
- Backup: inspect the archive path and non-zero size before any destructive follow-up

Report the exact object changed, the verification result, and any rollback artifact. If verification fails, stop before further destructive action and return to the inventory step.
