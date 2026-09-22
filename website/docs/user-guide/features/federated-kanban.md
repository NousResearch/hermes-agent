---
sidebar_position: 8
title: Federated Kanban runners
---

# Federated Kanban runners

Federated Kanban is an opt-in runner pool for using multiple computers as one
queue. The Mac and Windows Hermes installations remain separate: each keeps its
own profiles, projects, credentials, workspaces, and Git checkout. Only task
metadata, runner capabilities, leases, and results cross the coordinator
boundary.

Windows does not become a fork. The Windows runner is a standalone compatibility
process that invokes the stock upstream Hermes executable already installed on
that computer.

## Availability model

Each runner advertises capacity only while its local Hermes/Desktop liveness
marker is present. Closing Windows removes only Windows capacity; the Mac can
continue claiming work. Closing the Mac removes only Mac capacity. If both are
closed, no runner claims new federated work. A disappeared runner's lease is
reclaimed after its expiry.

## Enable the lane

Configure the coordinator URL and explicitly enable admission:

```text
hermes fleet init --url https://mac.example/fleet --enable
```

Keep the bearer key in the Hermes secret environment:

```text
HERMES_FLEET_TOKEN=<machine-scoped-key>
```

Create a federated card with no profile assignee:

```text
hermes kanban create "Build LunaBot" --federated --project lunabot --workspace worktree
```

The normal local dispatcher ignores fleet-owned cards. The coordinator matches
the card's model, tool, project, and workspace requirements against live runner
capabilities.

## Run a stock-Windows compatibility runner

The runner requires a liveness marker maintained by the local Hermes/Desktop
launcher. It never copies `.env` files or Git history:

```text
hermes fleet runner \
  --node-id windows \
  --hermes-executable 'C:\\Users\\idrat\\AppData\\Local\\hermes\\bin\\hermes.exe' \
  --liveness-file 'C:\\Users\\idrat\\AppData\\Local\\hermes\\fleet-live' \
  --profile coding-expert \
  --profile task-orchestrator \
  --project LunaBot \
  --project Hermes-Agent \
  --model gpt-5
```

Use `hermes fleet status` to inspect reachability, task leases, and whether the
secret is configured. Status output never prints the bearer key.

The current coordinator uses leases and idempotency, so a network partition is
at-least-once and reclaimable rather than pretending to provide consensus-grade
exactly-once execution.

