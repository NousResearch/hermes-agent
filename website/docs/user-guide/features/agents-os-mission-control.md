---
sidebar_position: 16
title: "Agents OS Mission Control"
description: "Local-only control-plane UI for Agents OS tasks, approvals, runs, artifacts, workflows, and safety status"
---

# Agents OS Mission Control

Agents OS Mission Control is a local-only operations surface for the Agents OS control plane. It gives you a browser UI for task queues, approval gates, run/event read-back, artifacts, workflows, agent boundaries, and safety status.

It is intentionally separate from the general Hermes web dashboard. Mission Control focuses on the local Agents OS SQLite state under your active `HERMES_HOME`.

## Start it locally

```bash
hermes agents-os web
```

By default it binds to:

```text
http://127.0.0.1:18790/
```

The server is local-only. The `--host` value must be `127.0.0.1` or `localhost`; it does not support public binding.

## Status and launcher payload

Use JSON mode when a launcher, wrapper, or smoke test needs a machine-readable status without starting a server:

```bash
hermes agents-os web --json
```

The payload includes:

- `url` and `routes`
- local-only safety state
- operator UI health contract
- launcher metadata
- existing-server probe for safe reuse
- Windows browser launcher path/command when generated

## Safety model

Mission Control is designed to stay bounded:

- no deploy
- no gateway restart
- no public bind (`0.0.0.0` is rejected)
- no credential preview
- no provider/auth mutation
- approval-gated actions remain approval-gated
- external/public/financial/destructive/security-sensitive work is represented as an approval draft, not executed automatically

Stopping Mission Control means stopping only the Mission Control web process. It does not require restarting the Hermes gateway.

## Typical smoke check

```bash
curl -fsS http://127.0.0.1:18790/api/status
```

A healthy local instance returns `ok: true` and an `operator_ui` block for `Agents OS Mission Control`.
