---
sidebar_position: 10
title: "Barycenters"
description: "Optional third-party: admit a Hermes tool call in shadow before it runs"
---

# Barycenters

Third-party. Not a Nous product. This page is an optional integration, not a first-party security control.

Hook the **central dispatcher once**. Do not wrap every Hermes tool. Shadow records the decision and still lets the tool run. Enforce is a human flip.

## Install

```bash
pip install barycenters
```

PyPI package [`barycenters`](https://pypi.org/project/barycenters/) 1.0.0.

```python
from barycenters import Barycenters

bary = Barycenters()  # shadow by default
bary.admit("hermes_tool")
```

Decorator form of the same admit:

```python
from barycenters import Barycenters

bary = Barycenters()

@bary.boundary("hermes_tool")
def run_tool(**metadata):
    ...
```

With no `ADMIT_ENDPOINT`, `admit` is an honest AWAITING — not a fabricated pass. The JS client (`bary.govern`) is for OpenClaw / Node, not this Python agent.

What the service does with the call: it **adjudicates** an inert `{ action, namespace, metadata }` (or the Python equivalent) against chartered/bound law. It does not lock GitHub, Stripe, or Fly by itself.

## MCP side door

Not the once-hook. Streamable HTTP:

`https://barycenters-admit.fly.dev/mcp/`

Tools: `admit`, `check_readiness`. A bare GET is **406** (`Accept: text/event-stream` is required) — protocol, not downtime.

Spec: [barycenters.ai/adapters](https://barycenters.ai/adapters)
