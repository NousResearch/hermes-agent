---
sidebar_position: 13
title: "Agent2Agent (A2A)"
description: "Authenticated, task-oriented communication between named Hermes peers"
---

# Agent2Agent (A2A)

Hermes can communicate with another Hermes instance over the official A2A Protocol 1.0. The
integration is task-oriented: a request returns a task ID and context ID, and later requests can
continue the same remote context. It is intentionally exposed as a CLI plus an opt-in plugin skill,
so enabling it adds no permanent model tool schema.

## Security model

- Outbound commands accept **configured peer names only**, never arbitrary URLs.
- Every inbound principal and outbound peer has a dedicated bearer credential in the active
  profile's `~/.hermes/a2a/credentials.json`. Tokens are not stored in `config.yaml`.
- Never reuse `API_SERVER_KEY`, a Telegram token, or another service credential for A2A.
- The built-in server binds only to loopback. For another machine, put an authenticated TLS reverse
  proxy or private overlay ingress in front of it and configure that HTTPS URL as the public URL.
  Do not bind the Hermes listener directly to a public interface.
- The configured public URL must use HTTPS, including when the listener itself is local.

## Configure the receiving Hermes

Choose the HTTPS URL that the remote peer will use, while the Hermes listener remains on
`127.0.0.1:8645`:

```bash
hermes a2a setup --public-url https://hermes.example.com/a2a
hermes a2a principal add laptop --profile default
```

The principal command prints its inbound bearer once. Transfer it through a secure channel to the
calling machine. Start or restart the gateway after configuration:

```bash
hermes gateway run
```

## Configure a calling peer

Register the receiving endpoint under a local name. The bearer is read from a hidden prompt by
default:

```bash
hermes a2a peer add norbert https://hermes.example.com/a2a
hermes a2a peer list
```

For automation, use `--token-stdin` and pipe the secret directly from your secret manager instead
of placing it in a command-line argument.

Do not put the bearer in a shell argument, prompt, log, or checked-in file.

## Tasks and contexts

```bash
hermes a2a card norbert --json
printf '%s\n' 'Audit the deployment and report blockers.' | \
  hermes a2a ask norbert --stdin --json
hermes a2a get norbert TASK_ID --json
hermes a2a list norbert --json
hermes a2a cancel norbert TASK_ID --json
```

`ask` continues the last successful context for that named peer by default. Start unrelated work
with `--new-context`, or continue an explicitly returned context with `--context-id CONTEXT_ID`:

```bash
hermes a2a ask norbert 'Follow up on the previous audit.' --json
hermes a2a ask norbert 'Start a separate investigation.' --new-context --json
hermes a2a ask norbert 'Continue this exact thread.' --context-id CONTEXT_ID --json
```

Use `--stdin` for multiline or shell-sensitive content; it is never read implicitly. A positional
message and `--stdin` cannot be combined. `--json` emits the official protobuf JSON shape with
camelCase fields, suitable for scripts. Human output includes task ID, context ID, state, and text
artifacts.

The plugin-local skill is available by its qualified name `a2a-platform:a2a-peer`. It teaches an
agent to use these CLI commands without exposing credentials or adding a permanent A2A tool.
