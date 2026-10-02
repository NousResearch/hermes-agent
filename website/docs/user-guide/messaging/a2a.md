# A2A (Agent-to-Agent)

[A2A](https://a2a-protocol.org) is the open Agent2Agent protocol (v1.0, stewarded by the Linux Foundation) for communication between independent AI agents. The Hermes A2A plugin works in **both directions**: your agent can call other A2A agents as tools, and other agents can send tasks to your Hermes over HTTP.

It interoperates with any A2A-compliant peer — another Hermes, LangChain, CrewAI, Google ADK agents, or anything built on the official `a2a-sdk`.

## When to use A2A

- **Hermes ↔ Hermes across machines** — let your desktop agent hand tasks to a Hermes on a server, or vice versa, each with its own memory, tools, and credentials.
- **Delegating to specialist agents** — a peer that advertises `web_search`/`research`/`coding` skills on its Agent Card can be discovered and called mid-conversation.
- **Being a callable service** — expose your Hermes so other frameworks' agents can send it tasks.

When you want multiple agents on the **same machine**, prefer [delegation](../features/delegation.md) (in-process subagents) or the [kanban board](../features/kanban.md) (durable multi-profile work queue) — A2A is for crossing process/machine/framework boundaries.

## Enable

```bash
hermes gateway setup      # pick A2A
```

Or in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900
```

The outbound client tools ship as the `a2a` toolset, **off by default** — enable it per platform:

```bash
hermes tools enable a2a --platform cli        # CLI/TUI sessions
hermes tools enable a2a --platform telegram   # or any messaging platform
hermes tools enable a2a --platform a2a        # let inbound A2A tasks call peers (agent chaining)
```

The tools are available in every process type — CLI, TUI, gateway, and cron — without the inbound platform needing to be enabled.

## Outbound: calling other agents

With the `a2a` toolset enabled, the agent gets:

| Tool | What it does |
|---|---|
| `a2a_discover(url)` | Fetch and summarize a peer's Agent Card |
| `a2a_call(agent, message, context_id?, wait?)` | Send a task, get the reply; multi-turn via `context_id`. `wait=false` = accept-then-poll: get the task id at once instead of blocking for the whole job |
| `a2a_status(agent, task_id)` | Poll a task (`tasks/get`) — state + whatever the agent has produced so far |
| `a2a_list()` | Configured peers, saved conversations, metrics |
| `a2a_history(context_id)` | Recall a persisted A2A conversation |
| `a2a_orchestrate(capability, message, mode?)` | Fan a task out to every peer advertising a capability (`all` / `first` / `best`) |

Configure known peers in `config.yaml`:

```yaml
a2a_agents:
  researcher:
    url: "http://research-box.local:9900"
    auth: { type: bearer, token: "..." }
    timeout: 120
    capabilities: [web_search, research]
```

Then just ask: *"Ask the researcher agent to summarize today's arXiv postings."* Direct URLs work too — `a2a_call` accepts any A2A endpoint.

## Inbound: being callable

With the platform enabled, Hermes serves:

- **Agent Card** at `GET /.well-known/agent-card.json` (canonical v1.0 path; the legacy `agent.json` also answers) — advertises your agent's name, skills (derived from enabled toolsets), and auth requirements.
- **JSON-RPC 2.0** at `POST /` — canonical v1.0 methods (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, push-notification config CRUD) plus the pre-1.0 path-style aliases (`message/send`, …).
- **SSE streaming** for `SendStreamingMessage`, with spec-correct JSON-RPC-enveloped frames.
- **Push notifications** (webhooks) for long-running tasks, HMAC-SHA256 signed.
- **`GET /health`** reports `reply_timeout_seconds` and `job_timeout_seconds`, so a peer or operator can read the live windows instead of inferring them from `.env` files.

Inbound tasks are injected into a **live gateway session** — the same agent, memory, and tools that serve your other channels — and the final reply is returned to the caller as the task result. Conversations are keyed by the A2A `contextId`, so a peer can hold a multi-turn exchange.

Interoperability is verified against the official Python `a2a-sdk` (card resolution, `SendMessage`, streaming).

## Long jobs and the reply window

**The rule: a job that may outlive the peer's reply window must not be sent as a blocking `message/send`.**

`message/send` binds the reply to task completion — there is no early ack inside one call, so *"ack in 60 s,
then keep working"* is structurally impossible: the caller gets either the finished reply or nothing. Send
long work **non-blocking** (accept-then-poll), or hand it over as a [kanban card](../features/kanban.md),
which is the durable channel for jobs measured in tens of minutes.

### Three budgets, three different things

| Where | Knob | Bounds | Default |
|---|---|---|---|
| peer (server) | `A2A_REPLY_TIMEOUT` | one *blocking request* — how long the peer holds the HTTP call open | `300` |
| peer (server) | `A2A_JOB_TIMEOUT` | the *work* — how long a task may stay non-terminal once nobody waits on its request | `3600` |
| you (client) | `timeout:` under `a2a_agents.<peer>` | how long your own HTTP call waits | `120` |

Raising the reply window only moves the ceiling; it never removes it. A 60-minute suite outlives any sane
window, so the async path — not a bigger number — is the fix. Both peer-side values are readable live:
`curl http://peer:9900/health` → `{"reply_timeout_seconds": …, "job_timeout_seconds": …}`.

### Accept-then-poll (send it non-blocking)

```bash
curl -X POST http://peer:9900/ \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer ***' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
       "params":{"message":{"messageId":"m1","role":"ROLE_USER",
                 "parts":[{"text":"run the full test suite"}]},
                 "configuration":{"returnImmediately":true}}}'
```

The peer answers at once with a task in `TASK_STATE_WORKING` (its `status.message` says the task was
accepted for asynchronous execution). Poll until it reaches a terminal state:

```bash
curl -X POST http://peer:9900/ -H 'Content-Type: application/json' -H 'Authorization: Bearer ***' \
  -d '{"jsonrpc":"2.0","id":2,"method":"GetTask","params":{"taskId":"task-…"}}'
```

With the Hermes tools the same exchange is `a2a_call(agent="vps", message="…", wait=false)` followed by
`a2a_status(agent="vps", task_id="task-…")`. A2A v0.2 peers spell the flag `configuration.blocking: false`
(the Hermes adapter accepts both); a peer that ignores it will block, and `wait=false` fails loudly after
30 s instead of silently burning the reply window.

### If you block anyway: the result is no longer lost

When the reply window elapses with the agent still working, the peer does **not** fail the task. It hands
back `TASK_STATE_WORKING` with a `status.message` telling you to poll `tasks/get`, and when the work
finishes the reply lands in the task store — audit log, `tasks/get`, push callback — instead of dying with
the socket. The task stays pollable until `A2A_JOB_TIMEOUT`, after which the watchdog fails it with an
explicit budget message.

That makes a late result recoverable over the protocol, but the reply to *your request* is still gone —
which is why the non-blocking path is the rule and this is the safety net.

### Canonical requester pattern

1. **Optional:** read the peer's window from `GET /health` (`reply_timeout_seconds`) before deciding to block.
2. **Send non-blocking** (`wait=false` / `returnImmediately=true`). The task id you get back *is* the
   delivery proof — request it that way instead of asking for an early ack the protocol cannot give.
3. **Poll `tasks/get`** (or `tasks/subscribe` / `message/stream`) until the state is terminal. Keep the
   client timeout short for non-blocking calls: 30 s is generous, since a compliant peer answers at once.
4. **Write deliverables to a file and reply with `path + sha`** — still the right shape: the file survives
   anything, and the task store now carries the same reply, so SSH recovery is a fallback rather than the plan.
5. **Anything longer than `A2A_JOB_TIMEOUT`, or anything that must survive a peer restart** → a kanban
   card, not an A2A call.

## Security model

Secure by default; every widening step is explicit:

- **No token ⇒ localhost only.** The server binds `127.0.0.1`. Remote exposure requires a bearer token **and** an explicit `A2A_HOST`.
- **Per-peer tokens** — `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each peer its own credential; the authenticated name drives rate limiting, trust, and audit.
- **Prompt-injection filtering** — inbound text is filtered and framed as untrusted peer input. Remote peers cannot invoke operator slash commands.
- **Outbound redaction** — credential-shaped strings (API keys, JWTs, tokens) are scrubbed from replies.
- **Audit log** — every exchange appends to `~/.hermes/a2a_audit.jsonl`.
- **Anti-loop** — per-context turn caps stop two agents ping-ponging forever.

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `A2A_PEER_TOKENS` | _(unset)_ | Per-peer credentials `name:token,…` (preferred) |
| `A2A_BEARER_TOKEN` | _(unset)_ | Shared token; identity falls back to caller IP |
| `A2A_HOST` | `127.0.0.1` | Bind host — only widens when a token is set |
| `A2A_PORT` | `9900` | Inbound port |
| `A2A_AGENT_NAME` | hostname-derived | Name on the Agent Card |
| `A2A_PUBLIC_URL` | _(unset)_ | Routable URL advertised on the card (reverse proxies / k8s) |
| `A2A_TRUSTED_PEERS` | _(unset)_ | Allow-list of authenticated identities |
| `A2A_ALLOW_ALL_USERS` | `false` | Allow any authenticated peer (dev only) |
| `A2A_RATE_LIMIT` | `60` | Requests/minute per identity |
| `A2A_MAX_PINGPONG_TURNS` | `5` | Anti-loop turn cap per context (max 20) |
| `A2A_REPLY_TIMEOUT` | `300` | Seconds a *blocking* `message/send` waits for the agent's reply before it stops waiting and hands the caller a pollable `TASK_STATE_WORKING` task. The orphan-task sweep never fails a task before this window elapses (floor 300s), and never while a request is still waiting on it |
| `A2A_JOB_TIMEOUT` | `3600` | Seconds a task may stay non-terminal once nobody is waiting on its request (reply window elapsed, stream client gone, or accepted via `returnImmediately`) before the watchdog fails it. The reply window bounds one request; this bounds the work — and it is the only bound that applies to a parked task |
| `A2A_PUSH_SECRET` | bearer token | HMAC secret for push-notification signing |
| `A2A_ADVERTISED_TOOLSETS` | all registered | Restrict which skills appear on the Agent Card |

Behind a reverse proxy or Kubernetes Service, set `A2A_PUBLIC_URL` (or rely on `X-Forwarded-Host`/`X-Forwarded-Proto`) so the Agent Card advertises a URL peers can actually call back.

## Quick test

```bash
# From another machine / agent:
curl http://your-host:9900/.well-known/agent-card.json

curl -X POST http://your-host:9900/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
       "params":{"message":{"messageId":"m1","role":"ROLE_USER",
                 "parts":[{"text":"What tools do you have?"}]}}}'
```

## Troubleshooting

- **Peers can't reach the card URL** — the card was advertising your bind address; set `A2A_PUBLIC_URL` to the externally routable URL.
- **`401 Unauthorized`** — token mismatch; check `A2A_PEER_TOKENS`/`A2A_BEARER_TOKEN` on the server and the peer's `auth:` block.
- **Server won't bind non-localhost** — by design: set a bearer token first, then `A2A_HOST=0.0.0.0`.
- **Replies time out on long tasks** — do not raise `A2A_REPLY_TIMEOUT` first: see [Long jobs and the reply window](#long-jobs-and-the-reply-window). The fix is a non-blocking send (`returnImmediately` / `wait=false`) or a kanban card; a late reply is stored in the task store either way, and `A2A_JOB_TIMEOUT` is the ceiling on how long it may take. Use a push-notification config (or `GetTask`) if you want the completion pushed to you.
