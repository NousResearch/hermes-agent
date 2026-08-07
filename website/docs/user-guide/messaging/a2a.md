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

The outbound client tools ship as the `a2a` toolset, **off by default** — enable it with `hermes tools`.

## Outbound: calling other agents

With the `a2a` toolset enabled, the agent gets:

| Tool | What it does |
|---|---|
| `a2a_discover(url)` | Fetch and summarize a peer's Agent Card |
| `a2a_call(agent, message, context_id?, timeout?)` | Send a task, get the reply; multi-turn via `context_id`; `timeout` (seconds) for long tasks |
| `a2a_list()` | Configured peers, saved conversations, metrics |
| `a2a_history(context_id)` | Recall a persisted A2A conversation |
| `a2a_orchestrate(capability, message, mode?)` | Fan a task out to every peer advertising a capability (`all` / `first` / `best`) |

Configure known peers in `config.yaml`:

```yaml
a2a_agents:
  researcher:
    url: "http://research-box.local:9900"
    auth: { type: bearer, token: "..." }
    timeout: 900   # optional, seconds (default 330); raise for slow peers
    capabilities: [web_search, research]
```

Then just ask: *"Ask the researcher agent to summarize today's arXiv postings."* Direct URLs work too — `a2a_call` accepts any A2A endpoint.

## Inbound: being callable

With the platform enabled, Hermes serves:

- **Agent Card** at `GET /.well-known/agent-card.json` (canonical v1.0 path; the legacy `agent.json` also answers) — advertises your agent's name, skills (derived from enabled toolsets), and auth requirements.
- **JSON-RPC 2.0** at `POST /` — canonical v1.0 methods (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, push-notification config CRUD) plus the pre-1.0 path-style aliases (`message/send`, …).
- **SSE streaming** for `SendStreamingMessage`, with spec-correct JSON-RPC-enveloped frames.
- **Push notifications** (webhooks) for long-running tasks, HMAC-SHA256 signed.

Inbound tasks are injected into a **live gateway session** — the same agent, memory, and tools that serve your other channels — and the final reply is returned to the caller as the task result. Conversations are keyed by the A2A `contextId`, so a peer can hold a multi-turn exchange.

Interoperability is verified against the official Python `a2a-sdk` (card resolution, `SendMessage`, streaming).

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
| `A2A_REPLY_TIMEOUT` | `300` | Seconds to wait for the agent's reply |
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
- **Replies time out on long tasks** — raise `A2A_REPLY_TIMEOUT`, or have the caller register a push-notification config and poll `GetTask`.
