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
| `a2a_call(agent, message, context_id?)` | Send a task, get the reply; multi-turn via `context_id` |
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

## Served-agent launchers

The root/default A2A route is unchanged: it runs in the live gateway session and retains that session's Hermes memory and context. A served route with `profile:` and **no** `launcher:` is also unchanged: it uses the legacy Hermes profile launcher.

For a served agent, `launcher:` takes precedence over `profile:` and selects an external runtime. Invalid launcher routes are omitted from the Agent Card and tenant lookup; they do not prevent the root server from starting.

```text
root/default route                         → live Hermes gateway session
served route with launcher.transport=process → one owned subprocess per turn
served route with launcher.transport=pi_rpc  → one worker per (agent slug, context ID)
served route with profile and no launcher    → legacy Hermes profile launcher
```

Launcher specifications are parsed once into immutable route configuration. The only supported external transports are `process` and `pi_rpc`; `pi_rpc` supports only the code-owned `omp` and `feynman` profiles.

### Process launcher

`start` and optional `resume` are non-empty argv arrays, not shell command strings. Each element must be a string. Substitution happens inside an existing argv element and never invokes a shell or word-splits the value. The supported placeholders are `{prompt}`, `{context_id}`, `{session_key}`, `{peer}`, `{agent_slug}`, and `{session_id}`. `{session_key}` is a safe SHA-256 identifier derived from the agent slug and context ID.

Process continuity is exactly one of:

- **Stateless:** no session placeholder; every turn uses `start`.
- **Deterministic:** `{session_key}` in `start`; the configured program owns the deterministic session contract.
- **Opaque:** `resume` contains `{session_id}` and output extracts a session ID. The mapping is scoped by agent slug and A2A context under `<HERMES_HOME>/a2a_launchers/sessions.json`. A valid reply with no optional session ID still completes the current task and the next turn starts fresh.

`cwd` must already exist. `timeout` must be positive. Process output is limited to 4 MiB per stdout/stderr stream; replies are UTF-8 and limited to 1 MiB. Empty replies, decoding/parsing errors, missing required fields, a missing executable, non-zero exit, timeout, or an output limit failure fail the A2A task. The owned process group is terminated on timeout, cancellation, and adapter shutdown so descendants are reaped.

External process launchers default to `inherit_env: false`. They receive only available runtime variables from `PATH`, `HOME`, `USERPROFILE`, `SYSTEMROOT`, `PATHEXT`, `TEMP`, `TMP`, and `TMPDIR`; `pass_env` adds named parent variables and `env` adds literal string overrides. Use `pass_env` for credentials supplied by the parent environment rather than placing them in configuration. The legacy profile launcher retains its compatibility environment behavior.

Text output selects `stdout` or `stderr` using `reply_from`; optional opaque session metadata uses `session_id_from` plus a capture-group `session_id_regex`. Set `strip_session_match: true` only when removing that marker from the same reply stream. JSON output is read from stdout; `reply_field` is required and `session_id_field` is optional. Both fields use simple dot-separated object paths, not JSONPath.

#### Stateless Feynman process example

This verified Feynman invocation is deliberately stateless: Spike 005 showed that one-shot calls do not resume merely by sharing `--session-dir`.

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        agents:
          research:
            name: Feynman Research
            tenant: research
            launcher:
              transport: process
              start: [feynman, --new-session, --thinking, "off", chat, "{prompt}"]
              timeout: 300
              output:
                format: text
                reply_from: stdout
```

### Versioned RPC launchers

`pi_rpc` covers the Pi family of tools (bare Pi, OMP, Feynman, and compatible derivatives). The transport name is retained for configuration compatibility; it is not a promise that arbitrary Pi-compatible executables work — only profiles with proven versioned contracts are supported. `command` must be a non-empty argv array containing separate `--mode` and `rpc` elements; profiles define lifecycle behavior and do not rewrite the command. Workers use strict UTF-8 LF-delimited JSON objects, retain bounded frames (at most 1 MiB each), serialize turns per `(agent_slug, context_id)`, and never expose reasoning, tools, UI, or raw protocol frames as the reply.

| Profile | Verified executable | Startup | Successful completion | Reply | Safe worker reuse after abort |
|---|---|---|---|---|---|
| `omp` | OMP `17.2.12` | Requires `ready`; smaller advertised frame limits apply | `agent_end` | Final assistant `message_end` | Successful abort response **and** `agent_end` |
| `feynman` | Feynman `0.3.11` | Does not require `ready`; first prompt must receive correlated acceptance within `startup_timeout` | `agent_settled` | Last assistant `message_end` before settlement | Successful abort response **and** `agent_settled` |

Prompt acceptance alone is never a completed A2A task. A timeout, malformed frame, protocol rejection, unexpected process exit, or failed UI classification fails the task and evicts the worker. Blocking extension UI requests receive a correlated `extension_ui_response` with `cancelled: true`. Idle workers are reaped after `idle_timeout`; adapter disconnect closes all workers. Cancellation sends correlated `abort`; if the required acknowledgement and profile terminal event do not both arrive, the worker is evicted.

#### Feynman RPC example

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        agents:
          research:
            name: Feynman Research
            tenant: research
            launcher:
              transport: pi_rpc
              protocol_profile: feynman
              command: [feynman, --mode, rpc, --new-session, --thinking, "off"]
              timeout: 300
              startup_timeout: 30
              idle_timeout: 900
```

#### OMP RPC example

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        agents:
          reviewer:
            name: OMP Reviewer
            tenant: reviewer
            launcher:
              transport: pi_rpc
              protocol_profile: omp
              command: [omp, --mode, rpc, --no-session]
              timeout: 300
              startup_timeout: 30
              idle_timeout: 900
```

### Unsupported launcher recipes

Bare Pi has no supported launcher recipe: a successful-turn, versioned contract has not been established. Codex, generic RPC profiles, declarative RPC event mappings, A2A history replay, and automatic prompt/history replay are unsupported. Do not substitute a process recipe as proof of an RPC lifecycle contract.

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
