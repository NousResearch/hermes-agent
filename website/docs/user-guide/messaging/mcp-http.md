# MCP HTTP (chat with Hermes from Claude Code)

The `mcp_http` platform serves a [Streamable HTTP MCP](https://modelcontextprotocol.io) endpoint from inside the gateway so a **remote MCP client** — Claude Code, Codex, or any MCP host — can hand a task to your *running* Hermes and get the reply back. The Hermes that answers is the long-lived gateway agent with its memory, skills, tools and credentials, not a throwaway subprocess.

Each client presents a bearer token that maps to a **name**; Hermes sees that authenticated name (not whatever the prompt claims) and every client gets its own namespaced conversations.

## Why not `hermes mcp serve`?

`hermes mcp serve` exposes Hermes' **tools** to an MCP host, which then drives them with its own model. `mcp_http` exposes the **agent** as a chat peer: the host says what it wants, Hermes does the work with its own reasoning and context, and the host reads the answer. They are complementary.

## Enable

```bash
hermes gateway setup      # pick MCP HTTP → configure per-client tokens
```

Secrets go in `~/.hermes/.env`:

```bash
# Preferred: one token per client. The name is the identity Hermes sees.
MCP_HTTP_PEER_TOKENS="laptop-claude:tok-abc123,ci-bot:tok-def456"
# Or a single shared token (identity falls back to the caller IP):
# MCP_HTTP_BEARER_TOKEN="..."
```

Everything else lives in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    mcp_http:
      enabled: true
      extra:
        port: 8765
        host: 0.0.0.0                            # only honoured when a token is set
        public_url: https://hermes.example.net/mcp   # URL you give clients (tunnel / proxy)
        # allowed_hosts: [hermes.example.net]    # extra Host headers to accept
        # trusted_peers: [laptop-claude]          # allow-list of authenticated names
        # rate_limit: 30                          # chat() starts per minute per identity
        # reply_timeout: 300
```

Existing `MCP_HTTP_PORT`, `MCP_HTTP_HOST`, `MCP_HTTP_PUBLIC_URL`, `MCP_HTTP_ALLOWED_HOSTS`, `MCP_HTTP_TRUSTED_PEERS`, `MCP_HTTP_RATE_LIMIT` and `MCP_HTTP_REPLY_TIMEOUT` env vars still work and take precedence over `extra`.

Then restart the gateway. `GET http://127.0.0.1:8765/health` (no auth) reports the bind address, the advertised URL, and whether the gateway loop is attached.

## Connect from Claude Code

On the client machine:

```bash
claude mcp add --transport http hermes https://hermes.example.net/mcp \
  --header "Authorization: Bearer tok-abc123"
```

The client must be able to reach the port — same network, a Tailscale tailnet, an SSH tunnel, or a reverse proxy/tunnel you control. Do not expose it to the public internet without one.

## The chat / wait_reply loop

Hermes turns often take minutes, so `chat` returns immediately and the client polls:

| Tool | What it does |
|---|---|
| `whoami()` | Your authenticated name |
| `new_conversation()` | Mint a fresh conversation id (clean context) |
| `chat(message, conversation_id?)` | Start a turn; returns `accepted …` at once |
| `wait_reply(conversation_id, timeout_s≤55)` | Long-poll. While working returns `working … last activity: <tool activity>`; when finished returns `done` + the reply |
| `status(conversation_id)` | Working/idle, elapsed time, recent tool activity, last reply |
| `cancel(conversation_id)` | Interrupt the running turn |
| `history(conversation_id, limit)` | Recent exchanges — persisted, survives gateway restarts |

Omit `conversation_id` to use your stable per-identity default thread; reuse an id to continue with full context. The MCP server instructions tell the client model to keep polling rather than give up.

## Security model

Secure by default; every widening step is explicit:

- **No token ⇒ loopback only.** With neither `MCP_HTTP_PEER_TOKENS` nor `MCP_HTTP_BEARER_TOKEN` set, the server binds `127.0.0.1` even if `host: 0.0.0.0` is configured.
- **Bearer required** for everything except `/health`; unauthenticated requests get `401` before reaching the MCP app.
- **Per-token identity.** The name in `MCP_HTTP_PEER_TOKENS` is what Hermes sees; the inbound message is framed as untrusted input from that named agent, and operator slash commands are disabled for it.
- **Per-peer conversation namespacing.** Conversation ids are prefixed with the peer name; a client cannot read, poll or cancel another client's thread.
- **DNS-rebinding protection** (MCP SDK) — accepted `Host`/`Origin` values are derived only from loopback, `public_url` and `allowed_hosts`.
- **Prompt-injection filtering** on inbound text; **credential redaction** on outbound text.
- **Rate limit** per identity on `chat` starts; optional `trusted_peers` allow-list.
- **Audit log** at `~/.hermes/mcp_http_audit.jsonl`; transcripts under `~/.hermes/cache/mcp_http/history/`.

## Troubleshooting

- **`401 Unauthorized`** — token mismatch; check the `Authorization: Bearer …` header against `.env`. Token edits in `.env` are picked up without a restart.
- **Server stays on 127.0.0.1** — by design: set a token first, then `host`.
- **`421`/host rejected through a tunnel** — set `public_url` (or `allowed_hosts`) to the hostname the tunnel presents.
- **`chat` says the gateway is not attached** — the gateway loop has not finished startup; retry in a few seconds.
- **Replies cut off around 60–100 s** — `wait_reply` already caps each poll at 55 s; make sure the client keeps polling instead of treating one `working` response as final.
