# A2A sidecar

Hermes can expose a narrow [Agent2Agent (A2A)](https://a2a-protocol.org/) facade for independent peer agents without opening the full Hermes gateway, memory, filesystem, or tool surface.

The sidecar is a separate process:

```text
External A2A client
        ⇅ HTTPS + A2A JSON-RPC/SSE + Agent Card
hermes-a2a sidecar
        ⇅ local SQLite / Kanban API
Hermes Kanban dispatcher + profiles
```

## Install

The sidecar uses the official Python A2A SDK instead of a custom wire format:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python 'hermes-agent[a2a]'
# or, in a source checkout:
uv pip install --python venv/bin/python 'a2a-sdk[http-server]==1.1.0' 'uvicorn[standard]==0.41.0'
```

## Configure peers

Prefer storing bearer token hashes in `config.yaml`, not raw tokens. Generate a hash:

```bash
python - <<'PY'
import getpass, hashlib
print(hashlib.sha256(getpass.getpass('A2A bearer token: ').encode()).hexdigest())
PY
```

Example:

```yaml
a2a:
  enabled: true
  host: 127.0.0.1
  port: 8765
  public_url: https://a2a.example.com
  rpc_path: /a2a
  peers:
    partner_agent_1:
      token_sha256: "<sha256 of bearer token>"
      allowed_skills:
        - delegate_engineering_task
        - submit_artifact_for_review
      default_assignee: engineer
      tenant: a2a
      max_payload_bytes: 200000
      allowed_artifact_domains:
        - artifacts.partner.example
      requires_human_review_for:
        - code_execution
        - script_execution
        - config_change
        - file_write
        - external_posting
```

For local development only, you can set `HERMES_A2A_TOKEN` and run with the implicit `env` peer.

## Run

```bash
HERMES_A2A_TOKEN='dev-token' hermes-a2a --host 127.0.0.1 --port 8765 --public-url http://127.0.0.1:8765
```

Public discovery:

```bash
curl http://127.0.0.1:8765/.well-known/agent-card.json
```

Authenticated JSON-RPC calls go to `/a2a` and must include:

```text
Authorization: Bearer <peer-token>
A2A-Version: 1.0
```

## Security model

- Public Agent Card is sparse; sensitive details stay behind auth.
- Peer identity is determined at the HTTP/perimeter layer, not from task text.
- Per-peer policy controls allowed A2A skills, payload size, artifact domains, default assignee, and review-required actions.
- Remote content is written into Kanban as **untrusted** content.
- Remote scripts/configs/artifacts are staged by URL/checksum only; they are never executed by the sidecar.
- High-impact requests create blocked review cards (`input_required` in A2A) until a human/policy layer approves.
- All inbound task events are audited in `~/.hermes/a2a/sidecar.db` with a correlation id.

For production exposure, put the sidecar behind a standard perimeter such as Cloudflare Access, Tailscale, OIDC/oauth2-proxy, or mTLS. Keep the sidecar itself bound to localhost whenever possible.
