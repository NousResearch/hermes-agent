---
title: "Rest Graphql Debug — Debug REST/GraphQL APIs: status codes, auth, schemas, repro"
sidebar_label: "Rest Graphql Debug"
description: "Debug REST/GraphQL APIs: status codes, auth, schemas, repro"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Rest Graphql Debug

Debug REST/GraphQL APIs: status codes, auth, schemas, repro.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/rest-graphql-debug` |
| Path | `optional-skills/software-development/rest-graphql-debug` |
| Version | `1.2.0` |
| Author | eren-karakus0 |
| License | MIT |
| Tags | `api`, `rest`, `graphql`, `http`, `debugging`, `testing`, `curl`, `integration` |
| Related skills | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# API Testing & Debugging

Drive REST and GraphQL diagnosis through Hermes tools — `terminal` for `curl`, `execute_code` for Python `requests`, `web_extract` for vendor docs. Isolate the failing layer before guessing at the fix.

## When to Use

- API returns unexpected status or body
- Auth fails (401/403 after token refresh, OAuth, API key)
- Works in Postman but fails in code
- Webhook / callback integration debugging
- Building or reviewing API integration tests
- Rate limiting or pagination issues

Skip for UI rendering, DB query tuning, or DNS/firewall infra (escalate).

## Core Principle

**Isolate the layer, then fix.** A 200 OK can hide broken data. A 500 can mask a one-character auth typo. Walk the chain in order; never skip a step.

```
1. Connectivity   → can we reach the host at all?
1.5 Timeouts      → connect-slow vs read-slow?
2. TLS/SSL        → cert valid and trusted?
3. Auth           → credentials correct and unexpired?
4. Request format → payload shape match server expectations?
5. Response parse → does our code accept what came back?
6. Semantics      → does the data mean what we assume?
```

## 5-Minute Quickstart

### REST via terminal

```python
# Verbose request/response exchange
terminal('curl -v https://api.example.com/users/1')

# POST with JSON
terminal("""curl -X POST https://api.example.com/users \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"name":"test","email":"test@example.com"}'""")

# Headers only
terminal('curl -sI https://api.example.com/health')

# Pretty-print JSON
terminal('curl -s https://api.example.com/users | python3 -m json.tool')
```

### GraphQL via terminal

```python
terminal("""curl -X POST https://api.example.com/graphql \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"query":"{ user(id: 1) { name email } }"}'""")
```

**GraphQL gotcha:** servers often return HTTP 200 even when the query failed. Always inspect the `errors` field regardless of status code:

```python
execute_code('''
import os, requests
resp = requests.post(
    "https://api.example.com/graphql",
    json={"query": "{ user(id: 1) { name email } }"},
    headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
    timeout=10,
)
data = resp.json()
if data.get("errors"):
    for err in data["errors"]:
        print(f"GraphQL error: {err['message']} (path: {err.get('path')})")
print(data.get("data"))
''')
```

### Python (requests) via execute_code

```python
execute_code('''
import requests
resp = requests.get(
    "https://api.example.com/users/1",
    headers={"Authorization": "Bearer <TOKEN>"},
    timeout=(3.05, 30),  # (connect, read)
)
print(resp.status_code, dict(resp.headers))
print(resp.text[:500])
''')
```

## Layered Debug Flow

### Step 1 — Connectivity

```python
terminal('nslookup api.example.com')
terminal('curl -v --connect-timeout 5 https://api.example.com/health')
```

Failures: DNS not resolving, firewall, VPN required, proxy missing.

### Step 1.5 — Timeouts

Distinguish *can't reach* from *reaches but slow*:

```python
terminal('''curl -w "dns:%{time_namelookup}s connect:%{time_connect}s tls:%{time_appconnect}s ttfb:%{time_starttransfer}s total:%{time_total}s\\n" \\
  -o /dev/null -s https://api.example.com/endpoint''')
```

In Python, always pass a tuple timeout — `requests` has no default and will hang forever:

```python
execute_code('''
import requests
from requests.exceptions import ConnectTimeout, ReadTimeout
try:
    requests.get(url, timeout=(3.05, 30))
except ConnectTimeout:
    print("Cannot reach host — DNS, firewall, VPN")
except ReadTimeout:
    print("Connected but server is slow")
''')
```

Diagnosis: high `time_connect` is network/firewall; high `time_starttransfer` with low `time_connect` is a slow server.

### Step 2 — TLS/SSL

```python
terminal('curl -vI https://api.example.com 2>&1 | grep -E "SSL|subject|expire|issuer"')
```

Failures: expired cert, self-signed, hostname mismatch, missing CA bundle. Use `-k` only for ad-hoc debug, never in code.

### Step 3 — Authentication

```python
# Token validity check
terminal('curl -s -o /dev/null -w "%{http_code}\\n" -H "Authorization: Bearer $TOKEN" https://api.example.com/me')

# Decode JWT exp claim — handles base64url padding correctly
execute_code('''
import json, base64, os
tok = os.environ["TOKEN"]
payload = tok.split(".")[1]
payload += "=" * (-len(payload) % 4)
print(json.dumps(json.loads(base64.urlsafe_b64decode(payload)), indent=2))
''')
```

Checklist:
- Token expired? (`exp` claim in JWT)
- Right scheme? Bearer vs Basic vs Token vs `X-Api-Key`
- Right environment? Staging key on prod is a classic
- API key in header vs query param (`?api_key=…`)?

### Step 4 — Request Format

```python
terminal("""curl -v -X POST https://api.example.com/endpoint \\
  -H 'Content-Type: application/json' \\
  -d '{"key":"value"}' 2>&1""")
```

**Content-Type / body mismatch — the silent 415/400:**

```python
# WRONG — data= sends form-encoded, header lies
requests.post(url, data='{"k":"v"}', headers={"Content-Type": "application/json"})

# RIGHT — json= auto-sets header AND serializes
requests.post(url, json={"k": "v"})

# WRONG — Accept says XML, code calls .json()
requests.get(url, headers={"Accept": "text/xml"})

# RIGHT — let requests build multipart with boundary
requests.post(url, files={"file": open("doc.pdf", "rb")})
```

Common: form-encoded vs JSON, missing required fields, wrong HTTP method, unencoded query params.

### Step 5 — Response Parsing

Always inspect content-type before calling `.json()`:

```python
execute_code('''
import requests
resp = requests.post(url, json=payload, timeout=10)
print(f"status={resp.status_code}")
print(f"headers={dict(resp.headers)}")
ct = resp.headers.get("Content-Type", "")
if "application/json" in ct:
    print(resp.json())
else:
    print(f"unexpected content-type {ct!r}, body={resp.text[:500]!r}")
''')
```

Failures: HTML error page where JSON expected, empty body, wrong charset.

### Step 6 — Semantic Validation

Parsed cleanly — but is the data *correct*?

- Does `"status": "active"` mean what your code thinks?
- ID in response matches the one requested?
- Timestamps in expected timezone?
- Pagination returning all results, or just page 1?

## HTTP Status Playbook

### 401 Unauthorized — credentials missing or invalid

1. `Authorization` header actually present? (`curl -v` to confirm)
2. Token correct and unexpired?
3. Right auth scheme? (`Bearer` vs `Basic` vs `Token`)
4. Some APIs use query param (`?api_key=…`) instead of header.

### 403 Forbidden — authenticated but not authorized

1. Token has the required scopes/permissions?
2. Resource owned by a different account?
3. IP allowlist blocking you?
4. CORS in browser? (check `Access-Control-Allow-Origin`)

### 404 Not Found — resource doesn't exist or URL is wrong

1. Path correct? (trailing slash, typo, version prefix)
2. Resource ID exists?
3. Right API version (`/v1/` vs `/v2/`)?
4. Right base URL (staging vs prod)?

### 409 Conflict — state collision

1. Resource already exists (duplicate create)?
2. Stale `ETag` / `If-Match`?
3. Concurrent modification by another process?

### 422 Unprocessable Entity — valid JSON, invalid data

The error body usually names the bad fields. Check:
- Field types (string vs int, date format)
- Required vs optional
- Enum values inside the allowed set

### 429 Too Many Requests — rate limited

Check `Retry-After` and `X-RateLimit-*` headers. Exponential backoff:

```python
execute_code('''
import time, requests

def with_backoff(method, url, **kwargs):
    for attempt in range(5):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = int(resp.headers.get("Retry-After", 2 ** attempt))
        time.sleep(wait)
    return resp
''')
```

### 5xx — server-side, usually not your fault

- **500** — server bug. Capture correlation ID, file with provider.
- **502** — upstream down. Backoff + retry.
- **503** — overloaded / maintenance. Check status page.
- **504** — upstream timeout. Reduce payload or raise timeout.

For all 5xx: backoff with jitter, alert on persistence.

## Pagination & Idempotency

**Pagination.** Verify you're getting *all* results. Look for `next_cursor`, `next_page`, `total_count`. Two patterns:
- Offset (`?limit=100&offset=200`) — simple, can skip items if data shifts.
- Cursor (`?cursor=abc123`) — preferred for live or large datasets.

**Idempotency.** For non-idempotent operations (POST), send `Idempotency-Key: <uuid>` so retries don't double-charge / double-create. Mandatory for payments and orders.

## Contract Validation

Catch schema drift before it hits production:

```python
execute_code('''
import requests

def validate_user(data: dict) -> list[str]:
    errors = []
    required = {"id": int, "email": str, "created_at": str}
    for field, expected in required.items():
        if field not in data:
            errors.append(f"missing field: {field}")
        elif not isinstance(data[field], expected):
            errors.append(f"{field}: want {expected.__name__}, got {type(data[field]).__name__}")
    return errors

resp = requests.get(f"{BASE}/users/1", headers=HEADERS, timeout=10)
issues = validate_user(resp.json())
if issues:
    print(f"contract violations: {issues}")
''')
```

Run after API upgrades, when integrating new third parties, or in CI smoke tests.

## Correlation IDs

Always capture the provider's request ID — fastest path to vendor support:

```python
execute_code('''
import requests
resp = requests.post(url, json=payload, headers=headers, timeout=10)
request_id = (
    resp.headers.get("X-Request-Id")
    or resp.headers.get("X-Trace-Id")
    or resp.headers.get("CF-Ray")  # Cloudflare
)
if resp.status_code >= 400:
    print(f"failed status={resp.status_code} req_id={request_id} ts={resp.headers.get('Date')}")
''')
```

**Vendor bug-report template:**

```
Endpoint:    POST /api/v1/orders
Request ID:  req_abc123xyz
Timestamp:   2026-03-17T14:30:00Z
Status:      500
Expected:    201 with order object
Actual:      500 {"error":"internal server error"}
Repro:       curl -X POST … (auth: <REDACTED>)
```

## Regression Test Template

Drop this into `tests/` and run via `terminal('pytest tests/test_api_smoke.py -v')`:

```python
import os, requests, pytest

BASE_URL = os.environ.get("API_BASE_URL", "https://api.example.com")
TOKEN    = os.environ.get("API_TOKEN", "")
HEADERS  = {"Authorization": f"Bearer {TOKEN}"}

class TestAPISmoke:
    def test_health(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_list_users_returns_array(self):
        resp = requests.get(f"{BASE_URL}/users", headers=HEADERS, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("data", data), list)

    def test_get_user_required_fields(self):
        resp = requests.get(f"{BASE_URL}/users/1", headers=HEADERS, timeout=10)
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            user = resp.json()
            assert "id" in user and "email" in user

    def test_invalid_auth_returns_401(self):
        resp = requests.get(
            f"{BASE_URL}/users",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10,
        )
        assert resp.status_code == 401
```

## Security

### Token handling
- Never log full tokens. Redact: `Bearer <REDACTED>`.
- Never hardcode tokens in scripts. Read from env (`os.environ["API_TOKEN"]`) or `~/.hermes/.env`.
- Rotate immediately if a token surfaces in logs, error messages, or git history.

### Safe logging

```python
def redact_auth(headers: dict) -> dict:
    sensitive = {"authorization", "x-api-key", "cookie", "set-cookie"}
    return {k: ("<REDACTED>" if k.lower() in sensitive else v) for k, v in headers.items()}
```

### Leak checklist

- [ ] **Credentials in URLs.** API keys in query strings end up in server logs, browser history, referrer headers — use headers.
- [ ] **PII in error responses.** `404 on /users/123` shouldn't reveal whether the user exists (enumeration).
- [ ] **Stack traces in prod.** 500s shouldn't leak file paths, framework versions.
- [ ] **Internal hostnames/IPs.** `10.x.x.x`, `internal-api.corp.local` in error bodies.
- [ ] **Tokens echoed back.** Some APIs include the auth token in error details. Verify they don't.
- [ ] **Verbose `Server` / `X-Powered-By`.** Stack-info leaks. Note for security review.

## Hermes Tool Patterns

### terminal — for curl, dig, openssl

```python
terminal('curl -sI https://api.example.com')
terminal('openssl s_client -connect api.example.com:443 -servername api.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates')
```

### execute_code — for multi-step Python flows

When debugging spans auth → fetch → paginate → validate, use `execute_code`. Variables persist for the script, results print to stdout, no risk of token spam in your context:

```python
execute_code('''
import os, requests

token = os.environ["API_TOKEN"]
base  = "https://api.example.com"
H     = {"Authorization": f"Bearer {token}"}

# 1. auth
me = requests.get(f"{base}/me", headers=H, timeout=10)
print(f"auth {me.status_code}")

# 2. paginate
all_users, cursor = [], None
while True:
    params = {"cursor": cursor} if cursor else {}
    r = requests.get(f"{base}/users", headers=H, params=params, timeout=10)
    body = r.json()
    all_users.extend(body["data"])
    cursor = body.get("next_cursor")
    if not cursor:
        break
print(f"users={len(all_users)}")
''')
```

### web_extract — for vendor API docs

Pull the spec for the endpoint you're debugging instead of guessing:

```python
web_extract(urls=["https://docs.example.com/api/v1/users"])
```

### delegate_task — for full CRUD test sweeps

```python
delegate_task(
    goal="Test all CRUD endpoints for /api/v1/users",
    context="""
Follow the rest-graphql-debug skill (optional-skills/software-development/rest-graphql-debug).
Base URL: https://api.example.com
Auth: Bearer token from API_TOKEN env var.

For each verb (POST, GET, PATCH, DELETE):
  - happy path: assert status + response schema
  - error cases: 400, 404, 422
  - log a repro curl for any failure (redact tokens)

Output: pass/fail per endpoint + correlation IDs for failures.
""",
    toolsets=["terminal", "file"],
)
```

## Output Format

When reporting findings:

```
## Finding
Endpoint: POST /api/v1/users
Status:   422 Unprocessable Entity
Req ID:   req_abc123xyz

## Repro
curl -X POST https://api.example.com/api/v1/users \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <REDACTED>' \
  -d '{"name":"test"}'

## Root Cause
Missing required field `email`. Server validation rejects before processing.

## Fix
-d '{"name":"test","email":"test@example.com"}'
```

## Related

- `systematic-debugging` — once the failing API layer is isolated, root-cause your code
- `test-driven-development` — write the regression test before shipping the fix
