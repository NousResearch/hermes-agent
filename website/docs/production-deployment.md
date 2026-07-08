# Production deployment

This guide documents a production-oriented deployment baseline for Hermes Agent's API server/gateway. It is intentionally conservative: the sample Compose file binds the API server to localhost, requires an API key before any public exposure, runs the container as a non-root user, and places TLS/rate limiting concerns at a reverse proxy.

## Production readiness checklist

- Generate a long random `API_SERVER_KEY` before exposing the API server.
- Keep `docker-compose.yml` bound to `127.0.0.1` unless a hardened network perimeter is in place.
- Terminate TLS at nginx or Caddy and forward `Authorization`, `X-Hermes-Session-Id`, and `X-Hermes-Session-Key` headers.
- Configure `API_SERVER_CORS_ORIGINS` explicitly for browser clients; do not use wildcard origins in production.
- If using cookie authentication for browser clients, configure `API_SERVER_COOKIE_NAME` and send a matching CSRF header/cookie on mutating requests.
- Store revoked bearer tokens as SHA-256 digests in `API_SERVER_REVOKED_TOKEN_SHA256`; never store raw revoked tokens.
- Back up `HERMES_HOME` regularly and test restore procedures.
- Rotate logs or ship them to centralized logging.

## Files added

- `Dockerfile` — non-root Python runtime image with API-server healthcheck.
- `.dockerignore` — excludes virtualenvs, caches, local state, prototypes, logs, and env files.
- `docker-compose.yml` — localhost-bound production baseline with persistent volume and container hardening options.
- `.env.production.example` — safe template for runtime settings; copy to `.env.production` and never commit the real file.
- `deploy/nginx/hermes-agent.conf` — TLS reverse proxy sample for nginx.
- `deploy/caddy/Caddyfile` — TLS reverse proxy sample for Caddy.
- `deploy/scripts/backup-hermes.sh` — tar-based backup helper for `HERMES_HOME`.
- `deploy/logrotate/hermes-agent` — host logrotate sample.

## Build and run with Docker Compose

```bash
cp .env.production.example .env.production
python - <<'PY'
import secrets
print('API_SERVER_KEY=' + secrets.token_urlsafe(48))
PY
# Put the generated value into .env.production, then:
docker compose up --build -d
curl -fsS http://127.0.0.1:8642/health
curl -fsS -H "Authorization: Bearer $API_SERVER_KEY" http://127.0.0.1:8642/v1/capabilities
```

The Compose file maps `127.0.0.1:8642:8642` by default. For public deployments, keep the API server private and put nginx/Caddy on the public interface.

## Authentication and token rotation

Hermes API server supports bearer authentication via `API_SERVER_KEY` or `gateway.platforms.api_server.extra.key`.

Revocation is supported with SHA-256 digests:

```bash
python - <<'PY'
import hashlib, getpass
secret = getpass.getpass('Token to revoke: ')
print(hashlib.sha256(secret.encode('utf-8')).hexdigest())
PY
```

Then add the digest to `API_SERVER_REVOKED_TOKEN_SHA256` as a comma-separated list. Prefixes like `sha256:<digest>` are accepted for readability.

`API_SERVER_MAX_BEARER_TOKEN_AGE_SECONDS` is surfaced in `/v1/capabilities` as operator guidance for clients and token issuers. Opaque API keys cannot be age-validated without a token-issuance database, so age enforcement belongs in a future JWT/OIDC integration.

## Cookie auth and CSRF

Bearer tokens are preferred for server-to-server clients. If a browser app uses an auth cookie:

- Set `API_SERVER_COOKIE_NAME` to the cookie carrying the same API token.
- For `POST`, `PUT`, `PATCH`, and `DELETE`, send a double-submit CSRF pair:
  - cookie: `API_SERVER_CSRF_COOKIE` value, default `hermes_csrf`
  - header: `API_SERVER_CSRF_HEADER` value, default `X-CSRF-Token`
- The header and cookie values must match exactly.

Bearer-authenticated requests do not require CSRF because browsers do not attach bearer headers automatically.

## Reverse proxy

Use either:

- `deploy/nginx/hermes-agent.conf`
- `deploy/caddy/Caddyfile`

Both samples preserve authorization/session headers and allow long-running/SSE agent requests. Replace `hermes.example.com` and certificate paths before use.

## Backups

Run:

```bash
HERMES_HOME=/path/to/hermes-home BACKUP_DIR=/secure/backups deploy/scripts/backup-hermes.sh
```

By default, secret-like files are excluded. To include secrets for a full disaster-recovery backup, set `INCLUDE_SECRETS=1` and protect the resulting archive with strong access controls.

Always test restore from backup. For SQLite/state files under heavy write load, prefer backing up during a maintenance window or after stopping the service.

## Log rotation

Install the sample as root after adjusting paths/user/group:

```bash
sudo cp deploy/logrotate/hermes-agent /etc/logrotate.d/hermes-agent
sudo logrotate -d /etc/logrotate.d/hermes-agent
```

The sample uses `copytruncate` so Hermes does not need to reopen log files. For higher-volume deployments, ship logs to Loki, ELK, CloudWatch, or another centralized logging backend.

## JWT/OIDC future guidance

This baseline intentionally does not add JWT validation dependencies. If JWT/OIDC is added later, production defaults should be:

- access tokens: 15–30 minutes
- refresh tokens: 7–30 days with rotation
- asymmetric signing: RS256 or ES256
- required claims: `iss`, `sub`, `aud`, `exp`, `iat`, `jti`
- audience: `hermes-api`
- revocation: denylist by `jti` with TTL no longer than token lifetime
- clock skew tolerance: 60 seconds or less
