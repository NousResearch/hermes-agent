---
sidebar_position: 6
title: "Rehearsing Free-Tier Failures"
description: "Point Hermes Desktop at a local stand-in for the account service and the welcome host to watch its free-tier failure handling"
---

# Rehearsing Free-Tier Failures

The Nous free tier depends on two services: the account service (NAS), which creates the
free-tier identity and exchanges it for short-lived tokens, and the welcome inference host, which
serves the free model under its own rate limits and capacity caps. Hermes has a ruled behaviour for
every way either can refuse or fail: the `ANON_*` codes and the welcome refusal parser in
`hermes_cli/anon_auth.py`, and the onboarding notice and sign-in screens keyed on them in the
desktop. Those behaviours are pinned by tests (`tests/hermes_cli/test_anon_failure_modes.py`,
`tests/agent/test_welcome_tier_recovery.py`). This page is how to watch them on a real desktop
without touching production.

## Point Hermes at a stand-in

Run any local HTTP server that answers as the two services do, then start the desktop with:

```bash
D=$(mktemp -d)
env -u NODE_ENV \
  HERMES_GUEST_ONBOARDING=1 \
  HERMES_HOME=$D/.hermes HERMES_DESKTOP_USER_DATA_DIR=$D/electron-user-data \
  HERMES_SHARED_AUTH_DIR=$D/.hermes/shared \
  HERMES_PORTAL_BASE_URL=http://127.0.0.1:8765 \
  NOUS_INFERENCE_BASE_URL=http://127.0.0.1:8765/v1 \
  HERMES_EXTRA_WELCOME_HOSTS=127.0.0.1 \
  npm run dev
```

| Variable | Why |
|---|---|
| `HERMES_GUEST_ONBOARDING=1` | The free tier's launch gate; nothing is created without it. |
| `HERMES_PORTAL_BASE_URL` | Sends every account-service call (sign-up, token exchange, sign-in) to the stand-in. |
| `NOUS_INFERENCE_BASE_URL` | Sends chat completions to the stand-in. Trusted as the user's own setting; bypasses the network-side host allowlist. |
| `HERMES_EXTRA_WELCOME_HOSTS` | Dev-only. Makes Hermes treat the stand-in's hostname as the welcome host, so the free-tier route rules (model pinning, the route-keyed dark-tier 403, the "sign in for a bigger allowance" copy) apply as in production. Comma-separated hostnames, ports ignored. Never widens the allowlist for URLs the network hands back, and never belongs outside a rehearsal environment. |

## What the stand-in has to speak

On the account-service side: `POST /api/anonymous/create` (201 with an `anon_` token),
`POST /api/anonymous/token` (200 with `access_token`, `expires_in`, `inference_base_url`), and,
for sign-in rehearsals, `POST /api/oauth/device/code`, `/api/anonymous/promotion-intent` and
`/api/anonymous/promotion-status`. The refusals Hermes classifies are the ones NAS actually sends:
404 `not_found`, 503 `temporarily_disabled`, 429 with `Retry-After`, 428 `pow_*`, 404
`unknown_token`, 403 `account_locked`; see the code table at the top of `anon_auth.py`.

On the inference side: `POST /v1/chat/completions` (streaming or not), with the gateway's
structured 429 body `{status, message, reason, retry_after, alternates, upgrade_url}` for
`rate_limited` / `at_capacity` / `model_not_free`, a generic-bodied 403 for a dark tier, the
"Anonymous accounts must use…" 400 for a wrong host, and plain 5xx for outages.

A small stdlib `http.server` with a `POST /__scenario` switch is enough; keep it CORS-open and you
can flip scenarios from the desktop's JavaScript console with one `fetch`. One such server was
written alongside this work and can be regenerated from the description above; it is not kept in
the repo because it only serves one-off manual rehearsal and would drift silently from the real
services.

## Reading the result

The failure-to-message mapping the desktop should follow is in the pull request that introduced
it and in the tests named above: the boot-time notice above the provider picker (with *Try again*
when a retry can work, and the sign-in pointer only when the account service answered at all), the
chat copy for each welcome refusal, and the sign-in dialog's busy / unreachable / unavailable
screens.
