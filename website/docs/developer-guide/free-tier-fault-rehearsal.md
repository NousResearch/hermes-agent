---
sidebar_position: 6
title: "Rehearsing Free-Tier Failures"
description: "Drive every way the Nous free tier can refuse or fail, from the desktop's JavaScript console, against a local stand-in for the account service and the welcome host"
---

# Rehearsing Free-Tier Failures

The Nous free tier depends on two services: the account service (NAS), which creates the
free-tier identity and exchanges it for short-lived tokens, and the welcome inference host, which
serves the free model under its own rate limits and capacity caps. Hermes has a ruled behaviour for
every way either one can refuse or fail (see `hermes_cli/anon_auth.py`, the `ANON_*` codes and the
welcome refusal parser). This page is how you watch those behaviours happen on a real desktop
without touching production.

`scripts/free_tier_fault_server.py` is a single stdlib-only process that plays **both** services
and answers with the exact status codes, bodies and headers the real ones send. Which failure it
serves is a live switch you flip from the desktop's JavaScript console, `curl`, or a browser tab.

## 1. Start the stand-in

From the repo root:

```bash
python scripts/free_tier_fault_server.py
```

It listens on `127.0.0.1:8765` and prints the environment Hermes needs. You can also start it
already failing:

```bash
python scripts/free_tier_fault_server.py --nas paused --inference rate_limited
```

## 2. Start Hermes Desktop against it

Use a fresh temporary home so the rehearsal starts from a first launch, and add the four
variables the server printed to the usual guided-onboarding command from `apps/desktop`:

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

What each one does:

| Variable | Why |
|---|---|
| `HERMES_GUEST_ONBOARDING=1` | The free tier's launch gate; nothing is created without it. |
| `HERMES_PORTAL_BASE_URL` | Sends every account-service call (sign-up, token exchange, sign-in) to the stand-in. |
| `NOUS_INFERENCE_BASE_URL` | Sends chat completions to the stand-in instead of the welcome host. This override is trusted as the user's own setting and bypasses the network-side host allowlist. |
| `HERMES_EXTRA_WELCOME_HOSTS` | Dev-only. Makes Hermes treat `127.0.0.1` as the welcome host, so the free-tier route rules (model pinning, the dark-tier 403, the "sign in for a bigger allowance" copy) apply exactly as they do in production. Comma-separated hostnames; ports are ignored. It never widens the allowlist for URLs the network hands back. |

On a clean start the desktop boots, mints an identity against the stand-in, and lands on the
free-tier ready screen. Send a message and the stand-in replies with a canned line.

## 3. Flip a scenario

From the desktop's JavaScript console (View → Toggle Developer Tools):

```js
await (await fetch('http://127.0.0.1:8765/__scenario', {
  method: 'POST',
  body: JSON.stringify({ inference: 'rate_limited' })
})).json()
```

Or from a shell:

```bash
curl -s -X POST http://127.0.0.1:8765/__scenario -d '{"nas": "paused"}'
```

Then do the thing the scenario is about (send a message, restart the app, click *Try again*,
start a sign-in) and watch what the desktop says. The full catalogue, with what each scenario
means, is at `GET /__scenarios`; the current switches at `GET /__scenario`; the requests the
server has answered, with the scenario that was live for each, at `GET /__log`.

Control fields:

| Field | Meaning |
|---|---|
| `nas` | The account-service scenario (sign-up, token exchange, sign-in). |
| `inference` | The welcome-host scenario (chat completions). |
| `once` | `true` makes the next failure a one-off: after serving it, that service drops back to `ok`. Use it for "rate limited once, then fine". |
| `retry_after` | Overrides the wait the server names, in seconds, for any scenario that names one. |

`POST /__reset` puts everything back to the happy path and clears the log.

## 4. What to rehearse, and what you should see

**At boot (the account service).** Set the `nas` scenario, then quit and relaunch the desktop
with a fresh `HERMES_HOME`, or click *Try again* on the notice.

| Scenario | The desktop should |
|---|---|
| `not_enabled` | Show the provider picker with the "can't start without a Nous account" notice, no retry button, and the Nous row as the way in. |
| `paused` | Show the "paused for a moment" notice with *Try again*; the backend retries on its own with a one-minute floor, and the picker gives way by itself when the stand-in is set back to `ok`. |
| `rate_limited` | Show the "lots of people are getting started" notice naming the wait; the backend retries after `Retry-After`. |
| `server_error`, `timeout` | Show the "couldn't reach / had a hiccup" notice with *Try again* and **no** sign-in pointer, since the same service would refuse the sign-in. |
| `pow` | Show the proof-of-work sentence and point at sign-in; no retry. |
| `locked` | Show the "can't continue without signing in" notice. |
| `dead_once` | Nothing visible: the next token exchange fails as "unknown token", Hermes retires the identity and mints a replacement silently. Check `GET /__log` to see the two exchanges. |

**Mid-chat (the welcome host).** Set the `inference` scenario, then send a message.

| Scenario | The desktop should |
|---|---|
| `rate_limited` | Stop and say the allowance is used up, name the reset ("about 10 minutes"), and offer sign-in. The next send is refused locally until the reset. |
| `rate_limited_short` | Wait the few seconds quietly, then the reply arrives (set `once: true` first). |
| `at_capacity` | Retry with the named delay, then, once the retries are spent, say the free service is busy and offer both doors. |
| `model_not_free` | Only reachable when the session asks for another model; Hermes moves back to the free model once and retries. |
| `tier_disabled` | Say that using Hermes without signing in is switched off, stop retrying, and offer both doors. |
| `wrong_host` | Heal silently: Hermes re-reads its route and retries. The "different Nous server" sentence appears only if `NOUS_INFERENCE_BASE_URL` is the thing pointing at the wrong place. |
| `bare_429` | Same as `rate_limited`, driven by the `x-ratelimit-*` headers instead of the body. |
| `upstream_503`, `upstream_500` | Retry with backoff, then say the free model is having trouble responding. |
| `invalid_token` | Silent: one credential refresh, then a re-mint against the account service. |
| `timeout` | The turn's own transport timeout and retries. |

**Signing in.** With an identity in place, open the sign-in dialog from the status-bar chip, with
`nas` set to `signin_busy`, `signin_paused` or `signin_server_error`. The dialog should show
the "busy, try again in about a minute" or "couldn't reach" screen with *Try again*, and the
free-tier session should still be there afterwards. The stand-in can start a sign-in and report
`pending`, and you can settle it as declined or busy:

```bash
curl -s -X POST http://127.0.0.1:8765/__signin -d '{"status": "voided", "reason": "user_declined"}'
```

A **completed** sign-in is deliberately not something the stand-in can finish: the real portal
issues signed tokens the agent then verifies. Rehearse the happy path of a sign-in against
staging.

## 5. Notes

- The stand-in is for rehearsal, not for tests of Hermes' own logic. Those live in
  `tests/hermes_cli/test_anon_failure_modes.py` and `tests/agent/test_welcome_tier_recovery.py`
  and drive the same client code through in-process fakes. `tests/scripts/test_free_tier_fault_server.py`
  checks that the stand-in speaks the real contract, which is what makes a rehearsal against it
  meaningful.
- Because `HERMES_EXTRA_WELCOME_HOSTS` changes routing rules, never set it outside a rehearsal
  environment. It is read from the environment only, is never written to config, and is not
  shown in setup.
- The renderer talks to the stand-in's control surface directly (it is CORS-open). If a future
  content-security policy blocks that, the `curl` forms above do the same thing.
