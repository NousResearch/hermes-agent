# WHOOP

Hermes can read WHOOP profile, recovery, sleep, cycle, and workout data through WHOOP's official Developer API. The integration is read-only: it exposes your metrics to the agent, but it does not write data back to WHOOP and it does not provide medical advice.

WHOOP health telemetry is personal data. Treat it like private context and only share summaries outside your own Hermes session when you explicitly choose to.

## Prerequisites

- A WHOOP account.
- Hermes Agent installed and running.
- A WHOOP Developer app with OAuth credentials from the WHOOP Developer Dashboard.

## Setup

### 1. Create a WHOOP Developer app

Open the WHOOP Developer docs/dashboard:

```bash
open https://developer.whoop.com/api
```

Create an app and configure the redirect URI exactly as:

```text
http://127.0.0.1:43828/whoop/callback
```

Requested scopes:

```text
offline read:profile read:body_measurement read:cycles read:recovery read:sleep read:workout
```

WHOOP requires an OAuth `state` value with exactly eight characters. Hermes generates that automatically during `hermes auth whoop`.

### 2. Store credentials locally

Do not paste your WHOOP client secret into chat. Put it only in your local shell or Hermes `.env` file.

Find the env file:

```bash
hermes config env-path
```

Then add:

```bash
HERMES_WHOOP_CLIENT_ID="your-client-id"
HERMES_WHOOP_CLIENT_SECRET="your-client-secret"
```

Hermes also accepts `WHOOP_CLIENT_ID` and `WHOOP_CLIENT_SECRET`, but the `HERMES_` names are preferred.

### 3. Run OAuth login

```bash
hermes auth whoop
```

After approval, tokens are stored under `providers.whoop` in your Hermes auth store. This does not change your active LLM provider.

## Verify

```bash
hermes auth status whoop
```

A logged-in status means WHOOP tools can appear when the `whoop` toolset is enabled. Access tokens refresh automatically when expiring, and WHOOP API 401 responses trigger one forced refresh and retry.

## Enable the toolset

```bash
hermes tools
```

Toggle the WHOOP toolset on, then start a new session. Toolset changes take effect on a fresh session so prompt/tool caching stays stable.

## Tools

WHOOP tools are read-only and return raw API data plus light wrapper metadata. They do not diagnose, prescribe, or provide medical advice.

### `whoop_profile`

Fetches your basic WHOOP profile.

Example prompt:

```text
Show my WHOOP profile.
```

### `whoop_cycles`

Fetches physiological cycle records.

Actions:

- `latest` — fetch the most recent record by listing with `limit=1`.
- `list` — fetch paginated records.
- `get` — fetch a record by `id`.

Common args for `list` / `latest`:

- `start` — ISO-8601 start timestamp.
- `end` — ISO-8601 end timestamp.
- `limit` — clamped to 1-25.
- `next_token` / `nextToken` — pagination token.
- `max_pages` — clamped to 1-10.

### `whoop_recovery`

Fetches recovery records.

Actions:

- `latest` — fetch the most recent recovery record by listing with `limit=1`.
- `list` — fetch paginated recovery records.
- `get` — fetch recovery for a specific WHOOP cycle via `cycle_id` (`id` is accepted as an alias).

Example prompt:

```text
What was my latest WHOOP recovery score? Keep it factual, no medical advice.
```

### `whoop_sleep`

Fetches sleep records.

Example prompt:

```text
Pull my WHOOP sleep records for the last week and summarize duration and disturbances.
```

### `whoop_workouts`

Fetches workout/activity records.

Example prompt:

```text
List my latest WHOOP workouts and total the strain by day.
```

## Pagination and rate limits

WHOOP collections return `records` and may include `next_token`. Follow-up requests use `nextToken`. Hermes handles that internally when `max_pages` is greater than 1.

Default WHOOP API limits are 100 requests/minute and 10,000/day. If WHOOP returns `429`, Hermes surfaces the rate-limit context and `Retry-After` value when present.

## Privacy and safety

- Never paste WHOOP client secrets, access tokens, refresh tokens, or OAuth codes into chat.
- WHOOP data is personal health telemetry. Keep it in private sessions unless you intentionally share it.
- Hermes can summarize metrics, but it is not a clinician. For medical decisions, talk to a qualified professional. Annoying boilerplate, yes. Still true.
