---
title: SerpApi travel search
sidebar_position: 95
---

# SerpApi travel search

Hermes can use SerpApi to run search-only Google Flights and Google Hotels lookups. These tools are for research and planning only: Hermes must never book, reserve, hold, purchase, or otherwise commit to travel from these results.

## Setup

Jonas must provide a Bitwarden Secrets Manager secret with the exact name `SERPAPI_API_KEY`.

Do not paste the SerpApi API key into chat, Kanban comments, docs, logs, or shell history. If the key is missing, Hermes should tell Jonas to add the `SERPAPI_API_KEY` secret in Bitwarden Secrets Manager and then reload/restart Hermes; it must never ask Jonas to send the key in chat.

Recommended setup flow:

1. Store the SerpApi API key in Bitwarden Secrets Manager as `SERPAPI_API_KEY`.
2. Ensure the Hermes profile that will use travel search loads Bitwarden Secrets Manager secrets.
3. Enable the `serpapi` toolset for the relevant platform/profile with `hermes tools` or equivalent config.
4. Reload or restart Hermes so the environment contains `SERPAPI_API_KEY`.
5. Run the read-only probe below.

The environment variable is documented in the environment variables reference, but the preferred source is Bitwarden Secrets Manager with the same secret name.

## Read-only credential probe

Use `serpapi_read_only_probe` before live searches to verify the key is present and accepted:

```json
{
  "tool": "serpapi_read_only_probe",
  "arguments": {}
}
```

The probe calls SerpApi's Account API at `https://serpapi.com/account.json`. It is read-only, does not book or purchase anything, and SerpApi documents the Account API as free/not counted toward monthly search quota. The returned payload is redacted and must not include `SERPAPI_API_KEY`.

Expected success fields:

- `success`: `true`
- `provider`: `serpapi`
- `tool`: `serpapi_read_only_probe`
- `capability`: confirmation text
- `endpoint`: SerpApi Account API URL
- `quota_cost`: notes that the account probe is free/not counted toward monthly search quota
- `account`: non-secret account fields such as `account_id`, `account_email`, `plan_name`, `searches_per_month`, `plan_searches_left`, `total_searches_left`, or usage/rate-limit counters when SerpApi returns them

If the secret is missing, the tool returns `success: false`, `missing_secret: SERPAPI_API_KEY`, and an actionable error telling the operator to store the key in Bitwarden Secrets Manager. It must not ask for the key in chat.

## Google Flights search

Tool: `serpapi_google_flights_search`

Example round-trip query:

```json
{
  "tool": "serpapi_google_flights_search",
  "arguments": {
    "origin": "AMS",
    "destination": "JFK",
    "departure_date": "2026-10-25",
    "return_date": "2026-11-01",
    "adults": 2,
    "children": 1,
    "cabin_class": "economy",
    "currency": "EUR",
    "gl": "nl",
    "hl": "en",
    "limit": 5
  }
}
```

Example one-way query with additional SerpApi-compatible filters:

```json
{
  "tool": "serpapi_google_flights_search",
  "arguments": {
    "origin": "AMS",
    "destination": "LHR",
    "departure_date": "2026-06-12",
    "trip_type": "one_way",
    "adults": 1,
    "cabin_class": "business",
    "currency": "EUR",
    "filters": {
      "stops": "0"
    },
    "limit": 10
  }
}
```

Inputs:

- `origin`: origin airport/city code such as `AMS`
- `destination`: destination airport/city code such as `JFK`
- `departure_date`: `YYYY-MM-DD`
- `return_date`: optional `YYYY-MM-DD`; when provided the default trip type is round-trip
- `trip_type`: `round_trip` or `one_way`; defaults from `return_date`
- `adults`, `children`, `infants_in_seat`, `infants_on_lap`
- `cabin_class`: `economy`, `premium_economy`, `business`, or `first`
- `currency`: currency code such as `EUR` or `USD`
- `gl`, `hl`: optional Google country/language parameters
- `filters`: optional SerpApi `google_flights` parameters; `api_key` is ignored if supplied
- `limit`: number of normalized options to return, capped at 50

Expected output fields:

- `success`, `provider`, `tool`, `engine`
- `query`: normalized origin, destination, dates, trip type, passenger counts, cabin class, currency, and redacted request parameters
- `results_count`
- `results`: each flight option includes:
  - `rank`
  - `source_bucket`: usually `best_flights` or `other_flights`
  - `price.amount` and `price.currency`
  - `type`
  - `provider`: `Google Flights via SerpApi`
  - `airlines`
  - `total_duration_minutes`
  - `segments`, with departure/arrival airport names, IDs, times, airline, flight number, aircraft, duration, and travel class when available
  - `booking_link`, when SerpApi returns one
  - `booking_token_available`: boolean only; the raw booking token is intentionally not returned
  - compact `raw_metadata` such as extensions, carbon emissions, logo, or layovers
- `source_metadata`: SerpApi search ID/status/json endpoint and redacted source search parameters
- `caveat`: volatility and no-booking policy

## Google Hotels search

Tool: `serpapi_google_hotels_search`

Example query:

```json
{
  "tool": "serpapi_google_hotels_search",
  "arguments": {
    "destination": "Amsterdam",
    "check_in_date": "2026-10-25",
    "check_out_date": "2026-10-28",
    "adults": 2,
    "children": 0,
    "rooms": 1,
    "currency": "EUR",
    "gl": "nl",
    "hl": "en",
    "limit": 5
  }
}
```

Example with optional SerpApi-compatible filters:

```json
{
  "tool": "serpapi_google_hotels_search",
  "arguments": {
    "destination": "hotels near Schiphol Airport",
    "check_in_date": "2026-10-25",
    "check_out_date": "2026-10-26",
    "adults": 1,
    "rooms": 1,
    "currency": "EUR",
    "filters": {
      "hotel_class": "4"
    },
    "limit": 10
  }
}
```

Inputs:

- `destination`: city, region, landmark, or hotel search query
- `check_in_date`: `YYYY-MM-DD`
- `check_out_date`: `YYYY-MM-DD`
- `adults`, `children`, `rooms`
- `currency`: currency code such as `EUR` or `USD`
- `gl`, `hl`: optional Google country/language parameters
- `filters`: optional SerpApi `google_hotels` parameters; `api_key` is ignored if supplied
- `limit`: number of normalized hotel results to return, capped at 50

Expected output fields:

- `success`, `provider`, `tool`, `engine`
- `query`: redacted request parameters
- `results_count`
- `hotels`: each hotel includes:
  - `name`
  - `price_rate.lowest`, `price_rate.extracted_lowest`, `price_rate.currency`, and `price_rate.source`
  - `dates.check_in` and `dates.check_out`
  - `provider_source`, usually `Google Hotels via SerpApi` or a provider returned by SerpApi
  - `link`, property token, or source link when available
  - `location`
  - `rating`, `reviews`, and `hotel_class` when available
  - compact `raw_metadata` such as property token, coordinates, amenities, images, or thumbnail
- `search_metadata`
- `search_parameters`: redacted source search parameters
- `caveat`: volatility and no-booking policy

## Caveats and policy

- Travel prices, availability, schedules, hotel rates, and inventory are volatile. Always recheck with the airline, hotel, or provider before making decisions.
- These tools are read-only search helpers. Hermes must not book, reserve, hold, purchase, cancel, or modify travel.
- Search results may include booking links or provider tokens. They are informational only and are not authorization to transact.
- Do not expose `SERPAPI_API_KEY`. The wrappers strip `api_key` from returned request metadata, and the flights wrapper returns only `booking_token_available` rather than raw booking tokens.
- Live SerpApi searches consume SerpApi search quota. Use mocked tests for CI and manual live QA only when quota use is acceptable.

## Manual QA checklist

Live API tests are not required for CI because they depend on a real SerpApi account and quota. For manual verification:

- [ ] Confirm `SERPAPI_API_KEY` exists via Bitwarden Secrets Manager and is loaded into the Hermes environment without printing the value.
- [ ] Enable the `serpapi` toolset and restart/reload Hermes.
- [ ] Run `serpapi_read_only_probe`; verify `success: true` or an actionable non-secret error.
- [ ] Run a small `serpapi_google_flights_search` with `limit: 1`; verify normalized flight fields, volatility caveat, no raw API key, and no raw booking token.
- [ ] Run a small `serpapi_google_hotels_search` with `limit: 1`; verify normalized hotel fields, volatility caveat, and no raw API key.
- [ ] Confirm user-facing summaries say results are volatile and Hermes cannot book or purchase travel.
- [ ] If the secret is missing, confirm the error tells Jonas to add `SERPAPI_API_KEY` in Bitwarden Secrets Manager and does not ask for the key in chat.
