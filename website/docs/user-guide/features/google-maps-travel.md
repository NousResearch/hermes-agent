---
title: Google Maps travel POIs and routing
sidebar_position: 96
---

# Google Maps travel POIs and routing

Hermes can optionally use Google Maps Platform for higher-confidence travel place lookups and route estimates. This is an optional paid-provider path: the existing OpenStreetMap-based maps behavior remains the no-key fallback and should continue to work when no Google Maps key is configured.

## Setup

Jonas must provide a Bitwarden Secrets Manager secret with the exact name `GOOGLE_MAPS_API_KEY` if he wants the Google-backed path enabled.

Do not paste the Google Maps API key into chat, Kanban comments, docs, logs, shell history, or tool output. If the key is missing, Hermes should say that `GOOGLE_MAPS_API_KEY` needs to be added in Bitwarden Secrets Manager and then reload/restart Hermes; it must never ask Jonas to send the key in chat.

Recommended setup flow:

1. Create or use a Google Cloud project with billing configured for Google Maps Platform.
2. Enable only the Google Maps Platform APIs needed by the implemented Hermes features:
   - Places API: required for place search and place details.
   - Geocoding API: required only if the Google-backed implementation performs address-to-coordinate or coordinate-to-address geocoding.
   - Routes API: required only if the Google-backed implementation performs route estimates, distance, duration, or directions.
3. Store the API key in Bitwarden Secrets Manager as `GOOGLE_MAPS_API_KEY`.
4. Ensure the Hermes profile that will use travel POIs/routing loads Bitwarden Secrets Manager secrets.
5. Reload or restart Hermes so the environment contains `GOOGLE_MAPS_API_KEY`.
6. Run the implemented safe Google Maps capability/probe tools, if available, before relying on live results.

The environment variable is documented in the environment variables reference, but the preferred source is Bitwarden Secrets Manager with the same secret name.

## Required APIs by feature

Enable only the APIs used by the feature you are turning on:

| Hermes feature | Google Maps Platform API |
| --- | --- |
| Place search, nearby search, text search, place details, POI confidence checks | Places API |
| Address geocoding or reverse geocoding | Geocoding API |
| Route estimates, distance/duration estimates, directions-style route calculation | Routes API |

Do not enable APIs preemptively for unsupported Hermes features. Do not use alternate secret names or invented config keys; the supported key name for this optional path is exactly `GOOGLE_MAPS_API_KEY`.

## Fallback behavior

`GOOGLE_MAPS_API_KEY` is optional. When it is absent, blank, invalid, or when Google-backed tools are not enabled, Hermes should keep using the existing OpenStreetMap/Nominatim/Overpass/OSRM maps behavior. No-key deployments should not lose the current OSM-based geocoding, nearby POI, distance, directions, timezone, area, or bounding-box workflows.

When both OSM and Google results are available, responses should make the source clear so Jonas can tell whether an answer came from Google Maps Platform, OSM-based sources, or both.

## Billing, quotas, and source caveats

Google Maps Platform APIs may bill per request and may have quotas, SKU-specific pricing, and project-specific restrictions. Use small probes and mocked tests where possible, and avoid live API calls unless the operator expects potential Google Maps Platform charges.

Travel POI/routing data can be incomplete or stale. Always treat hours, place availability, travel times, and route estimates as planning aids that may need verification in Google Maps, the venue site, or another authoritative provider before making decisions.

## Secret handling requirements

- Never log, display, return, persist, or include `GOOGLE_MAPS_API_KEY` in tool output.
- Redact request metadata so it never includes the raw key.
- Error messages should name the missing secret as `GOOGLE_MAPS_API_KEY` but must not include any key value.
- Do not ask Jonas to paste the key in chat; ask him to add or fix the Bitwarden Secrets Manager secret instead.
- Keep the existing OSM fallback available even when Google Maps credentials are absent.
