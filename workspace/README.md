# Hermes Workspace — Custom Patches

These files are patches for the [hermes-workspace](https://github.com/outsourc-e/hermes-workspace)
frontend. Apply them to fix the session rename functionality.

## Changes

### `hermes-dashboard-api.ts`
- Added `updateSession()` function that sends PATCH to the dashboard API

### `hermes-api.ts`
- `updateSession()` now routes to the dashboard (like `deleteSession()` does)
  instead of sending PATCH to the gateway (which doesn't support it)

## How to apply

1. Clone `outsourc-e/hermes-workspace`
2. Copy these files into `src/server/`
3. Run `pnpm install && npx vite build`
4. Mount `dist/server/` into the workspace container

Or build a custom Docker image from the patched workspace.
