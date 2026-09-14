# Control Room — Performance Evidence (CR-606)

Measured against the plan's requirements: idle polling budget, no
rerender-per-token, no terminal input latency regression, bounded list sizes.

## Polling budget (idle)

| Surface | Poll | Budget |
|---------|------|--------|
| CLI status segment | `ControlRoomService` 2.0s cache (`DEFAULT_CACHE_MAX_AGE_SECONDS=2.0`, service.py:47) — the segment reads the cached snapshot on normal status-bar refresh; **no second poll loop** | 1 RPC-equivalent / 2s max |
| TUI attention badge | `POLL_MS=5000` `setInterval` (controlRoomBadge.tsx:19) — counts only, cancelled on unmount | 1 RPC / 5s |
| Desktop status badge | `POLL_MS=5000` (control-room-status-badge.tsx:16) — counts only, cancelled on unmount | 1 RPC / 5s |
| Dashboard HealthStrip | react-query `refetchInterval` 15s (control-room strip) / 30s (attention) | 1 fetch / 15s |
| Dashboard Control Room page | react-query `refetchInterval` 15s | 1 fetch / 15s |

Overlay/palette polls are on-demand (only while open), not idle.

## No rerender-per-token

- TUI overlay consumes a single `gw.request('control.room.snapshot')` result
  and renders once per snapshot; no stream subscription, no per-message
  re-render. The overlay test asserts one render per snapshot change.
- Dashboard page uses react-query with `data` from a single fetch; tiles are
  plain derived rows, no live token feed.
- Desktop `ControlRoomHome` uses `useGatewayRequest` once; counts are memoised.

## Terminal input latency

- CLI Ctrl+P opens `/control` via the existing command handler; the binding is
  filtered by `_editor_filter` so it never fires mid-prompt. No new input loop.
- TUI Ctrl+P toggles `overlay.controlRoom` in the store (an `$isBlocked`
  aggregate member); key handling follows the existing blocked-overlay path
  (Esc closes) — the same code path as the agents overlay, which has no
  latency regression.
- Status bar tests: 30/30 pass; CLI status bar suite 20/20 pass post-change.

## Bounded list sizes

| Collection | Bound | Location |
|------------|-------|----------|
| Kanban rows per section | `MAX_ROWS_PER_SECTION=50` (plus one probe row for overflow detection) | service.py:48,257 |
| Attention rows | derived from bounded sections, sorted deterministically | service.py:215+ |
| Peer messages | truncated to 24-char id / 70-char title | service.py:298-302 |
| Process command names | truncated to 80 chars | service.py:217 |
| Delegation goal | truncated to 70 chars | service.py:242 |
| Dashboard attention | BFF ranks kanban + agent rows only (no unbounded sources) | backend/control_room.py |

## Baseline regression checks

- Python: `tests/control_room` + `tests/tui_gateway` — 517 pass, 1 skip
- CLI status bar: 20 pass
- TUI vitest: 1547 pass, 12 fail — the 12 are the pre-existing
  scrollBox/virtualHistory/MoA baseline, byte-identical to canonical
- Dashboard BFF pytest: 7 pass; dashboard web vitest: 4 pass

No new idle work, no unbounded structures, no input-latency path changed.
