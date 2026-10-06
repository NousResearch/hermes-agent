import type { AnyGatewayEvent } from '../gatewayTypes.js'

// Surface-global event types: emitted without a session (or with an explicitly
// empty one) on purpose — the change watcher, the skin loader, pet progress and
// the billing step-up flow broadcast to every connected client. They must never
// be dropped for session mismatch, even when they carry an empty session_id.
const GLOBAL_PREFIXES = ['gateway.', 'pet.', 'skin.', 'billing.']

// Fail-closed session filter for gateway events (#51058).
//
// `sid` is momentarily null during a session switch/reset (resetSession →
// activate → setHistoryItems), and the backend's `_emit()` can stamp an
// explicitly empty `session_id` when a caller omits one. The old
// `ev.session_id && sid && ev.session_id !== sid` guard short-circuited to
// false in both windows, so events from another concurrently-live session
// bled into the active transcript.
//
// This predicate inverts the default: an event that carries a session_id key
// is dropped unless it provably belongs to the active session — when there is
// no active session (`sid` falsy), when the event's id is empty, or when it
// differs from the active one. A keyless event stays unscoped-by-design
// (CLI-direct / global) and passes. The null-sid window is deliberately
// conservative: with nothing to route a session-scoped event to, all of them
// drop; the window is transient and the streaming buffer is flushed on switch,
// so nothing is lost that the resumed session will not re-send.
export function isForeignSessionEvent(ev: AnyGatewayEvent, activeSid: string | null): boolean {
  if (!Object.prototype.hasOwnProperty.call(ev, 'session_id')) {
    return false
  }

  if (GLOBAL_PREFIXES.some(p => ev.type.startsWith(p))) {
    return false
  }

  const evSid = ev.session_id

  return !activeSid || !evSid || evSid !== activeSid
}
