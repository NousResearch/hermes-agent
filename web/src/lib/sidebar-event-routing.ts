/**
 * Pure session-routing decisions for the dashboard ChatSidebar's event feed.
 *
 * The sidebar subscribes to one PTY child's channel via `/api/events`. A single
 * embedded `hermes --tui` can hold several sessions at once (the user runs
 * `/new` or switches sessions inside it), and EVERY session in that child emits
 * its `tool.*` frames on the SAME channel. Without filtering, a backgrounded
 * session's tool calls render in the focused session's tool list — the web
 * analogue of the desktop/TUI cross-session bleed (issue #49106 / #47709).
 *
 * The fix mirrors the TUI's match-by-id guard
 * (`ui-tui/src/app/createGatewayEventHandler.ts`): learn the live runtime
 * session id from `session.info` frames, then drop any scoped frame whose
 * `session_id` differs from it. Frames without a `session_id`, and every frame
 * arriving before the live id is learned, fall through — so we never swallow
 * the legitimate live feed (the #42359 regression that reverted #42178).
 */

/** Control events that are never session-scoped and must never be gated. */
const UNSCOPED_CONTROL_EVENTS = new Set<string>(["dashboard.new_session_requested"]);

/**
 * Whether a channel-feed frame must be dropped because it belongs to a
 * different session than the one currently live in this sidebar's PTY child.
 *
 * @param eventType        the frame's `type` (e.g. "tool.start", "session.info")
 * @param frameSessionId   the frame's `session_id`, if the gateway stamped one
 * @param liveSessionId    the live runtime session id learned from `session.info`
 */
export function shouldDropSidebarEvent(
  eventType: string | undefined,
  frameSessionId: string | undefined,
  liveSessionId: string | null,
): boolean {
  if (eventType && UNSCOPED_CONTROL_EVENTS.has(eventType)) {
    return false;
  }

  // Unscoped frame, or we haven't learned the live id yet: let it through so
  // the legitimate live feed is never swallowed.
  if (!frameSessionId || !liveSessionId) {
    return false;
  }

  return frameSessionId !== liveSessionId;
}

/**
 * The live runtime session id after observing one frame. The sidebar tracks
 * this in a ref; this helper keeps the "learn from session.info" rule in one
 * testable place. Returns the new id when a `session.info` frame carries one,
 * otherwise the unchanged previous value.
 */
export function nextLiveSessionId(
  eventType: string | undefined,
  frameSessionId: string | undefined,
  prev: string | null,
): string | null {
  if (eventType === "session.info" && frameSessionId) {
    return frameSessionId;
  }

  return prev;
}
