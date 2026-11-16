export function normalizeSessionTitle(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const title = raw.trim();
  return title ? title : null;
}

export function titleFromSessionInfoPayload(
  payload: unknown,
): string | null | undefined {
  if (!payload || typeof payload !== "object" || !("title" in payload)) {
    return undefined;
  }

  return normalizeSessionTitle((payload as { title?: unknown }).title);
}

/** The stored session id a PTY `session.info` names — the session the embedded TUI is actually
 *  running, which can differ from `?resume=` after `/resume`, `/new`, `/branch` or the server's
 *  active-session fallback. */
export function storedSessionIdFromSessionInfoPayload(payload: unknown): string | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  const id = (payload as { stored_session_id?: unknown }).stored_session_id;
  return typeof id === "string" && id ? id : undefined;
}

/** The PTY session a ``session.info`` reported, keyed to the PTY process it came from. */
export interface LiveChatSession {
  scope: string
  id: string
}

/** The session the embedded TUI is actually running, else ``?resume=``. They differ after ``/resume``,
 *  ``/new`` or ``/branch`` inside the PTY, or the server's active-session fallback (#94716). The live id is
 *  scoped to the PTY process, not the socket: a transport reconnect reattaches the same PTY, whose TUI does
 *  not re-send session.info; a fresh PTY or a new channel drops it. */
export function activeChatSessionId(live: LiveChatSession | null, scope: string, resume: string | null): string | null {
  return (live?.scope === scope ? live.id : null) ?? resume
}

/** The next live-session state for an id reported on ``scope``; the same object when nothing changed. */
export function nextLiveSession(prev: LiveChatSession | null, scope: string, id: string): LiveChatSession {
  return prev?.scope === scope && prev.id === id ? prev : { scope, id }
}
