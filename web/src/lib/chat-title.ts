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
