/**
 * Summarize an overview poll as the ordered list of session ids it returned.
 *
 * Sessions are ordered by last activity, so a conversation continued in
 * another process changes the order of the list. It does not always change
 * the first id: if the head session was also active in the same poll window,
 * it stays first while the sessions behind it swap places. Comparing the whole
 * ordered id list catches both cases. Returns null for an empty list.
 */
export function sessionsSignature(
  sessions: ReadonlyArray<{ id: string }>,
): string | null {
  return sessions.length > 0 ? sessions.map((s) => s.id).join("\n") : null;
}

/**
 * Decide whether the paginated sessions list should be silently
 * re-fetched after an overview poll.
 *
 * The dashboard's FastAPI server and a terminal CLI are separate
 * processes that share the same SQLite session DB. There is no
 * inter-process push channel, so the Sessions page polls the 50 most
 * recently active sessions every few seconds (the "overview" poll). When
 * the ordered ids of that response (see `sessionsSignature`) differ from
 * the previous poll, a session was created or continued in another process
 * and the paginated list is stale — refresh it.
 *
 * Returns false on the very first poll (no baseline yet) and when
 * either signature is null (empty DB / transient empty response), so we
 * never trigger a spurious reload on mount or while the DB is empty.
 */
export function shouldRefreshSessions(
  prevSignature: string | null,
  currentSignature: string | null,
): boolean {
  return (
    prevSignature !== null &&
    currentSignature !== null &&
    prevSignature !== currentSignature
  );
}
