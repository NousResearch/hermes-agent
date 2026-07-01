/**
 * Decide whether the paginated sessions list should be silently
 * re-fetched after an overview poll.
 *
 * The dashboard's FastAPI server and a terminal CLI are separate
 * processes that share the same SQLite session DB. There is no
 * inter-process push channel, so the Sessions page polls the 50 newest
 * sessions every few seconds (the "overview" poll). When that poll
 * surfaces a new session at the head of the list, or detects visible
 * changes to an active session already in the list, the paginated list
 * is stale and should be refreshed silently.
 *
 * Returns false on the very first poll (no baseline yet) and when the
 * current response is empty, so we never trigger a spurious reload on
 * mount or while the DB is empty.
 */
export interface SessionRefreshSnapshotItem {
  id: string;
  title?: string | null;
  last_active?: number | null;
  ended_at?: number | null;
  is_active?: boolean;
  message_count?: number | null;
  tool_call_count?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  preview?: string | null;
}

export function shouldRefreshSessions(
  prevSessions: readonly SessionRefreshSnapshotItem[] | null,
  currentSessions: readonly SessionRefreshSnapshotItem[],
): boolean {
  if (prevSessions === null || currentSessions.length === 0) {
    return false;
  }
  if (prevSessions.length === 0) {
    return true;
  }

  const prevNewestId = prevSessions[0]?.id ?? null;
  const currentNewestId = currentSessions[0]?.id ?? null;
  if (prevNewestId === null || currentNewestId === null) {
    return false;
  }
  if (prevNewestId !== currentNewestId) {
    return true;
  }

  const prevById = new Map(prevSessions.map((session) => [session.id, session]));
  return currentSessions.some((current) => {
    const previous = prevById.get(current.id);
    if (!previous || (!previous.is_active && !current.is_active)) {
      return false;
    }

    return hasVisibleSessionChange(previous, current);
  });
}

function hasVisibleSessionChange(
  previous: SessionRefreshSnapshotItem,
  current: SessionRefreshSnapshotItem,
): boolean {
  return (
    previous.title !== current.title ||
    previous.last_active !== current.last_active ||
    previous.ended_at !== current.ended_at ||
    previous.is_active !== current.is_active ||
    previous.message_count !== current.message_count ||
    previous.tool_call_count !== current.tool_call_count ||
    previous.input_tokens !== current.input_tokens ||
    previous.output_tokens !== current.output_tokens ||
    previous.preview !== current.preview
  );
}
