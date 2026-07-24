export type SessionMessageRefreshMode = "none" | "once" | "poll";

export interface SessionMessageRefreshState {
  isExpanded: boolean;
  hasMessages: boolean;
  wasActive: boolean;
  isActive: boolean;
}

export interface SessionActivitySnapshot {
  id: string;
  is_active: boolean;
  ended_at: number | null;
  last_active: number;
  message_count: number;
}

/** Decide how an expanded row should refresh messages for this render. */
export function getSessionMessageRefreshMode({
  isExpanded,
  hasMessages,
  wasActive,
  isActive,
}: SessionMessageRefreshState): SessionMessageRefreshMode {
  if (!isExpanded) return "none";
  if (isActive) return "poll";
  if (!hasMessages || wasActive) return "once";
  return "none";
}

/** Detect same-ID activity changes that newest-ID-only polling misses. */
export function hasSessionActivityChanged(
  previous: SessionActivitySnapshot[],
  current: SessionActivitySnapshot[],
): boolean {
  const currentById = new Map(current.map((session) => [session.id, session]));
  return previous.some((session) => {
    const next = currentById.get(session.id);
    return (
      next !== undefined &&
      (next.is_active !== session.is_active ||
        next.ended_at !== session.ended_at ||
        next.last_active !== session.last_active ||
        next.message_count !== session.message_count)
    );
  });
}

/** Update visible activity metadata without replacing row order/search membership. */
export function reconcileSessionActivity<T extends SessionActivitySnapshot>(
  visible: T[],
  current: SessionActivitySnapshot[],
): T[] {
  const currentById = new Map(current.map((session) => [session.id, session]));
  return visible.map((session) => {
    const next = currentById.get(session.id);
    if (!next) return session;
    return {
      ...session,
      is_active: next.is_active,
      ended_at: next.ended_at,
      last_active: next.last_active,
      message_count: next.message_count,
    };
  });
}

/**
 * Decide whether the paginated sessions list should be silently
 * re-fetched after an overview poll.
 *
 * Returns false on the first poll and for transient empty responses.
 */
export function shouldRefreshSessions(
  prevNewestId: string | null,
  currentNewestId: string | null,
): boolean {
  return (
    prevNewestId !== null &&
    currentNewestId !== null &&
    prevNewestId !== currentNewestId
  );
}
