const CHAT_SESSION_PINS_STORAGE_PREFIX = "hermes.chat-session-list.pinned:";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

/** Return the local-only pin store key for one dashboard profile scope. */
export function getChatSessionPinsStorageKey(profile?: string): string {
  return `${CHAT_SESSION_PINS_STORAGE_PREFIX}${encodeURIComponent(profile ?? "")}`;
}

/** Read valid, unique pinned session ids for a profile. */
export function readPinnedSessionIds(
  storage: StorageLike | null | undefined,
  profile?: string,
): string[] {
  if (!storage) return [];

  try {
    const raw = storage.getItem(getChatSessionPinsStorageKey(profile));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return Array.from(
      new Set(parsed.filter((id): id is string => typeof id === "string" && id.length > 0)),
    );
  } catch {
    return [];
  }
}

/** Persist valid, unique pinned session ids for a profile. */
export function writePinnedSessionIds(
  storage: StorageLike | null | undefined,
  profile: string | undefined,
  sessionIds: readonly string[],
): void {
  if (!storage) return;

  try {
    const uniqueIds = Array.from(
      new Set(sessionIds.filter((id) => typeof id === "string" && id.length > 0)),
    );
    storage.setItem(
      getChatSessionPinsStorageKey(profile),
      JSON.stringify(uniqueIds),
    );
  } catch {
    // localStorage may be unavailable or full; pins remain usable in memory.
  }
}

/** Add or remove one session id while preserving the remaining pin order. */
export function togglePinnedSessionIds(
  sessionIds: readonly string[],
  sessionId: string,
): string[] {
  const uniqueIds = Array.from(new Set(sessionIds));
  if (uniqueIds.includes(sessionId)) {
    return uniqueIds.filter((id) => id !== sessionId);
  }
  return [...uniqueIds, sessionId];
}

/** Split a rendered session list into pinned and non-pinned rows exactly once. */
export function partitionPinnedSessions<T extends { id: string }>(
  sessions: readonly T[],
  pinnedSessionIds: readonly string[],
): { pinned: T[]; recent: T[] } {
  const pinned = new Set(pinnedSessionIds);
  const seen = new Set<string>();
  return sessions.reduce<{ pinned: T[]; recent: T[] }>(
    (groups, session) => {
      if (seen.has(session.id)) return groups;
      seen.add(session.id);
      (pinned.has(session.id) ? groups.pinned : groups.recent).push(session);
      return groups;
    },
    { pinned: [], recent: [] },
  );
}
