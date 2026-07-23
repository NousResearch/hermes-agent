/**
 * PTY session-ID persistence helpers.
 *
 * Why: A PTY process can die (agent exit, gateway restart) and needs to resume
 * the conversation. The frontend persists the last-seen session ID in
 * localStorage keyed by BOTH profile scope AND per-tab id so that two
 * same-profile tabs never overwrite each other's resume id.
 *
 * Key design:
 *   localStorage["hermes.pty.sessionId"] = JSON object of shape
 *     { "<scope>::<tabId>": "<sessionId>", ... }
 *
 *   Tab identity ("tabId") lives in sessionStorage["hermes.pty.tabId"].
 *   sessionStorage is per-tab and survives same-tab page reloads, so:
 *     - a same-tab reload keeps the same tabId → resumes the same conversation
 *     - a new tab gets a fresh tabId → starts independently
 *
 * Test: see pty-session-storage.test.ts
 */

export const PTY_SESSION_ID_KEY = "hermes.pty.sessionId";
export const PTY_TAB_ID_SS_KEY = "hermes.pty.tabId";

// Module-level cache: avoids repeated sessionStorage reads within one page load.
let _cachedTabId: string | null = null;

/**
 * Why: Provides a stable, per-tab random id that persists across same-tab
 * reloads (via sessionStorage) but is fresh for every new tab.
 * What: Lazily generates a UUID, writes it to sessionStorage, and caches it.
 * Test: Clear sessionStorage and module cache; first call should produce a UUID
 * that equals sessionStorage["hermes.pty.tabId"]; a second call with the cache
 * cleared should return the same UUID that was stored.
 */
export function getPtyTabId(): string {
  if (_cachedTabId) return _cachedTabId;
  try {
    let id = window.sessionStorage.getItem(PTY_TAB_ID_SS_KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.sessionStorage.setItem(PTY_TAB_ID_SS_KEY, id);
    }
    _cachedTabId = id;
  } catch {
    // sessionStorage blocked (private mode edge case): fall back to a
    // module-lifetime random so at least within this page load the id is
    // stable and consistent.
    _cachedTabId = crypto.randomUUID();
  }
  return _cachedTabId;
}

/** Exposed for tests to reset the in-process module cache between cases. */
export function _resetTabIdCache(): void {
  _cachedTabId = null;
}

/**
 * Why: Retrieves this tab's persisted session id so a reconnect can pass
 * ?resume= and continue the conversation rather than starting fresh.
 * What: Reads localStorage, returns the value for `<scope>::<tabId>`.
 *   Falls back to the legacy `<scope>` key (written by the old single-key
 *   format) for the first reconnect after upgrade; migrates that entry to the
 *   per-tab key so a second same-profile tab cannot steal it.
 * Test: Store a per-tab entry, assert it is returned. Store only a legacy
 *   scope-only entry, assert it is returned AND migrated to the per-tab key.
 */
export function ptyStoredSessionId(
  scope: string,
  tabId: string,
): string | null {
  try {
    const raw = window.localStorage.getItem(PTY_SESSION_ID_KEY);
    if (!raw) return null;
    const entries = JSON.parse(raw) as Record<string, string>;
    const perTabKey = `${scope}::${tabId}`;
    if (entries[perTabKey] !== undefined) return entries[perTabKey];
    // Legacy migration: old format used scope-only key, shared by all tabs.
    if (entries[scope] !== undefined) {
      const legacyId = entries[scope];
      entries[perTabKey] = legacyId;
      delete entries[scope];
      window.localStorage.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));
      return legacyId;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Why: Persists the session id so a later reconnect (same tab) can resume.
 * What: Writes `<scope>::<tabId>` → sessionId into the shared localStorage
 *   object; other tabs' entries (different tabId) are left untouched.
 * Test: Two tabs (different tabIds) write different ids; assert each entry
 *   exists independently under its own key.
 */
export function ptyStoreSessionId(
  scope: string,
  tabId: string,
  sessionId: string,
): void {
  try {
    const raw = window.localStorage.getItem(PTY_SESSION_ID_KEY);
    const entries: Record<string, string> = raw ? JSON.parse(raw) : {};
    entries[`${scope}::${tabId}`] = sessionId;
    window.localStorage.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));
  } catch {
    /* localStorage blocked */
  }
}

/**
 * Why: Clears this tab's resume id when the user starts a forced-fresh
 *   session so the next spawn does not accidentally resume the old one.
 * What: Deletes only the `<scope>::<tabId>` entry; other tabs are unaffected.
 * Test: Store two per-tab entries; clear one; assert only that one is gone.
 */
export function ptyClearStoredSessionId(scope: string, tabId: string): void {
  try {
    const raw = window.localStorage.getItem(PTY_SESSION_ID_KEY);
    if (!raw) return;
    const entries: Record<string, string> = JSON.parse(raw);
    delete entries[`${scope}::${tabId}`];
    window.localStorage.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));
  } catch {
    /* localStorage blocked */
  }
}
