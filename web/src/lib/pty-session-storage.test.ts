/**
 * Tests for pty-session-storage: per-tab resume-id persistence.
 *
 * Covers the two gaps called out in the maintainer review:
 *   1. Same-profile multi-tab isolation — two tabs must not overwrite each other.
 *   2. Storage → ?resume= restart path — a persisted id is returned verbatim
 *      for URL construction.
 *
 * The vitest environment is "node" (no DOM), so we provide lightweight
 * Storage fakes and mock crypto.randomUUID.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  PTY_SESSION_ID_KEY,
  PTY_TAB_ID_SS_KEY,
  _resetTabIdCache,
  ptyClearStoredSessionId,
  ptyStoredSessionId,
  ptyStoreSessionId,
} from "./pty-session-storage";

// ---------------------------------------------------------------------------
// Minimal localStorage / sessionStorage fakes
// ---------------------------------------------------------------------------

function makeStorage(): Storage {
  const store: Record<string, string> = {};
  return {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => {
      store[k] = v;
    },
    removeItem: (k: string) => {
      delete store[k];
    },
    clear: () => {
      for (const k of Object.keys(store)) delete store[k];
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (i: number) => Object.keys(store)[i] ?? null,
  } as Storage;
}

let fakeLocal: Storage;
let fakeSession: Storage;

beforeEach(() => {
  fakeLocal = makeStorage();
  fakeSession = makeStorage();
  // Inject fakes into the global window used by the module under test.
  vi.stubGlobal("window", {
    localStorage: fakeLocal,
    sessionStorage: fakeSession,
  });
  // Reset module-level tabId cache so each test starts fresh.
  _resetTabIdCache();
});

// ---------------------------------------------------------------------------
// 1. Same-profile multi-tab isolation
// ---------------------------------------------------------------------------

describe("same-profile multi-tab isolation", () => {
  it("two tabs with the same profile scope write independent entries", () => {
    const scope = "profile-alice";
    const tabA = "tab-uuid-aaaa";
    const tabB = "tab-uuid-bbbb";

    ptyStoreSessionId(scope, tabA, "session-from-tab-a");
    ptyStoreSessionId(scope, tabB, "session-from-tab-b");

    // Each tab reads back its own id — no cross-overwrite.
    expect(ptyStoredSessionId(scope, tabA)).toBe("session-from-tab-a");
    expect(ptyStoredSessionId(scope, tabB)).toBe("session-from-tab-b");
  });

  it("overwriting tab A's entry does not affect tab B", () => {
    const scope = "profile-alice";
    const tabA = "tab-uuid-aaaa";
    const tabB = "tab-uuid-bbbb";

    ptyStoreSessionId(scope, tabA, "session-a-v1");
    ptyStoreSessionId(scope, tabB, "session-b-v1");
    // Tab A gets a new session (e.g. after a reconnect).
    ptyStoreSessionId(scope, tabA, "session-a-v2");

    expect(ptyStoredSessionId(scope, tabA)).toBe("session-a-v2");
    expect(ptyStoredSessionId(scope, tabB)).toBe("session-b-v1");
  });

  it("clearing tab A does not affect tab B", () => {
    const scope = "profile-alice";
    const tabA = "tab-uuid-aaaa";
    const tabB = "tab-uuid-bbbb";

    ptyStoreSessionId(scope, tabA, "session-a");
    ptyStoreSessionId(scope, tabB, "session-b");
    ptyClearStoredSessionId(scope, tabA);

    expect(ptyStoredSessionId(scope, tabA)).toBeNull();
    expect(ptyStoredSessionId(scope, tabB)).toBe("session-b");
  });

  it("different profiles are also isolated", () => {
    const tabId = "tab-uuid-shared";

    ptyStoreSessionId("profile-alice", tabId, "session-alice");
    ptyStoreSessionId("profile-bob", tabId, "session-bob");

    expect(ptyStoredSessionId("profile-alice", tabId)).toBe("session-alice");
    expect(ptyStoredSessionId("profile-bob", tabId)).toBe("session-bob");
  });
});

// ---------------------------------------------------------------------------
// 2. Storage → ?resume= restart path
// ---------------------------------------------------------------------------

describe("storage → ?resume= restart path", () => {
  it("returns null when nothing is stored for this tab", () => {
    expect(ptyStoredSessionId("profile-alice", "tab-uuid-aaaa")).toBeNull();
  });

  it("returns the stored session id so the caller can set params.resume", () => {
    const scope = "profile-alice";
    const tabId = "tab-uuid-aaaa";
    const sessionId = "ses_01abc";

    ptyStoreSessionId(scope, tabId, sessionId);

    // Simulate the reconnect path: read stored id → assign to params.resume.
    const params: Record<string, string> = { channel: "ch-xyz" };
    const stored = ptyStoredSessionId(scope, tabId);
    if (stored) params.resume = stored;

    expect(params.resume).toBe(sessionId);
  });

  it("does not set params.resume when no id is stored (clean start)", () => {
    const params: Record<string, string> = { channel: "ch-xyz" };
    const stored = ptyStoredSessionId("profile-alice", "tab-uuid-aaaa");
    if (stored) params.resume = stored;

    expect("resume" in params).toBe(false);
  });

  it("returns null after the entry is cleared (forced-fresh start)", () => {
    const scope = "profile-alice";
    const tabId = "tab-uuid-aaaa";

    ptyStoreSessionId(scope, tabId, "ses_01abc");
    ptyClearStoredSessionId(scope, tabId);

    expect(ptyStoredSessionId(scope, tabId)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 3. Backward-compat: legacy scope-only key migration
// ---------------------------------------------------------------------------

describe("legacy scope-only key migration", () => {
  it("returns the legacy id when no per-tab entry exists yet", () => {
    const scope = "profile-alice";
    const tabId = "tab-uuid-aaaa";
    const legacyId = "ses_legacy";

    // Write using the old format (scope key, no tabId).
    const entries: Record<string, string> = { [scope]: legacyId };
    fakeLocal.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));

    expect(ptyStoredSessionId(scope, tabId)).toBe(legacyId);
  });

  it("migrates the legacy entry to the per-tab key on first read", () => {
    const scope = "profile-alice";
    const tabId = "tab-uuid-aaaa";
    const legacyId = "ses_legacy";

    const entries: Record<string, string> = { [scope]: legacyId };
    fakeLocal.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));

    // First read migrates the entry.
    ptyStoredSessionId(scope, tabId);

    const stored = JSON.parse(
      fakeLocal.getItem(PTY_SESSION_ID_KEY) ?? "{}",
    ) as Record<string, string>;

    // The legacy scope-only key must be removed.
    expect(stored[scope]).toBeUndefined();
    // The per-tab key must hold the migrated value.
    expect(stored[`${scope}::${tabId}`]).toBe(legacyId);
  });

  it("a second tab cannot claim the legacy id after migration", () => {
    const scope = "profile-alice";
    const tabA = "tab-uuid-aaaa";
    const tabB = "tab-uuid-bbbb";
    const legacyId = "ses_legacy";

    // Seed legacy entry and let tab A migrate it.
    const entries: Record<string, string> = { [scope]: legacyId };
    fakeLocal.setItem(PTY_SESSION_ID_KEY, JSON.stringify(entries));
    ptyStoredSessionId(scope, tabA); // migrates

    // Tab B reads after migration — no legacy key left, should get null.
    expect(ptyStoredSessionId(scope, tabB)).toBeNull();
    // Tab A still has its entry.
    expect(ptyStoredSessionId(scope, tabA)).toBe(legacyId);
  });
});

// ---------------------------------------------------------------------------
// 4. getPtyTabId — sessionStorage-backed tab identity
// ---------------------------------------------------------------------------

describe("getPtyTabId (via sessionStorage)", () => {
  it("generates and stores a UUID on first call", async () => {
    // Import dynamically so each test call gets the post-stub module state.
    // The module cache is reset in beforeEach via _resetTabIdCache().
    const { getPtyTabId } = await import("./pty-session-storage");

    const id = getPtyTabId();
    expect(id).toBeTruthy();
    expect(fakeSession.getItem(PTY_TAB_ID_SS_KEY)).toBe(id);
  });

  it("returns the same id on repeated calls (module cache)", async () => {
    const { getPtyTabId } = await import("./pty-session-storage");

    const first = getPtyTabId();
    const second = getPtyTabId();
    expect(first).toBe(second);
  });

  it("reuses the id stored in sessionStorage after a simulated page reload", async () => {
    // Pre-seed sessionStorage as if a prior page load had stored an id.
    const existing = "pre-existing-tab-uuid";
    fakeSession.setItem(PTY_TAB_ID_SS_KEY, existing);

    const { getPtyTabId } = await import("./pty-session-storage");
    expect(getPtyTabId()).toBe(existing);
  });
});
