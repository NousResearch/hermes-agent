/**
 * PTY attach-token helpers — scoped per-browser keep-alive identity.
 *
 * A stable token lets a page refresh or transient WebSocket drop reattach
 * to the same live PTY process instead of spawning a fresh one. The token
 * is stored in localStorage so other devices cannot grab it.
 *
 * ``rotate`` mints a new token — used when the user explicitly starts a
 * fresh session so the old keep-alive PTY is NOT reattached.
 *
 * ``scope`` (e.g. "profile\0resume") scopes the token to a profile+session
 * combination so switching either one forces a new PTY spawn instead of
 * reattaching to a stale PTY with the wrong HERMES_HOME / session.
 */

export const PTY_ATTACH_TOKEN_BASE = "hermes.pty.token";

export interface PtyTokenStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface PtyTokenCrypto {
  getRandomValues<T extends Uint8Array>(array: T): T;
}

const DEFAULT_STORAGE: PtyTokenStorage =
  typeof window !== "undefined"
    ? {
        getItem(key: string) {
          return window.localStorage.getItem(key);
        },
        setItem(key: string, value: string) {
          window.localStorage.setItem(key, value);
        },
      }
    : { getItem: () => null, setItem: () => {} };

const DEFAULT_CRYPTO: PtyTokenCrypto =
  typeof crypto !== "undefined" && "getRandomValues" in crypto
    ? (crypto as PtyTokenCrypto)
    : {
        getRandomValues<T extends Uint8Array>(array: T): T {
          for (let i = 0; i < array.length; i++) {
            array[i] = Math.floor(Math.random() * 256);
          }
          return array;
        },
      };

/**
 * Returns a stable hex token for PTY attach.
 *
 * @param rotate  — if true, always mint a fresh token.
 * @param scope   — scoping suffix (e.g. "profile\0resume").
 * @param storage — injectable localStorage-like storage (defaults to window.localStorage).
 * @param crypto  — injectable crypto (defaults to globalThis.crypto).
 */
export function ptyAttachToken(
  rotate = false,
  scope = "",
  storage: PtyTokenStorage = DEFAULT_STORAGE,
  rand: PtyTokenCrypto = DEFAULT_CRYPTO,
): string {
  const key = scope
    ? `${PTY_ATTACH_TOKEN_BASE}.${scope}`
    : `${PTY_ATTACH_TOKEN_BASE}.chat`;

  let t = "";
  if (!rotate) {
    try {
      t = storage.getItem(key) ?? "";
    } catch {
      /* private mode / storage blocked */
    }
  }
  if (!t) {
    const a = new Uint8Array(16);
    rand.getRandomValues(a);
    t = Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
    try {
      storage.setItem(key, t);
    } catch {
      /* ignore */
    }
  }
  return t;
}
