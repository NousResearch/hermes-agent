// A tab opened before `hermes update` rebuilt the SPA still references the old
// hashed chunks; the rebuild deleted them, so the first lazy page it opens 404s
// and React unmounts the whole app (blank page). A reload picks up the new
// index.html, which is served no-store. One reload per 5 minutes per tab keeps a
// chunk that is really broken from looping.

const RELOAD_AT_KEY = "hermes.chunkLoadReloadAt";
const RELOAD_GUARD_MS = 5 * 60 * 1000;

const CHUNK_LOAD_ERROR = new RegExp(
  "^(Unable to preload CSS for |Failed to fetch dynamically imported module|" +
    "error loading dynamically imported module|Importing a module script failed)",
);

type ReloadStorage = Pick<Storage, "getItem" | "setItem">;

/** A failed load of one of our own build chunks. Safari's message carries no
 *  URL; the others must point at the build's `/assets/`, so a broken plugin
 *  bundle under `/dashboard-plugins/` never triggers a reload. */
export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error) || !CHUNK_LOAD_ERROR.test(error.message)) return false;
  const url = /https?:\/\/\S+|\/\S+/.exec(error.message)?.[0];
  return !url || url.includes("/assets/");
}

function sessionStore(): ReloadStorage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

/** Reload once for a chunk load failure. Returns whether a reload started. */
export function reloadForChunkLoadError(
  error: unknown,
  storage: ReloadStorage | null = sessionStore(),
  reload: () => void = () => window.location.reload(),
  now: number = Date.now(),
): boolean {
  // Without storage there is no loop guard, so keep the error visible instead.
  if (!storage || !isChunkLoadError(error)) return false;
  try {
    const last = Number(storage.getItem(RELOAD_AT_KEY));
    if (last > 0 && now - last < RELOAD_GUARD_MS) return false;
    storage.setItem(RELOAD_AT_KEY, String(now));
  } catch {
    return false;
  }
  reload();
  return true;
}

/** Vite dispatches `vite:preloadError` for every failed lazy import and chunk
 *  CSS preload, including imports nobody awaits. The event is not cancelled,
 *  so the import still rejects as before when no reload happens. */
export function startChunkLoadRecovery(target: Pick<Window, "addEventListener"> = window): void {
  target.addEventListener("vite:preloadError", (event) => {
    reloadForChunkLoadError((event as Event & { payload?: unknown }).payload);
  });
}
