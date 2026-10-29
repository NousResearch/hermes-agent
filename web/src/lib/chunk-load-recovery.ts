// A tab opened before `hermes update` rebuilt the SPA still references the old
// hashed chunks; the rebuild deleted them, so the first lazy page it opens 404s
// and React unmounts the whole app (blank page). A reload picks up the new
// index.html, which is served no-store.
//
// A failed chunk request looks the same whether the build changed or the
// network dropped, so the reload happens only when a fresh index.html names a
// different entry than the one running. Each build reloads at most once per
// tab session: a reload that lands on the same build again (a stale index, a
// chunk that is really broken) keeps the error visible instead of looping,
// while a later build can still recover.

const RELOADED_FROM_KEY = "hermes.chunkLoadReloadedFrom";

const CHUNK_LOAD_ERROR = new RegExp(
  "^(Unable to preload CSS for |Failed to fetch dynamically imported module|" +
    "error loading dynamically imported module|Importing a module script failed)",
);

const ENTRY_SCRIPT = /<script\b[^>]*\btype=["']module["'][^>]*\bsrc=["']([^"']+)["']/i;

type ReloadStorage = Pick<Storage, "getItem" | "setItem">;

export interface ChunkLoadRecoveryDeps {
  storage?: ReloadStorage | null;
  reload?: () => void;
  /** Absolute URL of the entry script this document runs. */
  runningEntry?: () => string | null;
  /** Absolute URL of the entry script a fresh index.html names; null when unknown. */
  publishedEntry?: () => Promise<string | null>;
  online?: () => boolean;
}

/** A failed load of one of our own build chunks. Safari's message carries no
 *  URL; the others must point at the build's `/assets/`, so a broken plugin
 *  bundle under `/dashboard-plugins/` never triggers a reload. */
export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error) || !CHUNK_LOAD_ERROR.test(error.message)) return false;
  const url = /https?:\/\/\S+|\/\S+/.exec(error.message)?.[0];
  return !url || url.includes("/assets/");
}

/** The built entry an index.html references, resolved against `base`. */
export function entryFromIndexHtml(html: string, base: string): string | null {
  const src = ENTRY_SCRIPT.exec(html)?.[1];
  return src ? new URL(src, base).href : null;
}

function sessionStore(): ReloadStorage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function documentEntry(): string | null {
  const script = document.querySelector<HTMLScriptElement>('script[type="module"][src*="/assets/"]');
  return script?.src || null;
}

/** Fetches the URL a reload would load, so both see the same index.html. */
async function freshIndexEntry(): Promise<string | null> {
  const url = window.location.href.split("#")[0];
  try {
    const res = await fetch(url, { cache: "no-store", credentials: "same-origin" });
    return res.ok ? entryFromIndexHtml(await res.text(), res.url || url) : null;
  } catch {
    return null;
  }
}

/** The entry this tab last reloaded away from; undefined when storage throws. */
function reloadedFrom(storage: ReloadStorage): string | null | undefined {
  try {
    return storage.getItem(RELOADED_FROM_KEY);
  } catch {
    return undefined;
  }
}

function markReloadedFrom(storage: ReloadStorage, entry: string): boolean {
  try {
    storage.setItem(RELOADED_FROM_KEY, entry);
    return true;
  } catch {
    return false;
  }
}

/** Reload into a newer build after a chunk load failure. Resolves to whether a
 *  reload started. */
export async function reloadForChunkLoadError(
  error: unknown,
  {
    storage = sessionStore(),
    reload = () => window.location.reload(),
    runningEntry = documentEntry,
    publishedEntry = freshIndexEntry,
    online = () => navigator.onLine !== false,
  }: ChunkLoadRecoveryDeps = {},
): Promise<boolean> {
  // Without storage there is no loop guard, so keep the error visible instead.
  if (!storage || !isChunkLoadError(error) || !online()) return false;
  const running = runningEntry();
  if (!running) return false;
  const guard = reloadedFrom(storage);
  if (guard === undefined || guard === running) return false;
  const published = await publishedEntry();
  if (!published || published === running) return false;
  // Re-check after the fetch: a concurrent failure may already be reloading.
  if (reloadedFrom(storage) === running || !markReloadedFrom(storage, running)) return false;
  reload();
  return true;
}

/** Vite dispatches `vite:preloadError` for every failed lazy import and chunk
 *  CSS preload, including imports nobody awaits. The event is not cancelled,
 *  so the import still rejects as before when no reload happens. */
export function startChunkLoadRecovery(
  target: Pick<Window, "addEventListener"> = window,
  deps: ChunkLoadRecoveryDeps = {},
): void {
  target.addEventListener("vite:preloadError", (event) => {
    void reloadForChunkLoadError((event as Event & { payload?: unknown }).payload, deps);
  });
}
