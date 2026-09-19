/**
 * JARVIS Live World Feed — Client-side persistence cache (localStorage)
 * Stale-while-revalidate: instant render from cache on load, then a silent
 * background refresh for any dataset whose TTL has expired. This avoids
 * re-hitting all live endpoints on every page load and reload.
 */

export interface CachedFeed<T> {
  data: T;
  fetchedAt: number;
  ttlMs: number;
}

const STORAGE_PREFIX = 'jarvis_live_feed_cache_v1__';

function getStorage(): Storage | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

function readEntry(key: string): CachedFeed<unknown> | null {
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(STORAGE_PREFIX + key);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) {
    console.warn('[LiveFeedCache] read failed:', e);
    return null;
  }
}

/** Return cached data for `key` (may be stale — caller decides), or null if none. */
export function getFeed<T>(key: string): T | null {
  const entry = readEntry(key);
  return entry ? (entry.data as T) : null;
}

/** Persist `data` under `key` with the given TTL. */
export function setFeed<T>(key: string, data: T, ttlMs: number): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    const entry: CachedFeed<T> = { data, fetchedAt: Date.now(), ttlMs };
    storage.setItem(STORAGE_PREFIX + key, JSON.stringify(entry));
  } catch (e) {
    console.warn('[LiveFeedCache] write failed:', e);
  }
}

/** True when a non-expired entry exists for `key`. */
export function isFeedFresh(key: string): boolean {
  const entry = readEntry(key);
  if (!entry) return false;
  const ageMs = Date.now() - entry.fetchedAt;
  return ageMs >= 0 && ageMs < entry.ttlMs;
}

/** True when ANY cached entry exists (even stale) for any given key. */
export function hasAnyFeed(keys: string[]): boolean {
  return keys.some((key) => getFeed(key) !== null);
}
