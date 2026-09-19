/*
 * Hermes dashboard service worker.
 *
 * Purpose is installability + fast repeat loads on phone networks, NOT offline
 * operation: the dashboard is a live control plane for a running agent, so
 * there is nothing useful to do without the backend. The rules below are
 * deliberately narrow so a stale cache can never strand the app.
 *
 * Never intercepted (always straight to network):
 *   - anything that is not a same-origin GET (WS upgrades, POST/PUT/DELETE)
 *   - /api/*            — live agent state, secrets, session token auth
 *   - /dashboard-plugins/*  — operator-supplied scripts, must never go stale
 *   - /auth/*           — the gated-mode login/callback flow
 *   - index.html / navigations — network-first; the server injects a
 *     per-process session token into the HTML, so a cached copy would carry a
 *     dead credential. Only the static offline notice is ever served instead.
 *
 * Cached:
 *   - /assets/*  cache-first. Vite content-hashes these filenames, so a given
 *                URL is immutable by construction and a hit is always correct.
 *   - fonts, /ds-assets/*, /icons/*, favicon  stale-while-revalidate. Not
 *                hashed, but changing rarely and cheap to refresh in the
 *                background.
 *
 * Bump CACHE_VERSION whenever the caching rules change; activate() drops every
 * cache that does not match, so an old worker's entries never survive.
 */

const CACHE_VERSION = "hermes-dash-v1";
const IMMUTABLE_CACHE = `${CACHE_VERSION}-immutable`;
const REVALIDATE_CACHE = `${CACHE_VERSION}-static`;

/** Path prefix this worker is registered under ("" at root, "/hermes" behind a
 *  reverse-proxy prefix). Derived from the registration scope so the same file
 *  works in both deployments with no build-time substitution. */
const SCOPE_PATH = new URL(self.registration.scope).pathname.replace(/\/+$/, "");

const OFFLINE_URL = `${SCOPE_PATH}/offline.html`;

/** Prefixes that must always reach the network. */
const NEVER_CACHE = ["/api/", "/dashboard-plugins/", "/auth/"];

/** Prefixes safe to serve cache-first (content-hashed filenames). */
const IMMUTABLE_PREFIXES = ["/assets/"];

/** Prefixes served stale-while-revalidate (stable URLs, infrequent changes). */
const REVALIDATE_PREFIXES = [
  "/fonts/",
  "/fonts-terminal/",
  "/ds-assets/",
  "/icons/",
];

/** Exact paths served stale-while-revalidate. */
const REVALIDATE_EXACT = ["/favicon.ico", "/manifest.webmanifest"];

/** Strip the scope prefix so the rule tables above stay prefix-agnostic. */
function scopedPath(url) {
  const path = url.pathname;
  if (SCOPE_PATH && path.startsWith(SCOPE_PATH)) {
    return path.slice(SCOPE_PATH.length) || "/";
  }
  return path;
}

function matchesPrefix(path, prefixes) {
  return prefixes.some((prefix) => path.startsWith(prefix));
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(REVALIDATE_CACHE);
      // cache.add() rejects the whole install on a single miss; the offline
      // notice is a nicety, so a failed fetch must not block activation.
      try {
        await cache.add(new Request(OFFLINE_URL, { cache: "reload" }));
      } catch {
        /* offline notice unavailable — navigations just surface the browser's */
      }
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([IMMUTABLE_CACHE, REVALIDATE_CACHE]);
      const names = await caches.keys();
      await Promise.all(
        names.filter((name) => !keep.has(name)).map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

/** Only a plain 200 is storable: cache.put() rejects 206 range responses, and
 *  an opaque cross-origin response would be cached as an unreadable stub. */
function storable(response) {
  return response.status === 200 && response.type !== "opaque";
}

async function cacheFirst(event, request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const response = await fetch(request);
  // waitUntil, not a bare promise: the fetch event may be terminated as soon
  // as respondWith settles, which would cancel the write mid-flight.
  if (storable(response)) event.waitUntil(cache.put(request, response.clone()));
  return response;
}

async function staleWhileRevalidate(event, request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  const network = fetch(request)
    .then(async (response) => {
      if (storable(response)) await cache.put(request, response.clone());
      return response;
    })
    .catch(() => undefined);
  // Keep the worker alive for the refresh even though the cached copy is
  // already on its way back to the page.
  event.waitUntil(network);
  if (hit) return hit;
  const response = await network;
  if (response) return response;
  return new Response("", { status: 504, statusText: "Offline" });
}

async function navigateNetworkFirst(request) {
  try {
    return await fetch(request);
  } catch {
    const cache = await caches.open(REVALIDATE_CACHE);
    const offline = await cache.match(OFFLINE_URL);
    if (offline) return offline;
    throw new Error("offline and no offline notice cached");
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  let url;
  try {
    url = new URL(request.url);
  } catch {
    return;
  }
  if (url.origin !== self.location.origin) return;

  const path = scopedPath(url);
  if (matchesPrefix(path, NEVER_CACHE)) return;

  if (request.mode === "navigate") {
    event.respondWith(navigateNetworkFirst(request));
    return;
  }

  if (matchesPrefix(path, IMMUTABLE_PREFIXES)) {
    event.respondWith(cacheFirst(event, request, IMMUTABLE_CACHE));
    return;
  }

  if (matchesPrefix(path, REVALIDATE_PREFIXES) || REVALIDATE_EXACT.includes(path)) {
    event.respondWith(staleWhileRevalidate(event, request, REVALIDATE_CACHE));
  }
});
