/* Hermes dashboard service worker.
 * Minimal PWA SW: enables Chrome "Install app" (installability requires a
 * registered SW with a fetch handler) and makes the standalone app load its
 * hashed bundles offline. Assets are cache-first; index.html and /api/* stay
 * network-first so sessions/auth never go stale.
 */
const CACHE = "hermes-dashboard-v1";

// Hashed, immutable bundles and static skin assets — safe to cache-first.
const ASSET_PREFIXES = ["/assets/", "/fonts/", "/fonts-terminal/", "/favicon.ico",
  "/icon-192.png", "/icon-512.png", "/manifest.webmanifest", "/ds-assets/"];

self.addEventListener("install", (event) => {
  // Don't let a half-installed SW hold the page; activate ASAP.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll([
      "/index.html",
      "/icon-192.png",
      "/icon-512.png",
      "/manifest.webmanifest",
    ])).catch(() => undefined),
  );
  // Take control of already-open pages immediately.
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const isAsset = ASSET_PREFIXES.some((p) => url.pathname.startsWith(p));
  const isShell = url.pathname === "/" || url.pathname === "/index.html";

  if (isAsset) {
    event.respondWith(cacheFirst(req, url));
  } else if (isShell) {
    event.respondWith(networkFirst(req, url));
  }
  // /api/* and everything else: network only (no stale session/auth data).
});

async function cacheFirst(req, url) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url.pathname);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok && res.status < 300) cache.put(url.pathname, res.clone());
  return res;
}

async function networkFirst(req, url) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok && res.status < 300) cache.put(url.pathname, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(url.pathname);
    if (hit) return hit;
    throw err;
  }
}
