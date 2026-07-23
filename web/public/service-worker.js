/*
 * Hermes dashboard PWA worker.
 *
 * The dashboard HTML contains an ephemeral session bootstrap token, so this
 * worker deliberately has no app-shell or offline-document cache. Only the
 * explicit public PWA files below and immutable, content-hashed Vite assets
 * are eligible for Cache Storage.
 */

const HERMES_PWA_CACHE_PREFIX = "hermes-pwa-static-";
const HERMES_PWA_CACHE_NAME = `${HERMES_PWA_CACHE_PREFIX}v1`;
const PUBLIC_PWA_FILES = new Set([
  "manifest.webmanifest",
  "icons/hermes-192.png",
  "icons/hermes-512.png",
]);
const HASHED_VITE_ASSET =
  /^assets\/[^/]+-[A-Za-z0-9_-]{8,}\.(?:css|js|mjs|woff|woff2|png|jpe?g|webp|svg)$/;

function relativeScopePath(url) {
  const scopeUrl = new URL(self.registration.scope);
  if (url.origin !== scopeUrl.origin) return null;

  const scopePath = scopeUrl.pathname.endsWith("/")
    ? scopeUrl.pathname
    : `${scopeUrl.pathname}/`;
  if (!url.pathname.startsWith(scopePath)) return null;

  return url.pathname.slice(scopePath.length);
}

function isCacheEligible(request) {
  if (request.method !== "GET") return false;
  if (request.mode === "navigate" || request.destination === "document") {
    return false;
  }

  const url = new URL(request.url);
  if (url.origin !== self.location.origin || url.search) return false;

  const relativePath = relativeScopePath(url);
  if (relativePath === null) return false;

  return PUBLIC_PWA_FILES.has(relativePath) || HASHED_VITE_ASSET.test(relativePath);
}

async function cacheFirst(request) {
  const cache = await caches.open(HERMES_PWA_CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok && response.type === "basic") {
    await cache.put(request, response.clone());
  }
  return response;
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames
          .filter(
            (name) =>
              name.startsWith(HERMES_PWA_CACHE_PREFIX) &&
              name !== HERMES_PWA_CACHE_NAME,
          )
          .map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  if (!isCacheEligible(event.request)) return;
  event.respondWith(cacheFirst(event.request));
});
