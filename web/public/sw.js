// Minimal service worker for Hermes Agent's mobile PWA shell.
//
// Scope: cache the app shell (index.html + built JS/CSS chunks) so the icon
// launches instantly and the browser treats this as installable. This does
// NOT cache API/WebSocket traffic — /api/* and /api/ws always hit the
// network live; caching agent responses would be actively wrong (stale
// chat state, stale auth). Offline is "the shell loads, then shows a
// connection error" rather than a fake cached conversation.

const CACHE_NAME = "hermes-shell-v1";
const SHELL_PATHS = ["/", "/mobile-chat", "/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_PATHS))
      .catch(() => {
        // Best-effort: a missing shell path (e.g. auth redirect on first
        // install) must not block SW installation.
      }),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept API calls or the WebSocket upgrade — those must always
  // be live network requests.
  if (url.pathname.startsWith("/api/")) {
    return;
  }

  // Network-first for navigations/assets, falling back to cache when
  // offline. Keeps the shell usable without ever serving stale JS after a
  // rebuild (cache is only the offline fallback, not the primary source).
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          if (event.request.method === "GET" && response.ok) {
            cache.put(event.request, copy);
          }
        });
        return response;
      })
      .catch(() => caches.match(event.request)),
  );
});
