// Minimal service worker — makes Hermes installable as a PWA without caching
// application code, so dashboard updates (new build hashes after `hermes
// update`) are never served stale. Network requests pass straight through.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
// A fetch handler is required for installability; pass through to the network.
self.addEventListener("fetch", () => {});
