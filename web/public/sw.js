self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", () => {
  // Registration-only service worker: keeps the app installable without
  // pretending voice/WebRTC can work offline or from stale cached assets.
});