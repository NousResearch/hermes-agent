// Lightweight PWA bootstrap for the Hermes Dashboard.
//
// Important: the dashboard injects auth/session bootstrap data into
// index.html. The service worker intentionally does NOT cache navigation
// documents or /api responses so a Home Screen launch always receives a fresh
// token/cookie-gate state from the Python backend.

declare global {
  interface Window {
    __HERMES_BASE_PATH__?: string;
  }
}

function normalizeBasePath(raw: string | undefined): string {
  if (!raw) return "";
  const withLead = raw.startsWith("/") ? raw : `/${raw}`;
  return withLead.replace(/\/+$/, "");
}

export function registerHermesPwa(): void {
  if (typeof window === "undefined" || !("serviceWorker" in navigator)) {
    return;
  }

  const basePath = normalizeBasePath(window.__HERMES_BASE_PATH__);
  const swUrl = `${basePath}/sw.js`;
  const scope = basePath ? `${basePath}/` : "/";

  window.addEventListener("load", () => {
    navigator.serviceWorker.register(swUrl, { scope }).catch((error: unknown) => {
      // PWA support is a convenience layer. Never block the dashboard if a
      // browser, reverse proxy, or development server refuses registration.
      console.warn("[Hermes PWA] Service worker registration failed", error);
    });
  });
}
