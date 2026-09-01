import { HERMES_BASE_PATH } from "@/lib/api";

export function serviceWorkerUrl(basePath: string): string {
  const normalized = basePath.replace(/\/+$/, "");
  return `${normalized}/sw.js` || "/sw.js";
}

export function serviceWorkerScope(basePath: string): string {
  const normalized = basePath.replace(/\/+$/, "");
  return `${normalized || ""}/`;
}

export function registerHermesServiceWorker(): void {
  if (!import.meta.env.PROD || typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;
  void navigator.serviceWorker.register(serviceWorkerUrl(HERMES_BASE_PATH), {
    scope: serviceWorkerScope(HERMES_BASE_PATH),
  }).catch(() => {
    // PWA is an enhancement; authentication and the dashboard remain usable
    // when the browser or deployment forbids service workers.
  });
}
