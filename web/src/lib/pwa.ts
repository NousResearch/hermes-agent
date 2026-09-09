/**
 * Service-worker registration for the dashboard PWA.
 *
 * The worker (``web/public/sw.js``) exists so the dashboard is installable to a
 * phone or laptop home screen and so repeat loads over a phone network skip
 * re-downloading the bundle. It caches static assets only — never ``/api``, so
 * no agent state or session token is ever written to CacheStorage.
 *
 * Registration is skipped in dev: the Vite dev server serves modules unhashed
 * and a worker holding them would fight HMR. Any worker left over from a
 * production build on the same origin is unregistered so ``npm run dev`` on
 * port 5173 cannot inherit stale caches from a previous ``hermes dashboard``.
 */

import { HERMES_BASE_PATH } from "@/lib/api";

/** Worker script URL, honouring an ``X-Forwarded-Prefix`` reverse-proxy mount. */
export function serviceWorkerUrl(basePath: string = HERMES_BASE_PATH): string {
  return `${basePath}/sw.js`;
}

/** Registration scope — the worker may only control its own mount subtree. */
export function serviceWorkerScope(basePath: string = HERMES_BASE_PATH): string {
  return `${basePath}/`;
}

function supported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "serviceWorker" in navigator &&
    typeof window !== "undefined" &&
    window.isSecureContext
  );
}

async function unregisterAll(): Promise<void> {
  const registrations = await navigator.serviceWorker.getRegistrations();
  await Promise.all(registrations.map((registration) => registration.unregister()));
}

/**
 * Register the worker (production) or tear down a stale one (dev).
 *
 * Never throws: a dashboard that cannot register a worker is fully functional,
 * so every failure is swallowed after a console warning.
 */
export async function registerServiceWorker(
  isProduction: boolean = import.meta.env.PROD,
): Promise<ServiceWorkerRegistration | undefined> {
  if (!supported()) return undefined;
  try {
    if (!isProduction) {
      await unregisterAll();
      return undefined;
    }
    return await navigator.serviceWorker.register(serviceWorkerUrl(), {
      scope: serviceWorkerScope(),
    });
  } catch (err) {
    console.warn("[hermes] service worker registration failed", err);
    return undefined;
  }
}
