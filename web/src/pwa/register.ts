import { HERMES_BASE_PATH } from "@/lib/api";

export interface ServiceWorkerRegistrar {
  register(
    scriptURL: string | URL,
    options?: RegistrationOptions,
  ): Promise<ServiceWorkerRegistration>;
}

interface RegisterHermesServiceWorkerOptions {
  isProduction?: boolean;
  runtimeBasePath?: string;
  viteBaseUrl?: string;
  serviceWorker?: ServiceWorkerRegistrar;
  warn?: (message: string) => void;
}

export interface ServiceWorkerTarget {
  scope: string;
  scriptUrl: string;
}

function normalizeBasePath(value: string): string {
  let pathname = "/";
  try {
    pathname = new URL(value || "/", "https://hermes.invalid/").pathname;
  } catch {
    pathname = "/";
  }

  const segments = pathname.split("/").filter(Boolean);
  return segments.length > 0 ? `/${segments.join("/")}/` : "/";
}

export function resolveServiceWorkerTarget(
  runtimeBasePath: string,
  viteBaseUrl: string,
): ServiceWorkerTarget {
  const scope = normalizeBasePath(runtimeBasePath || viteBaseUrl || "/");
  return {
    scope,
    scriptUrl: `${scope}service-worker.js`,
  };
}

export async function registerHermesServiceWorker(
  options: RegisterHermesServiceWorkerOptions = {},
): Promise<ServiceWorkerRegistration | undefined> {
  const isProduction = options.isProduction ?? import.meta.env.PROD;
  if (!isProduction) return undefined;

  const serviceWorker =
    options.serviceWorker ??
    (typeof navigator !== "undefined" && "serviceWorker" in navigator
      ? navigator.serviceWorker
      : undefined);
  if (!serviceWorker) return undefined;

  const target = resolveServiceWorkerTarget(
    options.runtimeBasePath ?? HERMES_BASE_PATH,
    options.viteBaseUrl ?? import.meta.env.BASE_URL,
  );

  try {
    return await serviceWorker.register(target.scriptUrl, {
      scope: target.scope,
      updateViaCache: "none",
    });
  } catch {
    const warn = options.warn ?? console.warn;
    warn("Hermes PWA service worker registration failed.");
    return undefined;
  }
}
