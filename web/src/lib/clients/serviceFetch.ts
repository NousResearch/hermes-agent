import { AuthUnauthorizedError, HERMES_BASE_PATH } from "@/lib/api";

export interface DashboardServiceClientConfig {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export class DashboardServiceError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly body: string;

  constructor(status: number, statusText: string, body: string) {
    super(`${status}: ${body || statusText}`);
    this.name = "DashboardServiceError";
    this.status = status;
    this.statusText = statusText;
    this.body = body;
  }
}

function readEnvValue(key: string): string | undefined {
  const env = import.meta.env as Record<string, string | undefined>;
  const value = env[key]?.trim();
  return value || undefined;
}

export function resolveServiceBaseUrl(envKey: string, explicitBaseUrl?: string): string {
  const raw = explicitBaseUrl ?? readEnvValue(envKey) ?? HERMES_BASE_PATH;
  if (!raw) return "";
  return raw.replace(/\/+$/, "");
}

export function buildServiceUrl(baseUrl: string, path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${normalizedPath}`;
}

export async function serviceFetchJSON<T>(
  envKey: string,
  path: string,
  init: RequestInit = {},
  config: DashboardServiceClientConfig = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  const token = typeof window === "undefined" ? undefined : window.__HERMES_SESSION_TOKEN__;
  if (token && !headers.has("X-Hermes-Session-Token")) {
    headers.set("X-Hermes-Session-Token", token);
  }

  const fetchImpl = config.fetchImpl ?? fetch;
  const response = await fetchImpl(buildServiceUrl(resolveServiceBaseUrl(envKey, config.baseUrl), path), {
    ...init,
    headers,
    credentials: init.credentials ?? "include",
  });

  if (!response.ok) {
    const body = await response.text().catch(() => response.statusText);
    if (response.status === 401 || response.status === 403) {
      throw new AuthUnauthorizedError(response.status);
    }
    throw new DashboardServiceError(response.status, response.statusText, body);
  }

  return response.json() as Promise<T>;
}
