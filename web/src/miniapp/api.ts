// The Mini App's own fetch wrapper — deliberately separate from
// web/src/lib/api.ts's fetchJSON. That wrapper authenticates the desktop
// dashboard operator (legacy X-Hermes-Session-Token header, or a cookie
// session in gated mode) and applies profile-switcher query params neither
// of which apply here: the Mini App authenticates with a single bearer
// credential (Telegram's initData, verified server-side by
// plugins/dashboard_auth/telegram_miniapp/initdata.py) and always targets
// the machine's default profile — there is no profile switcher inside
// Telegram. Reuses HERMES_BASE_PATH from lib/api.ts (the one piece of
// cross-cutting infra that does apply: reverse-proxy path-prefix support).
import { HERMES_BASE_PATH } from "@/lib/api";
import { getInitData } from "./telegram";

export class MiniAppApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// A 401 from any call means the reused initData is no longer accepted
// (almost always: the app has been open past max_age_seconds -- Telegram
// gives no way to refresh initData mid-session, so the only fix is to
// reopen). Register a single handler here so the whole app can show one
// clear "reopen" state instead of each action surfacing its own misleading
// failure ("restart failed to start" etc.) for what is really an expired
// session. MiniApp.tsx registers this on mount.
let authExpiredHandler: (() => void) | null = null;
export function onAuthExpired(handler: () => void): void {
  authExpiredHandler = handler;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const initData = getInitData();
  if (initData) {
    headers.set("Authorization", `Bearer ${initData}`);
  }
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${HERMES_BASE_PATH}${path}`, { ...init, headers });
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* empty body, e.g. a bare 204 */
  }
  if (!res.ok) {
    if (res.status === 401 && authExpiredHandler) {
      authExpiredHandler();
    }
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new MiniAppApiError(res.status, detail);
  }
  return body as T;
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

export function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: body !== undefined ? JSON.stringify(body) : undefined });
}

export function put<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "PUT", body: body !== undefined ? JSON.stringify(body) : undefined });
}

export function patch<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "PATCH", body: body !== undefined ? JSON.stringify(body) : undefined });
}

export function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" });
}
