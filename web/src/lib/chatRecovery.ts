const LAST_SESSION_KEY = "hermes.chat.lastSessionId";
const RECOVER_AFTER_AUTH_RELOAD_KEY = "hermes.chat.recoverAfterAuthReload";
const AUTH_RELOAD_AT_KEY = "hermes.chat.authReloadAt";

const SESSION_ID_RE = /^[A-Za-z0-9_.:-]{1,160}$/;
const AUTH_RELOAD_GUARD_MS = 5000;

function storage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function validSessionId(sessionId: string | null | undefined): sessionId is string {
  return !!sessionId && SESSION_ID_RE.test(sessionId);
}

export function rememberChatSessionId(sessionId: string | null | undefined): void {
  if (!validSessionId(sessionId)) return;
  storage()?.setItem(LAST_SESSION_KEY, sessionId);
}

export function getRememberedChatSessionId(): string | null {
  const value = storage()?.getItem(LAST_SESSION_KEY) ?? null;
  return validSessionId(value) ? value : null;
}

export function requestChatRecoveryAfterAuthReload(
  sessionId?: string | null,
): void {
  rememberChatSessionId(sessionId);
  storage()?.setItem(RECOVER_AFTER_AUTH_RELOAD_KEY, "1");
}

export function consumeChatRecoveryAfterAuthReload(): boolean {
  const store = storage();
  if (!store?.getItem(RECOVER_AFTER_AUTH_RELOAD_KEY)) return false;
  store.removeItem(RECOVER_AFTER_AUTH_RELOAD_KEY);
  return true;
}

export function scheduleDashboardAuthReload({
  sessionId,
  delayMs = 600,
}: {
  sessionId?: string | null;
  delayMs?: number;
} = {}): boolean {
  if (typeof window === "undefined") return false;

  requestChatRecoveryAfterAuthReload(sessionId ?? getRememberedChatSessionId());

  const store = storage();
  const now = Date.now();
  const lastReloadAt = Number(store?.getItem(AUTH_RELOAD_AT_KEY) ?? "0");
  if (lastReloadAt && now - lastReloadAt < AUTH_RELOAD_GUARD_MS) {
    return false;
  }

  store?.setItem(AUTH_RELOAD_AT_KEY, String(now));
  window.setTimeout(() => window.location.reload(), Math.max(0, delayMs));
  return true;
}
