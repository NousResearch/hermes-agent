export function shouldUseStructuredChatOnPhone(_input: {
  maxTouchPoints?: number;
  pointerCoarse?: boolean;
}): boolean {
  return false;
}

export function shouldRedirectChatToStructured(
  _input: { maxTouchPoints?: number; pointerCoarse?: boolean },
  _search: string,
): boolean {
  return false;
}

export function shouldRedirectStructuredToChat(pathname: string): boolean {
  const path = pathname.replace(/\/$/, "") || "/";
  return path === "/chat/structured";
}

export function chatLocationFromStructured(pathname: string, search: string): string {
  if (!shouldRedirectStructuredToChat(pathname)) return `${pathname}${search}`;
  return `/chat${search}`;
}

export function structuredChatLocationFromChat(pathname: string, search: string): string {
  return `${pathname}${search}`;
}

export function ptyChatHref(sessionId: string, profile = ""): string {
  const params = new URLSearchParams();
  params.set("resume", sessionId);
  if (profile) params.set("profile", profile);
  params.set("pty", "1");
  return `/chat?${params.toString()}`;
}

export function readPhonePointer(): { maxTouchPoints: number; pointerCoarse: boolean } {
  if (typeof navigator === "undefined" || typeof window === "undefined") {
    return { maxTouchPoints: 0, pointerCoarse: false };
  }
  const coarse = typeof window.matchMedia === "function"
    && window.matchMedia("(pointer: coarse)").matches;
  return { maxTouchPoints: navigator.maxTouchPoints ?? 0, pointerCoarse: coarse };
}
