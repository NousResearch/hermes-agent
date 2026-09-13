export function shouldUseStructuredChatOnPhone(input: {
  maxTouchPoints?: number;
  pointerCoarse?: boolean;
}): boolean {
  return Boolean(input.pointerCoarse);
}

export function shouldRedirectChatToStructured(
  input: { maxTouchPoints?: number; pointerCoarse?: boolean },
  search: string,
): boolean {
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  if (params.get("pty") === "1") return false;
  return shouldUseStructuredChatOnPhone(input);
}

export function structuredChatLocationFromChat(pathname: string, search: string): string {
  const path = pathname.replace(/\/$/, "") || "/";
  if (path !== "/chat") return `${pathname}${search}`;
  return `/chat/structured${search}`;
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
