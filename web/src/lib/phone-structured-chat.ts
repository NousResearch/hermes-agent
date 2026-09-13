export function shouldUseStructuredChatOnPhone(input: {
  maxTouchPoints?: number;
  pointerCoarse?: boolean;
}): boolean {
  return Boolean(input.pointerCoarse) || (input.maxTouchPoints ?? 0) > 0;
}

export function structuredChatLocationFromChat(pathname: string, search: string): string {
  const path = pathname.replace(/\/$/, "") || "/";
  if (path !== "/chat") return `${pathname}${search}`;
  return `/chat/structured${search}`;
}

export function readPhonePointer(): { maxTouchPoints: number; pointerCoarse: boolean } {
  if (typeof navigator === "undefined" || typeof window === "undefined") {
    return { maxTouchPoints: 0, pointerCoarse: false };
  }
  const coarse = typeof window.matchMedia === "function"
    && window.matchMedia("(pointer: coarse)").matches;
  return { maxTouchPoints: navigator.maxTouchPoints ?? 0, pointerCoarse: coarse };
}
