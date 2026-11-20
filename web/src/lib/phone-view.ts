/**
 * Phone-adaptive view state.
 *
 * The dashboard already ships a narrow (phone) layout that is selected by the
 * viewport width. The header's "phone view" control forces that layout inside
 * a fixed phone-sized frame so a desktop browser can see exactly what a phone
 * gets. The frame is a same-origin `<iframe>` at 390px wide: a real viewport,
 * so every responsive rule and `matchMedia` in the app behaves as it does on a
 * phone — a CSS transform/scale could never do that.
 *
 * The preview runs the app with `?phone-preview=1`. In that mode the app hides
 * its own view controls (no recursive preview) and ChatPage does not spawn a
 * PTY: a second live terminal on the same channel would supersede the parent
 * tab's session.
 */

export const PHONE_VIEW_STORAGE_KEY = "hermes-phone-view";
export const PHONE_PREVIEW_QUERY_PARAM = "phone-preview";

/** True when this document is the in-dashboard phone preview (the iframe). */
export function isPhonePreviewMode(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return (
      new URLSearchParams(window.location.search).get(PHONE_PREVIEW_QUERY_PARAM) ===
      "1"
    );
  } catch {
    return false;
  }
}

/** The persisted phone-view choice, or false when unavailable/unset. */
export function readStoredPhoneView(): boolean {
  try {
    return localStorage.getItem(PHONE_VIEW_STORAGE_KEY) === "true";
  } catch {
    // Private browsing / blocked storage — the toggle still works for the tab.
    return false;
  }
}

export function writeStoredPhoneView(value: boolean): void {
  try {
    localStorage.setItem(PHONE_VIEW_STORAGE_KEY, String(value));
  } catch {
    /* ignore — the toggle still works for this tab */
  }
}

/**
 * URL for the preview iframe: the current document URL with the preview flag
 * added, so the preview shows the same route (including `?resume=`) as the tab
 * that opened it.
 */
export function buildPhonePreviewUrl(): string {
  if (typeof window === "undefined") return "";
  try {
    const url = new URL(window.location.href);
    url.searchParams.set(PHONE_PREVIEW_QUERY_PARAM, "1");
    return url.href;
  } catch {
    return window.location.href;
  }
}
