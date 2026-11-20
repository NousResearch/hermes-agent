import { fetchJSON } from "@/lib/api";

/** The Portal's campaign answer, as `/api/in-app-offer` serves it. */
export interface InAppOffer {
  id: string;
  title: string;
  body: string;
  cta_label: string;
  cta_url: string;
}

/** The Nous Portal's in-app offer for the managed profile (null when none). `refresh` fetches the
 * account first; otherwise this is a local read. Profile-scoped by fetchJSON. */
export function getInAppOffer(refresh = false): Promise<{ in_app_offer: InAppOffer | null }> {
  return fetchJSON(refresh ? "/api/in-app-offer?refresh=1" : "/api/in-app-offer");
}

/** Offer events this browser already reported (`<id>:shown`, `<id>:clicked`). */
export const REPORTED_STORAGE_KEY = "hermes.dashboard.in-app-offer.reported-v1";

/**
 * Tell the Portal the offer was shown or clicked, once per offer per event in this browser (its
 * campaign counts). Recorded before the call so another tab or a retry never double-counts; a
 * failure is silent and not retried.
 */
export function reportInAppOfferEvent(
  storage: OfferStorage,
  offerId: string,
  event: "shown" | "clicked",
): void {
  const key = `${offerId}:${event}`;
  let reported: string[] = [];
  try {
    const parsed: unknown = JSON.parse(storage.getItem(REPORTED_STORAGE_KEY) ?? "[]");
    reported = Array.isArray(parsed) ? parsed.filter((k): k is string => typeof k === "string") : [];
  } catch {
    reported = [];
  }
  if (reported.includes(key)) return;
  storage.setItem(REPORTED_STORAGE_KEY, JSON.stringify([...reported, key].slice(-100)));
  void fetchJSON("/api/in-app-offer/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ offer_id: offerId, event }),
  }).catch(() => undefined);
}

/**
 * The dashboard's in-app offer, presentation rules (no React). The same rules as Hermes
 * Desktop's dialog (apps/desktop/src/store/in-app-offer.ts), restated for a browser tab: once per
 * offer, a daily cap across campaigns, never while another dialog owns the screen or the user is
 * typing, and never in a background tab.
 */

/** Offers this browser closed. Dismissal is permanent per offer id. */
export const DISMISSED_STORAGE_KEY = "hermes.dashboard.in-app-offer.dismissed-v1";
/** When the last offer dialog was shown here (epoch ms). Caps frequency. */
export const LAST_SHOWN_STORAGE_KEY = "hermes.dashboard.in-app-offer.last-shown-v1";
/** At most one offer dialog per day, whatever the campaign cadence. */
export const RESHOW_QUIET_MS = 24 * 60 * 60 * 1000;
/** How long the page must sit idle before the dialog may open. */
export const IDLE_SETTLE_MS = 8_000;
/** The first account fetch waits for boot to settle. */
export const FIRST_REFRESH_DELAY_MS = 20_000;
/** Campaigns are slow-moving; one account read per half hour is plenty. */
export const REFRESH_EVERY_MS = 30 * 60 * 1000;
/** Keeps the stored list from growing forever. */
export const MAX_DISMISSED_IDS = 50;

export type OfferStorage = Pick<Storage, "getItem" | "setItem">;

export function readDismissedIds(storage: OfferStorage): string[] {
  try {
    const parsed: unknown = JSON.parse(storage.getItem(DISMISSED_STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

/** Record a dismissal, re-reading storage first so another tab's dismissals are kept. */
export function recordDismissal(storage: OfferStorage, id: string): string[] {
  const next = [...readDismissedIds(storage).filter((stored) => stored !== id), id].slice(
    -MAX_DISMISSED_IDS,
  );
  storage.setItem(DISMISSED_STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function readLastShownAt(storage: OfferStorage): number {
  const parsed = Number.parseInt(storage.getItem(LAST_SHOWN_STORAGE_KEY) ?? "", 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function recordShown(storage: OfferStorage, now: number): void {
  storage.setItem(LAST_SHOWN_STORAGE_KEY, String(now));
}

/** Not dismissed (in any tab) and past the daily cap. */
export function offerIsEligible(storage: OfferStorage, offerId: string, now: number): boolean {
  if (readDismissedIds(storage).includes(offerId)) return false;
  return now - readLastShownAt(storage) >= RESHOW_QUIET_MS;
}

export type PageActivity = {
  visibleAndFocused: boolean;
  otherDialogOpen: boolean;
  typing: boolean;
};

export function pageIsIdle(activity: PageActivity): boolean {
  return activity.visibleAndFocused && !activity.otherDialogOpen && !activity.typing;
}

/**
 * Read `PageActivity` from the live document. The dashboard's own dialogs (ConfirmDialog, the
 * model picker, drawers) carry `aria-modal="true"`; Radix menus and popovers carry
 * `data-state="open"`. Either one means the screen is taken.
 */
export function readPageActivity(doc: Document): PageActivity {
  const active = doc.activeElement;
  const typing =
    active instanceof HTMLElement &&
    (active.isContentEditable ||
      active.tagName === "INPUT" ||
      active.tagName === "TEXTAREA" ||
      active.tagName === "SELECT");
  return {
    visibleAndFocused: doc.visibilityState === "visible" && doc.hasFocus(),
    otherDialogOpen:
      doc.querySelector(
        '[aria-modal="true"], [role="dialog"][data-state="open"], [role="alertdialog"], [role="menu"][data-state="open"]',
      ) !== null,
    typing,
  };
}
