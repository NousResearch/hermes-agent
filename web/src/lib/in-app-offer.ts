import { fetchJSON } from "@/lib/api";

/** The Portal's campaign answer, as `/api/in-app-offer` serves it. */
export interface InAppOffer {
  id: string;
  title: string;
  body: string;
  cta_label: string;
  cta_url: string;
  /** ISO instant the offer's promo stops being claimable; absent with no deadline. */
  claimable_until?: string;
}

/** The Nous Portal's in-app offer for the managed profile (null when none). `refresh` fetches the
 * account first; otherwise this is a local read. Profile-scoped by fetchJSON. The answer is
 * checked here, not trusted: a malformed offer, or a CTA that is not an https link, is no offer. */
export async function getInAppOffer(refresh = false): Promise<{ in_app_offer: InAppOffer | null }> {
  const res: unknown = await fetchJSON(refresh ? "/api/in-app-offer?refresh=1" : "/api/in-app-offer");
  const offer = typeof res === "object" && res !== null ? (res as { in_app_offer?: unknown }).in_app_offer : null;
  return { in_app_offer: validOffer(offer) };
}

const filled = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

/** The five wire fields as non-blank strings with an https CTA, or null. */
export function validOffer(value: unknown): InAppOffer | null {
  if (typeof value !== "object" || value === null) return null;
  const o = value as Partial<Record<keyof InAppOffer, unknown>>;
  if (!filled(o.id) || !filled(o.title) || !filled(o.body) || !filled(o.cta_label) || !filled(o.cta_url)) {
    return null;
  }
  try {
    const url = new URL(o.cta_url);
    if (url.protocol !== "https:" || !url.hostname) return null;
  } catch {
    return null;
  }
  const offer: InAppOffer = { id: o.id, title: o.title, body: o.body, cta_label: o.cta_label, cta_url: o.cta_url };
  if (typeof o.claimable_until === "string" && !Number.isNaN(new Date(o.claimable_until).getTime())) {
    offer.claimable_until = o.claimable_until;
  }
  return offer;
}

/** "Dec 15, 2026, 12:00 PM UTC": the Portal's own deadline wording, in UTC so every surface names
 * the same instant. */
export function formatClaimDeadline(iso: string): string {
  const date = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(new Date(iso));
  return `${date} UTC`;
}

/** Close the offer for the profile on every surface (Desktop and the Webapp read the same record).
 * Silent on failure: the browser's own dismissal still holds here. */
export function dismissInAppOfferForProfile(offerId: string): void {
  void fetchJSON("/api/in-app-offer/dismiss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ offer_id: offerId }),
  }).catch(() => undefined);
}

/** Offer events this browser already reported (`<id>:shown`, `<id>:clicked`). */
export const REPORTED_STORAGE_KEY = "hermes.dashboard.in-app-offer.reported-v1";

/** Reports in flight in this tab, so a re-render never sends one twice. */
const inFlight = new Set<string>();

/**
 * Tell the Portal the offer was shown or clicked (its campaign counts). The Portal counts each person
 * once per offer and event, so a repeat is harmless; this record only saves requests. It is written
 * once the Portal took the report, so a report lost to an expired token is sent again next time.
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
  if (reported.includes(key) || inFlight.has(key)) return;
  inFlight.add(key);
  void fetchJSON<{ reported?: boolean }>("/api/in-app-offer/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ offer_id: offerId, event }),
  })
    .then((res) => {
      if (res?.reported) {
        storage.setItem(REPORTED_STORAGE_KEY, JSON.stringify([...reported, key].slice(-100)));
      }
    })
    .catch(() => undefined)
    .finally(() => inFlight.delete(key));
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
