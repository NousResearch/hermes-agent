import { Button } from "@nous-research/ui/ui/components/button";
import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";
import {
  FIRST_REFRESH_DELAY_MS,
  dismissInAppOfferForProfile,
  getInAppOffer,
  type InAppOffer,
  IDLE_SETTLE_MS,
  offerIsEligible,
  pageIsIdle,
  readDismissedIds,
  readLastShownAt,
  readPageActivity,
  recordDismissal,
  recordShown,
  REFRESH_EVERY_MS,
  RESHOW_QUIET_MS,
  reportInAppOfferEvent,
} from "@/lib/in-app-offer";
import { cn, themedBody } from "@/lib/utils";

/**
 * The Nous Portal's in-app offer on the dashboard: one dismissible dialog, the twin of the dialog
 * Hermes Desktop and the Webapp show. The Portal decided who sees it; the backend memoises its
 * answer per profile (`/api/in-app-offer`). This component only picks the moment, by the rules in
 * `lib/in-app-offer.ts`: once per offer per browser, at most one a day, and only after the page has
 * sat idle for a few seconds with no other dialog open and nobody typing.
 *
 * Mounted once at the app shell, beside the other first-run banners.
 */
export function InAppOfferDialog() {
  const { t } = useI18n();
  const { profile } = useProfileScope();
  const [offer, setOffer] = useState<InAppOffer | null>(null);
  const [openFor, setOpenFor] = useState<string | null>(null);
  // Bumped when the daily cap may have expired, so a long-lived tab re-checks.
  const [capCheck, setCapCheck] = useState(0);
  const open = openFor !== null && openFor === offer?.id;
  const panelRef = useRef<HTMLDivElement>(null);

  // Read the memo now, then ask the backend to fetch the account after boot settles and on a slow
  // cadence. A failure (an older backend without the route, a network blip) is silent.
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = (refresh: boolean) =>
      getInAppOffer(refresh)
        .then((res) => {
          if (!live) return;
          const next = res.in_app_offer ?? null;
          setOffer(next);
          // A withdrawn offer forgets that it was open, so if it comes back it waits for an idle
          // moment again instead of reopening at once.
          if (next === null) setOpenFor(null);
        })
        .catch(() => undefined);
    const tick = () => {
      void load(true).finally(() => {
        if (live) timer = setTimeout(tick, REFRESH_EVERY_MS);
      });
    };
    void load(false);
    timer = setTimeout(tick, FIRST_REFRESH_DELAY_MS);
    return () => {
      live = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [profile]);

  // `open` holds the id it opened for, so an offer the Portal withdrew (campaign ended, account
  // changed) or replaced closes on the next render with no effect needed.
  const offerId = offer?.id ?? null;

  // Open on a settled idle moment. Every sign of activity restarts the clock.
  useEffect(() => {
    if (!offerId || open) return;
    if (!offerIsEligible(window.localStorage, offerId, Date.now())) {
      // Inside the daily cap (not dismissed): check again once it expires, so a tab left open all
      // day still shows the offer tomorrow.
      if (readDismissedIds(window.localStorage).includes(offerId)) return;
      const wait = Math.max(0, readLastShownAt(window.localStorage) + RESHOW_QUIET_MS - Date.now());
      const recheck = setTimeout(() => setCapCheck((n) => n + 1), Math.min(wait + 1_000, 2 ** 31 - 1));
      return () => clearTimeout(recheck);
    }
    let timer: ReturnType<typeof setTimeout> | null = null;
    const arm = () => {
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        if (!pageIsIdle(readPageActivity(document))) {
          arm();
          return;
        }
        // Another tab may have shown or dismissed one meanwhile.
        if (!offerIsEligible(window.localStorage, offerId, Date.now())) return;
        recordShown(window.localStorage, Date.now());
        reportInAppOfferEvent(window.localStorage, offerId, "shown");
        setOpenFor(offerId);
      }, IDLE_SETTLE_MS);
    };
    const events = ["keydown", "pointerdown", "focusin", "visibilitychange"] as const;
    events.forEach((name) => document.addEventListener(name, arm, true));
    window.addEventListener("focus", arm);
    arm();
    return () => {
      if (timer !== null) clearTimeout(timer);
      events.forEach((name) => document.removeEventListener(name, arm, true));
      window.removeEventListener("focus", arm);
    };
  }, [offerId, open, capCheck]);

  const close = () => {
    if (offerId) {
      recordDismissal(window.localStorage, offerId);
      // For the profile too, so Desktop and the Webapp never show it again either.
      dismissInAppOfferForProfile(offerId);
    }
    setOpenFor(null);
  };

  // Escape closes it. Focus moves INTO the panel (not onto a button), so a stray Space or Enter
  // typed as it opens cannot dismiss it unread.
  useEffect(() => {
    if (!open || !offerId) return;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        recordDismissal(window.localStorage, offerId);
        dismissInAppOfferForProfile(offerId);
        setOpenFor(null);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, offerId]);

  if (!open || !offer) return null;

  return createPortal(
    <div
      aria-describedby="in-app-offer-body"
      aria-labelledby="in-app-offer-title"
      aria-modal="true"
      className="fixed inset-0 z-[200] flex items-center justify-center bg-background/85 p-4"
      data-state="open"
      data-testid="in-app-offer-dialog"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
      role="dialog"
    >
      <div
        ref={panelRef}
        className={cn(
          themedBody,
          "relative w-full max-w-md border border-border bg-card shadow-2xl outline-none",
        )}
        tabIndex={-1}
      >
        <div className="flex items-start gap-3 border-b border-border p-4">
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <span className="text-[0.65rem] tracking-[0.2em] uppercase text-muted-foreground">
              {t.app.inAppOfferBanner ?? "From Nous"}
            </span>
            <h2
              className="font-mondwest text-display text-base tracking-wider"
              id="in-app-offer-title"
            >
              {offer.title}
            </h2>
            <p
              className="whitespace-pre-line break-words text-xs leading-relaxed text-muted-foreground"
              id="in-app-offer-body"
            >
              {offer.body}
            </p>
          </div>
          <button
            aria-label={t.app.dismiss ?? "Dismiss"}
            className="shrink-0 opacity-70 hover:opacity-100"
            onClick={close}
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex items-center justify-end gap-2 p-3">
          <Button onClick={close} outlined type="button">
            {t.app.inAppOfferDismiss ?? "Maybe later"}
          </Button>
          <Button
            onClick={() => {
              window.open(offer.cta_url, "_blank", "noopener,noreferrer");
              reportInAppOfferEvent(window.localStorage, offer.id, "clicked");
              close();
            }}
            type="button"
          >
            {offer.cta_label}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
