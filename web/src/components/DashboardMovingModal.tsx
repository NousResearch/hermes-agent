import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Bell, ChevronRight, Cloud, SlidersHorizontal } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import type { StatusResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

const STORAGE_KEY = "dashboardMovingModalDismissed";
/** The day the Hermes Cloud dashboard retires in favour of the web app. */
const RETIREMENT_DATE = new Date(2026, 11, 6);
/** Agent settings in the web app (a hash route, like every page there). */
const WEBAPP_SETTINGS_HASH = "/#/settings";

/** The machine row's status per web app state: its copy key, English fallback and colour. */
const MACHINE = {
  ready: {
    key: "dashboardMovingMachineReady",
    label: "Ready",
    tone: "text-success",
  },
  starting: {
    key: "dashboardMovingMachineStarting",
    label: "Starting",
    tone: "text-warning",
  },
  down: {
    key: "dashboardMovingMachineDown",
    label: "Offline",
    tone: "text-destructive",
  },
  not_available: {
    key: "dashboardMovingMachineNotAvailable",
    label: "Not available yet",
    tone: "opacity-65",
  },
} as const;

/**
 * Hermes Cloud's "your dashboard is moving" notice. A modal, not a banner:
 * it interrupts until the user answers, so there is no backdrop or Escape dismissal, only its
 * actions. The backend sets `webapp_state` only on a Hermes Cloud agent's signed-in dashboard, so
 * presence == show; a local dashboard never gets it. The machine row shows that state, and the
 * way into the web app is offered only while it is `ready`. "Later" hides the modal for this
 * browser session.
 */
export function DashboardMovingModal({
  status,
}: {
  status: StatusResponse | null;
}) {
  const { t, locale } = useI18n();
  const state = status?.webapp_state ?? null;
  const webappUrl = state === "ready" ? (status?.webapp_url ?? null) : null;
  const panelRef = useRef<HTMLDivElement>(null);
  const [dismissed, setDismissed] = useState(() => {
    try {
      return sessionStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const open = state !== null && !dismissed;

  useEffect(() => {
    if (!open) return;
    const prevActive = document.activeElement as HTMLElement | null;
    const actions = () =>
      Array.from(
        panelRef.current?.querySelectorAll<HTMLElement>("a[href], button") ??
          [],
      );
    // The way into the web app when there is one, else "Later".
    const initial = actions();
    (
      initial.find((el) => el.hasAttribute("data-primary")) ?? initial.at(-1)
    )?.focus();

    // Keep Tab inside the dialog: the page behind it is inert until answered.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const items = actions();
      const first = items[0];
      const last = items[items.length - 1];
      if (!first || !last) return;
      const active = document.activeElement;
      const outside = !panelRef.current?.contains(active);
      if (e.shiftKey ? active === first || outside : active === last || outside) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      prevActive?.focus?.();
    };
  }, [open, webappUrl]);

  if (!open || !state) return null;

  const later = () => {
    try {
      sessionStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* ignore */
    }
    setDismissed(true);
  };
  const date = new Intl.DateTimeFormat(locale, {
    month: "long",
    day: "numeric",
  }).format(RETIREMENT_DATE);
  const notice = (
    t.app.dashboardMovingNotice ??
    "This dashboard moves to the Hermes web app on {date}."
  ).replace("{date}", date);

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="dashboard-moving-title"
      aria-describedby="dashboard-moving-body"
      data-testid="dashboard-moving-modal"
      className="fixed inset-0 z-[150] flex items-center justify-center overflow-y-auto bg-background/85 p-4"
    >
      <div
        ref={panelRef}
        className="flex w-full max-w-[740px] flex-col gap-6 rounded-[6px] border border-border bg-card p-6 font-sans normal-case text-midground shadow-2xl sm:p-[50px]"
      >
        <p className="text-[10px] leading-[16px] tracking-[0.6px] uppercase opacity-40 sm:px-8">
          {t.app.dashboardMovingMeta ?? "Legacy dashboard / Retirement"}
        </p>

        <div className="flex items-center gap-2 rounded-[6px] bg-midground/5 px-4 py-2.5 text-[13px] leading-[17px]">
          <Bell aria-hidden className="h-4 w-4 shrink-0 opacity-80" />
          <span className="min-w-0 flex-1">{notice}</span>
        </div>

        <h2
          id="dashboard-moving-title"
          className="font-compressed text-[40px] leading-[48px] font-light tracking-[-1px] sm:px-8"
        >
          {t.app.dashboardMovingTitle ?? "Your dashboard is moving"}
        </h2>

        <p
          id="dashboard-moving-body"
          className="text-[16px] leading-[21px] opacity-80 sm:px-8"
        >
          {t.app.dashboardMovingBody ??
            "Your agents, chats, skills and routines come with you. Use the same sign-in."}
        </p>

        <MachineCard state={state} webappUrl={webappUrl} />

        <div className="flex flex-wrap gap-2.5 sm:px-8">
          {webappUrl && (
            <Button
              data-primary
              type="button"
              onClick={() => window.open(webappUrl, "_self")}
            >
              {t.app.dashboardMovingOpen ?? "Open in web app"}
            </Button>
          )}
          <Button type="button" outlined onClick={later}>
            {t.app.dashboardMovingLater ?? "Later"}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** The machine's web app state, and the way into its settings once it is up. */
function MachineCard({
  state,
  webappUrl,
}: {
  state: keyof typeof MACHINE;
  webappUrl: string | null;
}) {
  const { t } = useI18n();
  const machine = MACHINE[state];
  return (
    <div className="flex flex-col rounded-[15px] bg-midground/5 px-4 py-4 sm:px-8">
      <div className="flex min-h-16 items-center gap-4">
        <Cloud aria-hidden className="h-4 w-4 shrink-0 opacity-65" />
        <Row
          title={t.app.dashboardMovingMachineTitle ?? "Your cloud machine"}
          detail={
            state === "not_available"
              ? (t.app.dashboardMovingMachineNotAvailableDetail ??
                "The web app is not on this machine yet")
              : (t.app.dashboardMovingMachineDetail ??
                "Keep working in the new app")
          }
        />
        <span
          data-testid="dashboard-moving-state"
          className={`shrink-0 text-[13px] leading-[17px] ${machine.tone}`}
        >
          {t.app[machine.key] ?? machine.label}
        </span>
      </div>
      {webappUrl && (
        <>
          <div aria-hidden className="h-px bg-midground/5" />
          <a
            href={`${webappUrl}${WEBAPP_SETTINGS_HASH}`}
            className="flex min-h-16 items-center gap-4 outline-none focus-visible:ring-1 focus-visible:ring-midground"
          >
            <SlidersHorizontal
              aria-hidden
              className="h-4 w-4 shrink-0 opacity-65"
            />
            <Row
              title={t.app.dashboardMovingSettingsTitle ?? "Advanced settings"}
              detail={
                t.app.dashboardMovingSettingsDetail ??
                "Available in Agent settings"
              }
            />
            <ChevronRight aria-hidden className="h-4 w-4 shrink-0 opacity-80" />
          </a>
        </>
      )}
    </div>
  );
}

function Row({ title, detail }: { title: string; detail: string }) {
  return (
    <span className="flex min-w-0 flex-1 flex-col gap-0.5">
      <span className="text-[16px] leading-[21px]">{title}</span>
      <span className="text-[13px] leading-[17px] opacity-60">{detail}</span>
    </span>
  );
}
