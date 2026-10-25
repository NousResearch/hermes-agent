import { useEffect, useState } from "react";
import { Smartphone, X } from "lucide-react";
import { useI18n } from "@/i18n";

/**
 * Android white-band fix (PWA/standalone) — install hint.
 *
 * The dashboard is installable as a PWA (see manifest.webmanifest,
 * display: "standalone"), which makes Chrome render edge-to-edge and
 * removes the Android system navigation-bar white band below the terminal.
 * But a plain browser tab keeps the white band; the user must open the
 * dashboard as a standalone app for the fix to apply.
 *
 * This banner appears ONLY when the app is NOT already in standalone /
 * display-mode:standalone (i.e. normal browser browsing, where the white
 * band is visible) and tells the user how to install it (Add to home
 * screen / Install app). Once the app runs standalone, `display-mode`
 * matches and the banner disappears — no band, no nag.
 *
 * Dismissal is session-scoped (sessionStorage) so the hint doesn't nag on
 * every reload, but re-appears in a fresh session.
 */

const STORAGE_KEY = "pwaInstallBannerDismissed";

function readDismissed(): boolean {
  try {
    return sessionStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function writeDismissed(dismissed: boolean) {
  try {
    if (dismissed) sessionStorage.setItem(STORAGE_KEY, "1");
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function PwaInstallBanner() {
  const { t } = useI18n();
  const [standalone, setStandalone] = useState<boolean | null>(null);
  const [dismissed, setDismissed] = useState<boolean>(readDismissed);

  useEffect(() => {
    const mql = window.matchMedia("(display-mode: standalone)");
    const update = () => setStandalone(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, []);

  // Not yet decided (SSR/initial render) — hold off until matchMedia settles.
  if (standalone === null) return null;
  // Running standalone already: the white band is gone, no hint needed.
  if (standalone) return null;
  if (dismissed) return null;

  const dismiss = () => {
    writeDismissed(true);
    setDismissed(true);
  };

  const message =
    t.app.pwaInstallBanner ??
    "The white bar under the terminal is Android's navigation bar. Install this dashboard as an app (browser menu → Add to Home screen / Install) and open it as an app — the bar disappears.";

  return (
    <div
      role="alert"
      data-testid="pwa-install-banner"
      className="flex items-center gap-2 border-b border-sky-500/40 bg-sky-500/10 px-4 py-1.5 text-xs text-sky-300"
    >
      <Smartphone className="h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 flex-1">{message}</span>
      <button
        type="button"
        aria-label={t.app.dismiss ?? "Dismiss"}
        onClick={dismiss}
        className="shrink-0 opacity-70 hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
