import { fetchJSON } from "@/lib/api";

import { isSupportedLocale } from "./resolve-locale";
import type { Locale } from "./types";

/**
 * Server-side persistence for the dashboard UI language.
 *
 * The preference lives in the profile's config.yaml as `dashboard.locale`
 * (`GET`/`PUT /api/dashboard/locale`, see `hermes_cli/web_routers/dashboard_ui.py`),
 * so a user's language follows them to any browser. localStorage remains the
 * offline / first-paint mirror; neither side is authoritative on its own.
 *
 * Both helpers fail soft: a slow or unreachable backend must never block or
 * blank the UI, and a failed write must never lose the choice (localStorage
 * already holds it).
 */

const LOCALE_ENDPOINT = "/api/dashboard/locale";

interface DashboardLocaleResponse {
  /** Saved locale id, or `null` when unset/unsupported. */
  locale: string | null;
}

/**
 * True only when this SPA was served by the Hermes backend. The Python server
 * always injects `__HERMES_AUTH_REQUIRED__` (`true` gated, `false` loopback) and,
 * in loopback mode, `__HERMES_SESSION_TOKEN__` (see
 * `hermes_cli/web_server_dashboard.py`). Guarding on that marker keeps the
 * resolution chain purely local in unit tests and any non-server context, where
 * there is no backend to ask and a stray request would only add noise.
 */
function serverPrefsAvailable(): boolean {
  if (typeof window === "undefined") return false;
  return (
    typeof window.__HERMES_SESSION_TOKEN__ === "string" ||
    typeof window.__HERMES_AUTH_REQUIRED__ === "boolean"
  );
}

/** Read the server-saved locale; `null` when unset, unsupported, or unreachable. */
export async function fetchServerLocale(): Promise<Locale | null> {
  if (!serverPrefsAvailable()) return null;
  try {
    const res = await fetchJSON<DashboardLocaleResponse>(LOCALE_ENDPOINT);
    return isSupportedLocale(res?.locale) ? res.locale : null;
  } catch {
    // Backend slow/down — fall back to the locally resolved locale.
    return null;
  }
}

/** Persist the chosen locale for this profile; best-effort. */
export async function persistServerLocale(locale: Locale): Promise<void> {
  if (!serverPrefsAvailable()) return;
  try {
    await fetchJSON(LOCALE_ENDPOINT, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locale }),
    });
  } catch {
    // Best-effort — the localStorage write already recorded the choice.
  }
}
