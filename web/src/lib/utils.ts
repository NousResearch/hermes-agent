import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { intlTag } from "@hermes/shared/i18n";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Mondwest font only — use on layout shells; do not force normal-case here or `text-display` chrome (Segmented, badges) stops uppercasing. */
export const themedFont = "font-mondwest";

/** Mondwest body copy — sentence-case themed text (not uppercase chrome). */
export const themedBody = "font-mondwest normal-case";

/** Mondwest brand chrome — uppercase section headers and nav labels. */
export const themedChrome = "font-mondwest text-display";

/**
 * Relative time from a Unix epoch timestamp (seconds), rendered in the UI
 * locale via Intl.RelativeTimeFormat — native "yesterday"/«دیروز» wording and
 * pluralization instead of hardcoded English units. Signatures are the plugin
 * SDK's `utils.timeAgo`/`utils.isoTimeAgo`; keep them stable.
 */
export function timeAgo(ts: number): string {
  const delta = Date.now() / 1000 - ts;
  if (delta < 60) return relativeSeconds();
  if (delta < 3600) return relativeUnit(-Math.floor(delta / 60), "minute");
  if (delta < 86400) return relativeUnit(-Math.floor(delta / 3600), "hour");
  if (delta < 172800) return relativeDay(-1);
  return relativeDay(-Math.floor(delta / 86400));
}

/** Relative time from an ISO-8601 timestamp string. */
export function isoTimeAgo(iso: string): string {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 0 || Number.isNaN(delta)) return "unknown";
  if (delta < 60) return relativeSeconds();
  if (delta < 3600) return relativeUnit(-Math.floor(delta / 60), "minute");
  if (delta < 86400) return relativeUnit(-Math.floor(delta / 3600), "hour");
  return relativeDay(-Math.floor(delta / 86400));
}

const rtfCache = new Map<string, Intl.RelativeTimeFormat>();

function rtf(): Intl.RelativeTimeFormat {
  const tag = intlTag();
  let fmt = rtfCache.get(tag);
  if (!fmt) {
    fmt = new Intl.RelativeTimeFormat(tag, { numeric: "auto" });
    rtfCache.set(tag, fmt);
  }
  return fmt;
}

function relativeSeconds(): string {
  // numeric:"auto" renders 0 seconds as "now" in every locale.
  return rtf().format(0, "second");
}

function relativeUnit(value: number, unit: "second" | "minute" | "hour"): string {
  return rtf().format(value, unit);
}

function relativeDay(value: number): string {
  return rtf().format(value, "day");
}

/**
 * Absolute date/time in the UI locale. Persian (fa-*) additionally renders the
 * Jalali calendar with Persian digits — matching FilesPage's date formatting —
 * so a Persian UI never shows Gregorian/Latin dates. Returns "—" for missing
 * or unparseable input.
 */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const tag = intlTag();
  return d.toLocaleString(tag, {
    dateStyle: "medium",
    timeStyle: "short",
    ...(tag.startsWith("fa")
      ? { calendar: "persian", numberingSystem: "arabext" as const }
      : {}),
  });
}
