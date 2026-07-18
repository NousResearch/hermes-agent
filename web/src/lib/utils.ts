import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

import type { Locale } from "@/i18n/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Mondwest font only — use on layout shells; do not force normal-case here or `text-display` chrome (Segmented, badges) stops uppercasing. */
export const themedFont = "font-mondwest";

/** Mondwest body copy — sentence-case themed text (not uppercase chrome). */
export const themedBody = "font-mondwest normal-case";

/** Mondwest brand chrome — uppercase section headers and nav labels. */
export const themedChrome = "font-mondwest text-display";

/** Locale tag used only at the Intl compatibility boundary. */
export function intlLocale(locale: Locale): string {
  if (locale === "zh") return "zh-Hans";
  if (locale === "zh-hant") return "zh-Hant";
  return locale;
}

/** Relative time from a Unix epoch timestamp (seconds). */
export function timeAgo(ts: number, locale: Locale = "en"): string {
  const delta = Date.now() / 1000 - ts;
  const relative = new Intl.RelativeTimeFormat(intlLocale(locale), {
    numeric: "auto",
    style: "narrow",
  });
  if (delta < 60) return relative.format(0, "second");
  if (delta < 3600) return relative.format(-Math.floor(delta / 60), "minute");
  if (delta < 86400) return relative.format(-Math.floor(delta / 3600), "hour");
  return relative.format(-Math.floor(delta / 86400), "day");
}

/** Relative time from an ISO-8601 timestamp string. */
export function isoTimeAgo(
  iso: string,
  locale: Locale = "en",
  unknown = "unknown",
): string {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 0 || Number.isNaN(delta)) return unknown;
  return timeAgo(Date.now() / 1000 - delta, locale);
}

export function formatDateTime(
  value: Date | number | string,
  locale: Locale,
  options: Intl.DateTimeFormatOptions = {
    dateStyle: "medium",
    timeStyle: "short",
  },
): string {
  return new Intl.DateTimeFormat(intlLocale(locale), options).format(
    value instanceof Date ? value : new Date(value),
  );
}
