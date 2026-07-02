import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Mondwest font only — use on layout shells; do not force normal-case here or `text-display` chrome (Segmented, badges) stops uppercasing. */
export const themedFont = "font-mondwest";

/** Mondwest body copy — sentence-case themed text (not uppercase chrome). */
export const themedBody = "font-mondwest normal-case";

/** Mondwest brand chrome — uppercase section headers and nav labels. */
export const themedChrome = "font-mondwest text-display";

/** Localizable strings for timeAgo/isoTimeAgo ({n} = unit count). */
export interface TimeAgoStrings {
  justNow: string;
  minutesAgo: string;
  hoursAgo: string;
  yesterday: string;
  daysAgo: string;
  unknown: string;
}

// Defaults match the historical hardcoded English so callers that don't pass
// translations (plugins SDK included) keep rendering identical output.
const TIME_AGO_EN: TimeAgoStrings = {
  justNow: "just now",
  minutesAgo: "{n}m ago",
  hoursAgo: "{n}h ago",
  yesterday: "yesterday",
  daysAgo: "{n}d ago",
  unknown: "unknown",
};

/** Relative time from a Unix epoch timestamp (seconds). */
export function timeAgo(ts: number, strings: TimeAgoStrings = TIME_AGO_EN): string {
  const delta = Date.now() / 1000 - ts;
  if (delta < 60) return strings.justNow;
  if (delta < 3600)
    return strings.minutesAgo.replace("{n}", String(Math.floor(delta / 60)));
  if (delta < 86400)
    return strings.hoursAgo.replace("{n}", String(Math.floor(delta / 3600)));
  if (delta < 172800) return strings.yesterday;
  return strings.daysAgo.replace("{n}", String(Math.floor(delta / 86400)));
}

/** Relative time from an ISO-8601 timestamp string. */
export function isoTimeAgo(iso: string, strings: TimeAgoStrings = TIME_AGO_EN): string {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 0 || Number.isNaN(delta)) return strings.unknown;
  if (delta < 60) return strings.justNow;
  if (delta < 3600)
    return strings.minutesAgo.replace("{n}", String(Math.floor(delta / 60)));
  if (delta < 86400)
    return strings.hoursAgo.replace("{n}", String(Math.floor(delta / 3600)));
  return strings.daysAgo.replace("{n}", String(Math.floor(delta / 86400)));
}
