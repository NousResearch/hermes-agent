import type { Translations } from "@/i18n/types";

/**
 * T4.7 — the notice HermesConsoleModal paints when a `/api/console` output
 * frame arrives with `truncated: true`. The copy comes from the locale
 * (`app.consoleOutputTruncated`, `{count}` interpolated), never a hardcoded
 * ASCII marker, so users learn in their own language how much was omitted
 * and that the full buffer is one click away.
 */
export function consoleTruncationNotice(t: Translations, omittedBytes: number): string {
  return t.app.consoleOutputTruncated.replace("{count}", String(omittedBytes));
}
