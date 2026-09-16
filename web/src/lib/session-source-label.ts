import type { Translations } from "@/i18n/types";

/**
 * Human-readable label for a session source.
 *
 * Lookup order:
 *  1. The active locale's `sessions.sources` map (e.g. Persian labels).
 *  2. The English default for known sources.
 *  3. Title-casing of the raw source id for unknown sources
 *     (`my_custom_source` -> "My Custom Source").
 */
export function sourceLabel(t: Translations | null, source: string): string {
  // Localized source names live in t.sessions.sources; fall back to the
  // English title-case default for unknown sources.
  const map = t?.sessions.sources as Record<string, string | undefined> | undefined;
  const localized = map?.[source];
  if (localized !== undefined) return localized;
  switch (source) {
    case "api_server":
      return "API server";
    case "acp":
      return "ACP";
    case "cli":
      return "CLI";
    case "tui":
      return "TUI";
    case "telegram":
      return "Telegram";
    case "discord":
      return "Discord";
    case "slack":
      return "Slack";
    case "whatsapp":
      return "WhatsApp";
    case "whatsapp_cloud":
      return "WhatsApp Cloud";
    case "sms":
      return "SMS";
    case "cron":
      return "Cron";
    case "tool":
      return "Tool";
    case "hermes_flow":
      return "Hermes Flow";
    case "vulcan_delegate":
      return "Vulcan delegate";
    case "webhook":
      return "Webhook";
    default:
      return source
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
  }
}
