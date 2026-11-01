import type { Translations } from "@/i18n";
import {
  Clock,
  Database,
  Globe,
  Hash,
  MessageCircle,
  MessageSquare,
  Play,
  Terminal,
} from "lucide-react";

export const SOURCE_CONFIG: Record<string, { icon: typeof Terminal; color: string }> =
  {
    cli: { icon: Terminal, color: "text-primary" },
    tui: { icon: Terminal, color: "text-primary" },
    telegram: { icon: MessageCircle, color: "text-[oklch(0.65_0.15_250)]" },
    discord: { icon: Hash, color: "text-[oklch(0.65_0.15_280)]" },
    slack: { icon: MessageSquare, color: "text-[oklch(0.7_0.15_155)]" },
    whatsapp: { icon: Globe, color: "text-success" },
    whatsapp_cloud: { icon: Globe, color: "text-success" },
    signal: { icon: MessageCircle, color: "text-success" },
    matrix: { icon: MessageCircle, color: "text-[oklch(0.65_0.15_250)]" },
    email: { icon: MessageSquare, color: "text-[oklch(0.7_0.15_155)]" },
    sms: { icon: MessageCircle, color: "text-success" },
    cron: { icon: Clock, color: "text-warning" },
    tool: { icon: Play, color: "text-warning" },
    oneshot: { icon: Terminal, color: "text-warning" },
    api_server: { icon: Globe, color: "text-muted-foreground" },
    acp: { icon: Database, color: "text-muted-foreground" },
    hermes_flow: { icon: Play, color: "text-warning" },
    vulcan_delegate: { icon: Play, color: "text-warning" },
    webhook: { icon: Globe, color: "text-warning" },
  };

export const AUTOMATION_SESSION_SOURCES = [
  "cron",
  "tool",
  "oneshot",
  "api_server",
  "acp",
  "hermes_flow",
  "vulcan_delegate",
  "webhook",
];
const AUTOMATION_SESSION_SOURCE_SET = new Set(AUTOMATION_SESSION_SOURCES);
export const NO_MATCHING_SESSION_SOURCE = "__hermes_dashboard_no_matching_source__";

export type SessionFilterCategory = "chats" | "automation" | "all";
export type SourceSelectionsByCategory = Record<SessionFilterCategory, string[] | null>;

export function isAutomationSource(source: string): boolean {
  return AUTOMATION_SESSION_SOURCE_SET.has(source);
}

export function sourceBelongsToCategory(
  source: string,
  category: SessionFilterCategory,
): boolean {
  if (category === "all") return true;
  if (category === "automation") return isAutomationSource(source);
  return !isAutomationSource(source);
}

export function sourceLabel(source: string, translations: Translations["sessions"]): string {
  switch (source) {
    case "api_server":
      return translations.sourceApiServer;
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
      return translations.sourceCron;
    case "tool":
      return translations.sourceTool;
    case "hermes_flow":
      return translations.sourceHermesFlow;
    case "vulcan_delegate":
      return translations.sourceVulcanDelegate;
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
