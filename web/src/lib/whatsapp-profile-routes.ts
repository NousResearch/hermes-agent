import type { ProfileInfo } from "@/lib/api";

/** One gateway.profile_routes row (WhatsApp routes may carry extra keys). */
export type WhatsAppProfileRouteRow = Record<string, unknown> & {
  name: string;
  platform: "whatsapp";
  profile: string;
  chat_id: string;
};

export function resolveHostProfileName(profiles: ProfileInfo[]): string {
  const host = profiles.find((p) => p.is_default);
  return host?.name ?? "default";
}

export function extractProfileRoutes(config: Record<string, unknown>): unknown[] {
  const gateway = config.gateway;
  if (gateway && typeof gateway === "object" && !Array.isArray(gateway)) {
    const routes = (gateway as Record<string, unknown>).profile_routes;
    if (Array.isArray(routes)) return routes;
  }
  const top = config.profile_routes;
  if (Array.isArray(top)) return top;
  return [];
}

export function isWhatsAppProfileRoute(route: unknown): route is Record<string, unknown> {
  if (!route || typeof route !== "object" || Array.isArray(route)) return false;
  const platform = String((route as Record<string, unknown>).platform ?? "")
    .trim()
    .toLowerCase();
  return platform === "whatsapp";
}

export function whatsappProfileRoutesFromList(routes: unknown[]): WhatsAppProfileRouteRow[] {
  return routes.filter(isWhatsAppProfileRoute).map((row) => ({
    ...row,
    name: String(row.name ?? ""),
    platform: "whatsapp",
    profile: String(row.profile ?? ""),
    chat_id: String(row.chat_id ?? ""),
  })) as WhatsAppProfileRouteRow[];
}

/** Replace whatsapp rows; preserve discord/telegram/etc. */
export function mergeWhatsAppProfileRoutes(
  existingRoutes: unknown[],
  whatsappRoutes: WhatsAppProfileRouteRow[],
): unknown[] {
  const other = existingRoutes.filter((r) => !isWhatsAppProfileRoute(r));
  return [...other, ...whatsappRoutes];
}

export function validateWhatsAppRouteName(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return "Name is required.";
  return null;
}

/**
 * Group JIDs must end with @g.us. DMs may use a phone number or user JID (@s.whatsapp.net / @lid).
 */
export function validateWhatsAppRouteChatId(chatId: string): string | null {
  const trimmed = chatId.trim();
  if (!trimmed) return "Chat id is required.";
  if (/\s/.test(trimmed)) return "Chat id must not contain whitespace.";

  const lower = trimmed.toLowerCase();
  if (lower.includes("@")) {
    if (lower.endsWith("@g.us")) {
      if (!/^[^@\s]+@g\.us$/i.test(trimmed)) {
        return "Group chat id must be an exact JID ending in @g.us.";
      }
      return null;
    }
    if (lower.endsWith("@s.whatsapp.net") || lower.endsWith("@lid")) {
      return null;
    }
    return "Use a group JID ending in @g.us, a phone number, or a user JID (@s.whatsapp.net / @lid).";
  }

  if (!/^\+?[0-9]{5,}$/.test(trimmed)) {
    return "Enter a phone number, group JID (…@g.us), or user JID.";
  }
  return null;
}

export function validateWhatsAppRouteProfile(
  profile: string,
  installedNames: Set<string>,
): string | null {
  const trimmed = profile.trim();
  if (!trimmed) return "Profile is required.";
  if (!installedNames.has(trimmed)) return "Choose an installed profile.";
  return null;
}

export function buildWhatsAppRouteRow(
  fields: { name: string; chat_id: string; profile: string },
  preserveFrom?: Record<string, unknown> | null,
): WhatsAppProfileRouteRow {
  const base = preserveFrom ? { ...preserveFrom } : {};
  base.name = fields.name.trim();
  base.platform = "whatsapp";
  base.chat_id = fields.chat_id.trim();
  base.profile = fields.profile.trim();
  return base as WhatsAppProfileRouteRow;
}

export interface ProfileRoutesPersistence {
  getConfig: (hostProfile: string) => Promise<Record<string, unknown>>;
  saveConfig: (partial: Record<string, unknown>, hostProfile: string) => Promise<unknown>;
}

/** Writes only gateway.profile_routes on the host profile config. Does not restart the gateway. */
export async function persistWhatsAppProfileRoutes(
  deps: ProfileRoutesPersistence,
  hostProfile: string,
  whatsappRoutes: WhatsAppProfileRouteRow[],
): Promise<void> {
  const config = await deps.getConfig(hostProfile);
  const existing = extractProfileRoutes(config);
  const merged = mergeWhatsAppProfileRoutes(existing, whatsappRoutes);
  await deps.saveConfig({ gateway: { profile_routes: merged } }, hostProfile);
}
