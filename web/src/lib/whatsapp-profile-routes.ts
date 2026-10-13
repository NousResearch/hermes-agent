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

/** How many WhatsApp chats already target each profile, including unsaved table rows. */
export function whatsappRouteCountsByProfile(
  routes: { profile: string }[],
): Map<string, number> {
  const counts = new Map<string, number>();
  for (const row of routes) {
    const profile = row.profile.trim();
    if (!profile) continue;
    counts.set(profile, (counts.get(profile) ?? 0) + 1);
  }
  return counts;
}

export function formatProfileRouteOption(name: string, count: number): string {
  return `${name} (${count})`;
}

export interface WhatsAppSeenChatView {
  chat_id: string;
  name?: string | null;
  kind?: string | null;
  last_seen?: number | null;
}

export function whatsappChatKind(chatId: string, kind?: string | null): "group" | "dm" {
  if (kind === "group") return "group";
  if (chatId.toLowerCase().endsWith("@g.us")) return "group";
  return "dm";
}

export function whatsappChatLabel(chatId: string, name?: string | null): string {
  const trimmed = name?.trim() ?? "";
  if (trimmed && trimmed !== chatId) return trimmed;
  return chatId.toLowerCase().endsWith("@g.us") ? "Unnamed group" : "Unnamed contact";
}

export function formatWhatsAppLastSeen(unix: number | null | undefined): string {
  if (unix == null || !Number.isFinite(unix) || unix <= 0) return "";
  const date = new Date(unix * 1000);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function unassignedWhatsAppChats<T extends { chat_id: string }>(
  seen: T[],
  routes: { chat_id: string }[],
): T[] {
  const used = new Set(routes.map((row) => row.chat_id.trim().toLowerCase()));
  return seen.filter((chat) => {
    const chatId = chat.chat_id.trim().toLowerCase();
    return Boolean(chatId) && !used.has(chatId);
  });
}

export function whatsappRouteNameForChat(
  label: string,
  chatId: string,
  taken: Set<string>,
): string {
  const slug = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  const base = slug || "whatsapp-chat";
  if (!taken.has(base)) return base;
  const tail = chatId.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(-6) || "chat";
  let candidate = `${base}-${tail}`.slice(0, 48);
  let suffix = 2;
  while (taken.has(candidate)) {
    candidate = `${base}-${tail}-${suffix}`.slice(0, 48);
    suffix += 1;
  }
  return candidate;
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
