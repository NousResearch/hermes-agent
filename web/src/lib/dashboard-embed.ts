export type DashboardEmbedEvent =
  | "authenticated"
  | "ready"
  | "connecting"
  | "reconnecting"
  | "disconnected"
  | "ended";

export interface DashboardEmbedRequest {
  authBridge: boolean;
  embedId: string;
  parentOrigin: string;
  profile: string;
}

type MessageTarget = {
  postMessage(message: unknown, targetOrigin: string): void;
};

function normalizedOrigin(raw: string): string | null {
  try {
    const url = new URL(raw);
    if ((url.protocol !== "https:" && url.protocol !== "http:") || url.origin !== raw) {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

export function parseDashboardEmbedRequest(
  searchParams: URLSearchParams,
  allowedParentOrigins: readonly string[],
  configuredProfiles: Readonly<Record<string, string>>,
): DashboardEmbedRequest | null {
  const embedId = (searchParams.get("embed") ?? "").trim();
  const requestedOrigin = normalizedOrigin(
    (searchParams.get("parent_origin") ?? "").trim(),
  );
  const allowed = new Set(
    allowedParentOrigins
      .map((origin) => normalizedOrigin(origin))
      .filter((origin): origin is string => Boolean(origin)),
  );

  if (!embedId || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(embedId)) return null;
  if (!requestedOrigin || !allowed.has(requestedOrigin)) return null;
  if (!Object.prototype.hasOwnProperty.call(configuredProfiles, embedId)) return null;

  const rawProfile = (searchParams.get("profile") ?? "").trim().toLowerCase();
  const rawPinnedProfile = (configuredProfiles[embedId] ?? "").trim().toLowerCase();
  const profile = rawProfile === "default" ? "" : rawProfile;
  const pinnedProfile = rawPinnedProfile === "default" ? "" : rawPinnedProfile;
  if (profile !== pinnedProfile) return null;

  return {
    authBridge: searchParams.get("auth_bridge") === "1",
    embedId,
    parentOrigin: requestedOrigin,
    profile,
  };
}

export function postDashboardEmbedEvent(
  target: MessageTarget | null | undefined,
  parentOrigin: string,
  event: DashboardEmbedEvent,
  embedId: string,
): void {
  target?.postMessage(
    {
      type: "hermes.dashboard.embed",
      event,
      embedId,
    },
    parentOrigin,
  );
}

declare global {
  interface Window {
    __HERMES_DASHBOARD_EMBED_PARENT_ORIGINS__?: string[];
    __HERMES_DASHBOARD_EMBED_PROFILES__?: Record<string, string>;
  }
}
