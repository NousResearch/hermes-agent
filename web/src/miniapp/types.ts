export type MiniAppTier = "admin" | "paired" | null;

export interface MiniAppMeResponse {
  tier: MiniAppTier;
  user_id: string | null;
}

/** The two admin-tier-only fields GET /api/status carries for the Mini
 * App's restart-needed banner (hermes_cli/web_server.py's get_status).
 * Independently nullable — see needsRestart() in MiniApp.tsx. */
export interface MiniAppStatusExtra {
  gateway_start_time: number | null;
  telegram_allowlist_updated_at: number | null;
}

export interface TelegramAllowlistEntry {
  user_id: string;
  username: string | null;
  name: string | null;
  added_at: number | null;
  source: "pairing" | "env";
}

export interface TelegramAllowlistResponse {
  allowlist: TelegramAllowlistEntry[];
}
