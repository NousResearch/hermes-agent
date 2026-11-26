export interface MessagingPlatformEnvVar {
  key: string;
  required: boolean;
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  prompt: string;
  help: string;
  url: string | null;
  is_password: boolean;
  /** Comma-separated allowlist rendered one entry per ID (absent on older backends). */
  is_list?: boolean;
  /** Plain saved value, sent only for allowlists (they are IDs, not secrets). */
  value?: string | null;
  advanced: boolean;
}

export interface MessagingPlatform {
  id: string;
  name: string;
  description: string;
  docs_url: string;
  enabled: boolean;
  configured: boolean;
  gateway_running: boolean;
  /**
   * "connected" | "disabled" | "not_configured" | "pending_restart" |
   * "gateway_stopped" | "startup_failed" | "disconnected" | "fatal" | string
   */
  state: string;
  error_code: string | null;
  error_message: string | null;
  updated_at: string | null;
  home_channel: { platform: string; chat_id: string; name: string; thread_id?: string } | null;
  /** Multiplex secondary served on the default profile's shared listener: the vendor callback URL. */
  ingress_url?: string | null;
  /** Switched off by the GATEWAY_RELAY_URL relay-exclusive rule, not by the user (absent on older backends). */
  disabled_by_relay?: boolean;
  /** Team Gateway's shared bot carries this platform: setup links an account to it. */
  relay_fronted?: boolean;
  /** The relay owns this platform outright, so a direct bot would be switched off at every start. */
  relay_exclusive?: boolean;
  whatsapp_setup?: {
    mode?: string;
    allowed_users_set?: boolean;
    home_channel_set?: boolean;
  } | null;
  env_vars: MessagingPlatformEnvVar[];
}

export interface MessagingPlatformTestResult {
  ok: boolean;
  state: string;
  message: string;
}

export interface MessagingPlatformUpdate {
  enabled?: boolean;
  env?: Record<string, string>;
  clear_env?: string[];
}

/** A curated-catalog messaging platform the profile does not have yet (hermes_cli/catalog_platform_listing.py). */
export interface CatalogMessagingPlatform {
  name: string;
  title: string;
  badge: "official" | "verified";
  summary: string;
  /** The catalog's "Disclosure" sentence, "" when it has none. */
  disclosure: string;
  /** "" when a pick can install it, else a reason code ("lazy_installs_off"). */
  blocked: string;
  /** English sentence for ``blocked``, "" when not blocked. */
  blocked_reason: string;
  /** The one command that installs it into this profile. */
  install_command: string;
}

export interface MessagingPlatformsResponse {
  /** Digest of what this list shows; `GET /api/messaging/platforms/signature` moves with it. Absent on older backends. */
  signature?: string;
  env_path: string;
  gateway_start_command: string;
  platforms: MessagingPlatform[];
  /** Absent on older backends. */
  available?: CatalogMessagingPlatform[];
}

export type MessagingCatalogInstallResult =
  | { ok: true; plugin_name: string; platforms: string[]; warnings: string[] }
  | { ok: false; error: string; install_command: string };

export interface RelayLinkedAccount {
  platformUserId: string;
  handle?: string;
  displayName?: string;
  containerName?: string;
}

export interface RelayLinksResponse {
  /** "portal" when the Nous Portal access setting governs this agent: links are made there. */
  managed_by?: "portal" | "connector";
  portal_url?: string;
  platforms: { platform: string; bot_url: string | null; accounts: RelayLinkedAccount[] }[];
}

export interface RelayLinkCode {
  platform: string;
  code: string;
  /** The exact message to send the shared bot. */
  command: string;
  expires_at: string | null;
  bot_url: string | null;
}

export interface RelayLinkCodeState {
  /** pending = not redeemed yet; inactive = used or expired; refused = see reason. */
  state: "pending" | "inactive" | "refused";
  reason?: string | null;
}
