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
  whatsapp_setup?: {
    mode?: string;
    allowed_users_set?: boolean;
    home_channel_set?: boolean;
  } | null;
  env_vars: MessagingPlatformEnvVar[];
}
