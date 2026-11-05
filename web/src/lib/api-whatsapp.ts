export interface WhatsAppOnboardingStartRequest {
  mode?: "bot" | "self-chat";
  allowed_users?: string;
  reset_session?: boolean;
}

export interface WhatsAppOnboardingStartResponse {
  pairing_id: string;
  status:
    | "starting"
    | "installing"
    | "waiting"
    | "connected"
    | "error"
    | "expired"
    | "cancelled";
  qr_payload?: string | null;
  expires_at: string;
  mode: "bot" | "self-chat";
  allowed_users: string;
  account_id?: string | null;
  account_name?: string | null;
  account_phone?: string | null;
  error?: string | null;
}

export type WhatsAppOnboardingStatusResponse = WhatsAppOnboardingStartResponse;

export interface WhatsAppOnboardingApplyResponse {
  ok: boolean;
  platform: "whatsapp";
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

type Fetch = <T>(path: string, init?: RequestInit) => Promise<T>;

/** WhatsApp pairing flow, moved out of api.ts so that facade stays under its line cap. */
export function createWhatsAppApi(fetchJSON: Fetch) {
  return {
    startWhatsAppOnboarding: (body: WhatsAppOnboardingStartRequest) =>
      fetchJSON<WhatsAppOnboardingStartResponse>(
        "/api/messaging/whatsapp/onboarding/start",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      ),
    getWhatsAppOnboardingStatus: (pairingId: string) =>
      fetchJSON<WhatsAppOnboardingStatusResponse>(
        `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}`,
      ),
    applyWhatsAppOnboarding: (
      pairingId: string,
      body: { mode?: "bot" | "self-chat"; allowed_users?: string; profile?: string },
    ) =>
      fetchJSON<WhatsAppOnboardingApplyResponse>(
        `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}/apply`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      ),
    cancelWhatsAppOnboarding: (pairingId: string) =>
      fetchJSON<{ ok: boolean }>(
        `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}`,
        { method: "DELETE" },
      ),
  };
}
