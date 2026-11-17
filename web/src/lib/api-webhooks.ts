// Dashboard webhook subscription CRUD (/api/webhooks): types plus the calls `api` spreads in.

export interface WebhookRoute {
  name: string;
  description: string;
  events: string[];
  deliver: string;
  deliver_only: boolean;
  prompt: string;
  skills: string[];
  created_at: string | null;
  url: string;
  secret_set: boolean;
  enabled: boolean;
  signature: WebhookSignature | null;
}

export type WebhookSignature = "generic" | "github" | "gitlab" | "linear";

export interface WebhooksResponse {
  enabled: boolean;
  base_url: string;
  subscriptions: WebhookRoute[];
}

export interface WebhookEnableResponse {
  ok: boolean;
  platform: "webhook";
  enabled: true;
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

export interface WebhookCreate {
  name: string;
  description?: string;
  events?: string[];
  prompt?: string;
  skills?: string[];
  deliver?: string;
  deliver_only?: boolean;
  deliver_chat_id?: string;
  signature?: WebhookSignature;
}

// POST /api/webhooks/{name}/test: the backend signs and sends one test event; `ok` is a 2xx answer.
export interface WebhookTestResponse {
  ok: boolean;
  url: string;
  delivery_id: string;
  status: number;
  response: string;
}

type FetchJSON = <T>(url: string, init?: RequestInit) => Promise<T>;

const routePath = (name: string, tail = "") =>
  `/api/webhooks/${encodeURIComponent(name)}${tail}`;

export function webhookApi(fetchJSON: FetchJSON) {
  return {
    getWebhooks: () => fetchJSON<WebhooksResponse>("/api/webhooks"),
    enableWebhooks: () =>
      fetchJSON<WebhookEnableResponse>("/api/webhooks/enable", { method: "POST" }),
    createWebhook: (body: WebhookCreate) =>
      fetchJSON<WebhookRoute & { secret: string }>("/api/webhooks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    deleteWebhook: (name: string) =>
      fetchJSON<{ ok: boolean }>(routePath(name), { method: "DELETE" }),
    setWebhookEnabled: (name: string, enabled: boolean) =>
      fetchJSON<{ ok: boolean; name: string; enabled: boolean }>(routePath(name, "/enabled"), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      }),
    // Signed server-side with the stored secret, which never reaches the browser.
    sendWebhookTest: (name: string) =>
      fetchJSON<WebhookTestResponse>(routePath(name, "/test"), { method: "POST" }),
  };
}
