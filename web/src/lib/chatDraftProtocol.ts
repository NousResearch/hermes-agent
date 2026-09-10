import { DRAFT_CONTROL_PREFIX, type DraftIdentity, type DraftResult, type DraftState } from "@hermes/shared";

export function sameDraft(a: DraftIdentity, b: DraftIdentity): boolean {
  return a.pty_instance === b.pty_instance && a.connection_generation === b.connection_generation &&
    a.session_id === b.session_id && a.draft_id === b.draft_id;
}

function isIdentity(value: unknown): value is DraftIdentity {
  if (!value || typeof value !== "object") return false;
  const id = value as DraftIdentity;
  return typeof id.pty_instance === "string" && !!id.pty_instance &&
    Number.isSafeInteger(id.connection_generation) && id.connection_generation >= 0 &&
    typeof id.session_id === "string" && !!id.session_id && typeof id.draft_id === "string" && !!id.draft_id;
}

/** undefined = terminal data; null = reserved but invalid/unknown control. */
export function parseDraftControl(data: string): DraftState | DraftResult | null | undefined {
  const prefixed = data.startsWith(DRAFT_CONTROL_PREFIX);
  let value: unknown;
  try { value = JSON.parse(prefixed ? data.slice(DRAFT_CONTROL_PREFIX.length) : data); }
  catch { return prefixed || /"type"\s*:\s*"draft\./.test(data) ? null : undefined; }
  if (!value || typeof value !== "object") return prefixed ? null : undefined;
  const message = value as Record<string, unknown>;
  if (typeof message.type !== "string" || !message.type.startsWith("draft.")) return prefixed ? null : undefined;
  if (!isIdentity(message.identity)) return null;
  if (message.type === "draft.state" && typeof message.available === "boolean") {
    return { type: "draft.state", identity: message.identity, available: message.available };
  }
  if (message.type === "draft.result" && typeof message.request_id === "string" && message.request_id &&
      ["attached", "stale", "unavailable", "failed"].includes(message.status as string)) {
    return { type: "draft.result", request_id: message.request_id, identity: message.identity, status: message.status as DraftResult["status"] };
  }
  return null;
}
