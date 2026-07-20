// Forwarded Cmd+C sequence already understood by hermes-ink as Ctrl+Super+C.
// Dashboard TUI mode reserves it for a dedicated latest-response copy action.
export const DASHBOARD_COPY_LAST_SEQUENCE = "\x1b[99;13u";

const DASHBOARD_NATIVE_SUBMIT_PREFIX = "\x1b_HERMES_SUBMIT;";
const APC_END = "\x1b\\";
const MAX_NATIVE_DRAFT_BYTES = 256 * 1024;
const VALID_REQUEST_ID = /^[A-Za-z0-9-]{1,64}$/;
export const DASHBOARD_NATIVE_SUBMIT_ACK_OSC = 777;
export const NATIVE_SUBMIT_ACK_TTL_MS = 5_000;
export const MAX_OSC52_CLIPBOARD_BYTES = 1024 * 1024;
export const OSC52_COPY_REQUEST_TTL_MS = 5_000;

function encodeUtf8Base64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  if (bytes.length > MAX_NATIVE_DRAFT_BYTES) {
    throw new Error("native draft is too large");
  }

  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

export function buildNativeDraftSubmissionPayload(draft: string, requestId: string): string {
  if (!VALID_REQUEST_ID.test(requestId)) {
    throw new Error("invalid native submission request id");
  }
  return `${DASHBOARD_NATIVE_SUBMIT_PREFIX}${requestId};${encodeUtf8Base64(draft)}${APC_END}`;
}

export function sendNativeDraftSubmission(
  draft: string,
  requestId: string,
  send: (payload: string) => void,
): boolean {
  try {
    send(buildNativeDraftSubmissionPayload(draft, requestId));
    return true;
  } catch {
    return false;
  }
}

export function consumeNativeSubmitAck(data: string, pendingRequestId: string | null): boolean {
  return pendingRequestId !== null && data === `HERMES_SUBMIT_ACK;${pendingRequestId}`;
}

export interface Osc52WriteResult {
  text: string | null;
  pendingAt: null;
}

const VALID_BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

export function consumePendingOsc52Write(
  data: string,
  pendingAt: number | null,
  now = Date.now(),
): Osc52WriteResult {
  if (pendingAt === null || now - pendingAt > OSC52_COPY_REQUEST_TTL_MS) {
    return { text: null, pendingAt: null };
  }

  const semi = data.indexOf(";");
  const target = semi < 0 ? "" : data.slice(0, semi);
  const payload = semi < 0 ? "" : data.slice(semi + 1);
  const maxEncodedLength = Math.ceil(MAX_OSC52_CLIPBOARD_BYTES / 3) * 4;
  if (
    target !== "c" ||
    !payload ||
    payload.length > maxEncodedLength ||
    !VALID_BASE64.test(payload)
  ) {
    return { text: null, pendingAt: null };
  }

  try {
    const binary = atob(payload);
    if (binary.length > MAX_OSC52_CLIPBOARD_BYTES) {
      return { text: null, pendingAt: null };
    }
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { text, pendingAt: null };
  } catch {
    return { text: null, pendingAt: null };
  }
}
