/**
 * Typed frontend wrappers around the gateway JSON-RPC methods the Chat UI
 * needs. Each helper forwards to {@link GatewayClient.request} using the
 * generated contract types in `@hermes/shared` — no new transport, no new
 * endpoint, no invented fields.
 *
 * The wrappers exist so the (eventual) ChatGPT-like renderer can call methods
 * like `session.resume` / `prompt.submit` by name with full type safety and
 * without re-deriving param shapes from the contract on every call site.
 *
 * If a future caller needs to attach images to the next prompt, use
 * `image.attach` (host path) or `image.attach_bytes` (base64 payload) BEFORE
 * calling `prompt.submit` — `PromptSubmitParams.text` is `unknown` to accept
 * structured parts lists on the relay / hosted paths but it does NOT carry an
 * inline `images` field. See `gateway-contract.generated.ts` line ~2297.
 */
import type {
  AttachedImageResult,
  PdfAttachResult,
  RpcMethods,
} from "@hermes/shared";

import type { GatewayClient } from "@/lib/gatewayClient";

/**
 * Per-method request payload, narrowed from the generated contract.
 * Lets the helper signatures stay explicit without re-exporting every
 * generated type.
 */
type MethodParams<M extends keyof RpcMethods> = RpcMethods[M]["params"];
type MethodResult<M extends keyof RpcMethods> = RpcMethods[M]["result"];

function params<M extends keyof RpcMethods>(
  _method: M,
  value: MethodParams<M>,
): Record<string, unknown> {
  // The base client takes `Record<string, unknown>`; cast once at the seam.
  return value as unknown as Record<string, unknown>;
}

/**
 * Create a brand-new Hermes session.
 *
 * Returns the runtime `session_id`, the stored key, and (optionally) the
 * seeded transcript. The new `/chat-ui` page uses this for its "+ New Chat"
 * action — the CLI still spawns its own PTY child, so both surfaces end up
 * targeting separate session rows under the same session-store contract.
 */
export function sessionCreate(
  gw: GatewayClient,
  args: MethodParams<"session.create">,
): Promise<MethodResult<"session.create">> {
  return gw.request<MethodResult<"session.create">>(
    "session.create",
    params("session.create", args),
  );
}

/** Attach to an existing session (or create+resume by stored id / title). */
export function sessionResume(
  gw: GatewayClient,
  args: MethodParams<"session.resume">,
): Promise<MethodResult<"session.resume">> {
  return gw.request<MethodResult<"session.resume">>(
    "session.resume",
    params("session.resume", args),
  );
}

/** Read a slice of stored transcript messages for the session. */
export function sessionHistory(
  gw: GatewayClient,
  args: MethodParams<"session.history">,
): Promise<MethodResult<"session.history">> {
  return gw.request<MethodResult<"session.history">>(
    "session.history",
    params("session.history", args),
  );
}

/** Cancel the currently-running turn. */
export function sessionInterrupt(
  gw: GatewayClient,
  args: MethodParams<"session.interrupt">,
): Promise<MethodResult<"session.interrupt">> {
  return gw.request<MethodResult<"session.interrupt">>(
    "session.interrupt",
    params("session.interrupt", args),
  );
}

/**
 * Submit the user's next prompt to the running session.
 *
 * NOTE: `PromptSubmitParams.text` is `unknown` — string on the local path,
 * structured parts list on relay / hosted. Image attachment does NOT live
 * here; call `image.attach` (host path) or `image.attach_bytes` (base64)
 * before submitting, then let the queued attachments ride the next turn.
 */
export function promptSubmit(
  gw: GatewayClient,
  args: MethodParams<"prompt.submit">,
): Promise<MethodResult<"prompt.submit">> {
  return gw.request<MethodResult<"prompt.submit">>(
    "prompt.submit",
    params("prompt.submit", args),
  );
}

/**
 * Drain replay events since `last_seen` seq (post-reconnect / polling).
 * The `events` entries are loose `Record<string, unknown>` per the contract;
 * callers narrow them against the GatewayEvent union in `@hermes/shared`.
 */
export function sessionEventsSince(
  gw: GatewayClient,
  args: MethodParams<"session.events.since">,
): Promise<MethodResult<"session.events.since">> {
  return gw.request<MethodResult<"session.events.since">>(
    "session.events.since",
    params("session.events.since", args),
  );
}

/**
 * Inject a steering correction into the active turn.
 *
 * The generated contract reuses `SessionCorrectionParams` for this method —
 * `text` is a required string, `session_id` and optional `profile` complete
 * the payload. The result carries the same `CorrectionStatus` (`queued` /
 * `redirected` / `rejected`) used by `session.redirect`.
 */
export function sessionSteer(
  gw: GatewayClient,
  args: MethodParams<"session.steer">,
): Promise<MethodResult<"session.steer">> {
  return gw.request<MethodResult<"session.steer">>(
    "session.steer",
    params("session.steer", args),
  );
}

/**
 * Queue an image (or a clipboard paste) on the active session.
 *
 * `image.attach_bytes` is the browser-side path: bytes ride the JSON-RPC
 * frame as base64. The gateway validates magic bytes, writes the image to
 * `HERMES_HOME/images`, and queues it as an attachment for the *next* turn.
 * Use this BEFORE `prompt.submit` — `PromptSubmitParams.text` does NOT
 * carry an inline `images` field.
 *
 * The companion `clipboard.paste` method exists for the server-side TUI;
 * the browser can never see a real server clipboard, so we always use
 * `image.attach_bytes` from the React client.
 */
export function imageAttachBytes(
  gw: GatewayClient,
  args: MethodParams<"image.attach_bytes">,
): Promise<AttachedImageResult> {
  return gw.request<AttachedImageResult>(
    "image.attach_bytes",
    params("image.attach_bytes", args),
  );
}

/**
 * Server-side clipboard paste — exists so the embedded TUI can mirror the
 * browser's clipboard on the user's behalf. The structured Chat UI never
 * needs this path because the browser already has the bytes; see
 * {@link imageAttachBytes}.
 */
export function clipboardPaste(
  gw: GatewayClient,
  args: MethodParams<"clipboard.paste">,
): Promise<AttachedImageResult> {
  return gw.request<AttachedImageResult>(
    "clipboard.paste",
    params("clipboard.paste", args),
  );
}

/**
 * Queue a PDF on the active session for the next `prompt.submit`.
 *
 * The verified contract (`pdf.attach` — see
 * `apps/shared/src/gateway-contract.generated.ts` line ~4605) accepts
 * EITHER a host `path` OR a base64 payload (`content_base64` / `data`),
 * plus an optional `first_page` / `last_page` window. The browser client
 * always uses the base64 path because Hermes runs on a different host;
 * the server validates the PDF magic bytes, renders the page window, and
 * queues the text under HERMES_HOME/pdfs.
 *
 * Mirrors the image-attach flow: call this BEFORE `prompt.submit` so the
 * queued PDF rides the next turn — `PromptSubmitParams` does NOT carry an
 * inline `pdfs` field.
 *
 * Result carries `attached: boolean` plus the rendered pages; the caller
 * surfaces an inline error when `attached === false`.
 */
export function pdfAttach(
  gw: GatewayClient,
  args: MethodParams<"pdf.attach">,
): Promise<PdfAttachResult> {
  return gw.request<PdfAttachResult>("pdf.attach", params("pdf.attach", args));
}