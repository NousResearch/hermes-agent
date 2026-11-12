/**
 * User-facing error shape for the dashboard REST layer.
 *
 * `fetchJSON` used to throw `new Error("<status>: <raw body>")`, and ~40 toast
 * call sites interpolated the Error object (`Error: ${e}`), so users saw
 * `Error: Error: 404: {"detail":"Server 'foo' not found"}`. This module turns
 * that into a plain sentence (`Server 'foo' not found`) while keeping the
 * status and raw body on the error object for diagnostics.
 */

/** The dashboard's own backend could not be reached at all (fetch rejected). */
export const API_UNREACHABLE_MESSAGE =
  "Hermes dashboard cannot reach the Hermes service. Is `hermes dashboard` still running?";

/** Status → plain sentence, used when the body carries no usable `detail`. */
const STATUS_COPY: Record<number, string> = {
  400: "The dashboard sent a request the server could not understand.",
  401: "Your dashboard session has expired. Reload the page to sign in again.",
  403: "You are not allowed to do that on this dashboard.",
  404: "The server could not find what the dashboard asked for.",
  409: "That change conflicts with the current state on the server.",
  413: "That upload is too large for the server to accept.",
  422: "Some of the entered values are not valid.",
  429: "Too many requests. Wait a moment and try again.",
  500: "The Hermes service hit an internal error.",
  502: "The dashboard proxy could not reach the Hermes service.",
  503: "The Hermes service is not ready yet. Try again in a moment.",
  504: "The Hermes service took too long to respond.",
};

export function humanizeStatus(status: number): string {
  return STATUS_COPY[status] ?? `The Hermes service answered with an unexpected error (${status}).`;
}

/** Pull a human sentence out of a FastAPI-style error body, or null. */
export function extractDetail(body: string): string | null {
  const text = body.trim();
  if (!text) return null;
  if (text.startsWith("{") || text.startsWith("[")) {
    try {
      const parsed: unknown = JSON.parse(text);
      if (parsed && typeof parsed === "object") {
        const record = parsed as Record<string, unknown>;
        for (const key of ["detail", "message", "error"]) {
          const value = record[key];
          if (typeof value === "string" && value.trim()) return value.trim();
          // FastAPI validation errors: detail is a list of {msg, loc}.
          if (Array.isArray(value)) {
            const msgs = value
              .map((item) => (item && typeof item === "object" ? (item as { msg?: unknown }).msg : null))
              .filter((m): m is string => typeof m === "string");
            if (msgs.length) return msgs.join("; ");
          } else if (value && typeof value === "object") {
            // Structured detail ({error: "<code>", message: "<sentence>"}): the code is for
            // programs, the sentence is what the operator needs to act on.
            const message = (value as { message?: unknown }).message;
            if (typeof message === "string" && message.trim()) return message.trim();
          }
        }
      }
      return null;
    } catch {
      return null;
    }
  }
  if (text.startsWith("<")) return null; // HTML error page from a proxy
  return text.length <= 300 ? text : null;
}

export class ApiError extends Error {
  /** HTTP status; 0 when the request never reached the server. */
  readonly status: number;
  /** Raw response body (or the transport error text) for a Copy-details action. */
  readonly body: string;
  /** Request URL (path only), for diagnostics. */
  readonly url: string;

  constructor(message: string, init: { status: number; body: string; url: string }) {
    super(message);
    this.name = "ApiError";
    this.status = init.status;
    this.body = init.body;
    this.url = init.url;
  }

  /** Multi-line technical detail for a Copy-details action or console. */
  get details(): string {
    const head = this.status ? `HTTP ${this.status} ${this.url}` : `network failure ${this.url}`;
    return this.body ? `${head}\n${this.body}` : head;
  }
}

export function apiErrorFromResponse(status: number, body: string, url: string): ApiError {
  const detail = extractDetail(body);
  return new ApiError(detail ?? humanizeStatus(status), { status, body, url });
}

export function apiErrorFromNetworkFailure(cause: unknown, url: string): ApiError {
  const body = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
  return new ApiError(API_UNREACHABLE_MESSAGE, { status: 0, body, url });
}

/**
 * The one way to turn a caught `unknown` into toast/inline text. Never yields
 * a `Error: Error:` double prefix because it reads `.message`, not `String(e)`.
 */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message || err.name;
  if (typeof err === "string") return err;
  return String(err);
}
