/**
 * Mobile clipboard paste for the dashboard chat tab (PRD v2 / TRD §3).
 *
 * Pure and dependency-injected: no React, no xterm import, no globals. The
 * caller wires every side effect (see `PasteDeps`), which is what keeps the
 * paste path testable without a browser and makes both PRD v1 regressions
 * structurally impossible:
 *
 *   - Text leaves through `pasteText`, which the caller binds to
 *     `term.paste(text)`. `term.write` is the *output* direction (PTY →
 *     screen): writing the pasted text there fakes a local echo, the TUI
 *     echoes it back down the socket, and the content renders two to three
 *     times.
 *   - Bytes leave through `sendBytes`, which the caller binds to
 *     `sendPtyShortcutSequence()` so `shouldBlockPtyInput` gates them.
 *
 * This module adds the one-tap affordance; it is not a second paste engine.
 * The DOM `paste` listener (long-press → Paste) and the Ctrl+V handler keep
 * working and converge on the same primitives.
 */

export type PasteFailure =
  | "insecure-context"
  | "permission-denied"
  | "unsupported-api"
  | "socket-closed"
  | "too-large";

export type PasteOutcome =
  | { kind: "sent-text"; chars: number }
  | { kind: "sent-image"; count: number }
  | { kind: "needs-confirmation"; preview: string; text: string }
  | { kind: "cancelled" }
  | { kind: "empty" }
  | { kind: "unsupported"; reason: PasteFailure }
  | { kind: "blocked"; reason: PasteFailure }; // socket gate refused

export interface PasteDeps {
  /** xterm input path. MUST be term.paste, NEVER term.write. */
  pasteText(text: string): void;
  /** Guarded PTY byte send; returns false when shouldBlockPtyInput refuses. */
  sendBytes(bytes: string): boolean;
  /** Image clipboard reader; absent when unsupported. */
  readImages?(): Promise<File[]>;
  readText?(): Promise<string>;
  isSecureContext: boolean;
  maxChars: number;
}

export interface PasteRequest {
  /**
   * The user's answer to a previous `needs-confirmation` outcome. Omit to
   * read the clipboard afresh.
   */
  confirmation?: "confirm" | "cancel";
  /** Text carried by a `needs-confirmation` outcome (FR-8). */
  pendingText?: string;
}

export const DEFAULT_PASTE_MAX_CHARS = 100_000;
export const PASTE_PREVIEW_MAX_CHARS = 280;

const NEWLINE_RE = /[\r\n]/;

function chars(text: string): number {
  return Array.from(text).length;
}

/** CRLF → LF, then truncate. This is the text the user confirms (FR-8). */
export function pastePreview(
  text: string,
  max = PASTE_PREVIEW_MAX_CHARS,
): string {
  const normalized = text.replace(/\r\n?/g, "\n");
  return normalized.length > max ? `${normalized.slice(0, max)}…` : normalized;
}

/**
 * Interpolate a UI template with untrusted content, LITERALLY.
 *
 * `String.prototype.replace` expands `$&`, `` $` ``, `$'`, `$$` and `$n` when
 * the replacement is a **string**. Every value passed here is untrusted —
 * clipboard text (B1) or a filesystem error message — so a string replacement
 * renders something the user never pasted: the FR-8 confirmation can be
 * defaced into a plausible-looking but different string while the full
 * original payload still reaches the agent terminal. A replacer *function*
 * disables the substitution patterns.
 */
function interpolate(
  template: string,
  token: "{preview}" | "{message}",
  value: string,
): string {
  return template.split(token).join(value);
}

/** Render the FR-8 multi-line confirmation prompt. */
export function formatPasteConfirmation(
  template: string,
  preview: string,
): string {
  return interpolate(template, "{preview}", preview);
}

/** Render the image-upload failure banner (message may carry `$` patterns). */
export function formatImageUploadError(
  template: string,
  message: string,
): string {
  return interpolate(template, "{message}", message);
}

/**
 * Pre-flight the PTY gate.
 *
 * The gate is only reachable through the guarded sender, so this is a
 * zero-length write: `sendBytes("")` applies the same `readyState` +
 * `shouldBlockPtyInput` check as a real send and delivers no bytes (the
 * server skips empty frames — `chat_ws.py`, "if not raw: continue").
 *
 * Without the pre-flight, pasting into a half-open socket is dropped
 * silently: `term.paste` → `onData` refuses and the text is lost, which is
 * exactly the failure FR-3 exists to prevent. Reporting `blocked` instead
 * lets the UI say why the paste did not arrive.
 */
function ptyInputAllowed(deps: PasteDeps): boolean {
  return deps.sendBytes("");
}

/** The single text sink — every text path (fresh or confirmed) ends here. */
function deliverText(deps: PasteDeps, text: string): PasteOutcome {
  if (!text.trim()) return { kind: "empty" };

  const length = chars(text);
  if (length > deps.maxChars) {
    return { kind: "unsupported", reason: "too-large" };
  }
  if (!ptyInputAllowed(deps)) {
    return { kind: "blocked", reason: "socket-closed" };
  }

  deps.pasteText(text);

  return { kind: "sent-text", chars: length };
}

async function readClipboardImages(deps: PasteDeps): Promise<File[]> {
  if (!deps.readImages) return [];
  try {
    return await deps.readImages();
  } catch {
    // A refused or empty image read must not swallow the paste: fall through
    // to the text path, which reports the same refusal with its own reason.
    return [];
  }
}

/**
 * Resolve one paste gesture into an outcome.
 *
 * Images win over text (matching the existing Ctrl+V precedence), because
 * `readText()` alone resolves to "" for an image-only clipboard and the user
 * would get nothing where they get an upload + `/image` attach today.
 * Multi-line text returns `needs-confirmation` first: a multi-line paste into
 * a TUI submits each line, which is both an accidental-execution hazard and a
 * prompt-injection surface (FR-8). Single-line pastes send immediately.
 */
export async function runPtyClipboardPaste(
  deps: PasteDeps,
  request: PasteRequest = {},
): Promise<PasteOutcome> {
  if (request.confirmation === "cancel") return { kind: "cancelled" };
  if (request.confirmation === "confirm") {
    return deliverText(deps, request.pendingText ?? "");
  }

  const files = await readClipboardImages(deps);
  if (files.length) {
    // The image route also ends in bytes on the socket (`/image <path>` then
    // `\r`), so it answers to the same gate as the text route. Without this
    // the module reported `sent-image` for a socket the caller would refuse
    // one line later — the NS-591 half-open case, where `readyState` is still
    // OPEN during a reconnect and the send would be swallowed silently.
    if (!ptyInputAllowed(deps)) {
      return { kind: "blocked", reason: "socket-closed" };
    }
    return { kind: "sent-image", count: files.length };
  }

  if (!deps.readText) {
    return {
      kind: "unsupported",
      reason: deps.isSecureContext ? "unsupported-api" : "insecure-context",
    };
  }

  let text: string;
  try {
    text = await deps.readText();
  } catch {
    // Never log or re-emit the clipboard contents — the failure class is the
    // only thing the UI needs (PRD: clipboard data must not be persisted).
    return { kind: "unsupported", reason: "permission-denied" };
  }

  // Checked before the newline test so a whitespace-only clipboard is a
  // no-op rather than a confirmation prompt.
  if (!text.trim()) return { kind: "empty" };
  if (NEWLINE_RE.test(text)) {
    return { kind: "needs-confirmation", preview: pastePreview(text), text };
  }

  return deliverText(deps, text);
}
