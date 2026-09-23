/**
 * `useChatSession` — drives the new /chat-ui structured Chat UI.
 *
 * Owns ONE {@link GatewayClient} per active session (the brief's "primary
 * GatewayClient/session connection"), converts the existing gateway events
 * into the local {@link ChatUIAction} stream that {@link chatUIReducer}
 * expects, and exposes the imperative controls the composer needs
 * (submit prompt, interrupt, resume, new chat).
 *
 * Design notes:
 *
 *   - Pure state lives in the reducer (`@/lib/chat/reducer`). This hook
 *     only wires the GatewayClient and bridges it into the action stream,
 *     which keeps reducer tests deterministic and lets multiple surfaces
 *     reuse the same event mapping.
 *
 *   - The hook's connection lifetime is tied to the React mount via
 *     `useEffect`. Disconnect/unsubscribe are run on cleanup so a route
 *     change doesn't leave a dangling socket.
 *
 *   - The hook does NOT touch the PTY. The CLI Chat page keeps its PTY
 *     bridge untouched; this hook only talks JSON-RPC over WebSocket.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";

import type { GatewayClient as GatewayClientType } from "@/lib/gatewayClient";
import { GatewayClient } from "@/lib/gatewayClient";
import {
  imageAttachBytes,
  pdfAttach,
  promptSubmit,
  sessionCreate,
  sessionEventsSince,
  sessionInterrupt,
  sessionResume,
} from "@/lib/chat/gateway";
import {
  chatUIReducer,
  initialChatUIStore,
  resetChatUIIdCounters,
} from "@/lib/chat/reducer";
import type {
  ChatAttachment,
  ChatUIAction,
  ChatUIAssistantTurn,
  ChatUIStore,
  ChatUITurn,
} from "@/components/chat/types";
import type { AttachedImageResult, GatewayEvent } from "@hermes/shared";

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function pickPayload(event: GatewayEvent): unknown {
  if (isRecord(event)) {
    const payload = (event as { payload?: unknown }).payload;
    if (payload !== undefined) return payload;
    return event;
  }
  return event;
}

function dispatchFor(event: GatewayEvent): ChatUIAction | null {
  const payload = pickPayload(event);
  if (payload === undefined || payload === null) return null;
  switch (event.type) {
    case "message.start":
      return { type: "message/start" };
    case "message.delta": {
      const text = (payload as { text?: unknown }).text;
      const rendered = (payload as { rendered?: unknown }).rendered;
      return {
        type: "message/delta",
        payload: {
          text: typeof text === "string" ? text : "",
          rendered: typeof rendered === "string" ? rendered : null,
          verbose: null,
        },
      };
    }
    case "message.interim": {
      const text = (payload as { text?: unknown }).text;
      return {
        type: "message/interim",
        payload: { text: typeof text === "string" ? text : "" },
      };
    }
    case "message.complete":
      return { type: "message/complete", payload: payload as never };
    case "reasoning.delta": {
      const text = (payload as { text?: unknown }).text;
      return {
        type: "reasoning/delta",
        payload: {
          text: typeof text === "string" ? text : "",
          rendered: null,
          verbose: null,
        },
      };
    }
    case "reasoning.available": {
      const text = (payload as { text?: unknown }).text;
      return {
        type: "reasoning/available",
        payload: {
          text: typeof text === "string" ? text : "",
          rendered: null,
          verbose: null,
        },
      };
    }
    case "tool.start":
      return { type: "tool/start", payload: payload as never };
    case "tool.complete":
      return { type: "tool/complete", payload: payload as never };
    case "tool.generating": {
      const name = (payload as { name?: unknown }).name;
      return {
        type: "tool/generating",
        payload: { name: typeof name === "string" ? name : "tool" },
      };
    }
    case "tool.output_risk": {
      const toolId = (payload as { tool_id?: unknown }).tool_id;
      if (typeof toolId !== "string") return null;
      return {
        type: "tool/progress",
        payload: { tool_id: toolId, tail: "[output flagged as risky]" },
      };
    }
    case "session.title": {
      const title = (payload as { title?: unknown }).title;
      return {
        type: "session/title",
        title: typeof title === "string" ? title : "",
      };
    }
    case "session.info":
      return {
        type: "session/info",
        title:
          (payload as { title?: unknown }).title !== undefined
            ? ((payload as { title?: string | null }).title ?? null)
            : undefined,
        model:
          (payload as { model?: unknown }).model !== undefined
            ? ((payload as { model?: string | null }).model ?? null)
            : undefined,
        provider:
          (payload as { provider?: unknown }).provider !== undefined
            ? ((payload as { provider?: string | null }).provider ?? null)
            : undefined,
        reasoningEffort:
          (payload as { reasoning_effort?: unknown }).reasoning_effort !==
          undefined
            ? ((payload as { reasoning_effort?: string | null }).reasoning_effort ??
                null)
            : undefined,
      };
    default:
      return null;
  }
}

/* -------------------------------------------------------------------------- */
/* Hook                                                                        */
/* -------------------------------------------------------------------------- */

export interface UseChatSessionOptions {
  /** Resume target. When omitted, the hook boots a fresh session. */
  resumeSessionId?: string | null;
  /** Active profile scope (passed through to session.create/.resume). */
  profile?: string | null;
  /** Surface label sent on `prompt.submit` (matches the gateway contract). */
  surface?: string;
  /** Subscribe flag — when false the hook disconnects immediately. */
  enabled?: boolean;
}

export interface ChatSessionController {
  /** Current reducer state — pure renderable shape. */
  state: ChatUIStore;
  /**
   * Submit one prompt to the active session.
   *
   * Resolves to `true` on a successful dispatch (so the caller can clear
   * the input) and `false` on error (so the caller can keep the user's
   * text + attachments and let them retry).
   */
  submit: (
    text: string,
    attachments?: ReadonlyArray<ChatAttachment>,
  ) => Promise<boolean>;
  /** Interrupt the currently-running turn. */
  interrupt: () => Promise<void>;
  /** Resume a stored session by id (clears local history + rehydrates). */
  resume: (sessionId: string) => Promise<void>;
  /** Spin up a brand-new session. */
  newSession: () => Promise<void>;
  /**
   * Queue an image onto the active session via `image.attach_bytes`.
   * Returns a {@link ChatAttachment} projection for the composer chip, or
   * `null` when the gateway refused the upload (size, magic bytes, etc.).
   */
  attachImage: (file: File) => Promise<ChatAttachment | null>;
  /**
   * Queue a PDF onto the active session via `pdf.attach`. The browser
   * always uses the base64 path (`content_base64`) because Hermes runs on
   * a different host and the browser cannot hand it a local path.
   *
   * Returns a {@link ChatAttachment} projection (kind: `"pdf"`) for the
   * composer chip, or `null` when the gateway refused the upload (bad
   * magic bytes, page-cap exceeded, size limit, etc.).
   */
  attachPdf: (file: File) => Promise<ChatAttachment | null>;
}

/**
 * The chat hook. One connection, one socket, one reducer. Clears the
 * in-memory id caches on remount so test cases don't leak ids.
 */
export function useChatSession({
  resumeSessionId,
  profile,
  surface = "dashboard",
  enabled = true,
}: UseChatSessionOptions = {}): ChatSessionController {
  const [state, dispatch] = useReducer(chatUIReducer, initialChatUIStore);
  const gwRef = useRef<GatewayClientType | null>(null);
  const sessionRef = useRef<string | null>(null);
  // Mirror of the latest reducer state — read inside the imperative
  // callbacks without forcing them to re-bind on every state update.
  const stateRef = useRef<ChatUIStore>(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Reset id caches on mount so repeat renders don't leak entries.
  useEffect(() => {
    resetChatUIIdCounters();
  }, []);

  // Subscribe & connect.
  //
  // The hook honours `resumeSessionId` ONCE on the first connect; once a
  // session is running, mirroring the active id back to the URL via
  // `searchParams` updates `resumeSessionId`, but we don't want that to
  // spawn a reconnect loop. Track the initial target with a ref so only
  // the first call seeds the session; subsequent mounts/renders reuse
  // the same socket unless `resumeSessionId` itself changes (user picking
  // another conversation in the list).
  const initialResumeRef = useRef<string | null | undefined>(undefined);
  const targetResumeId =
    initialResumeRef.current === undefined
      ? (initialResumeRef.current = resumeSessionId)
      : initialResumeRef.current;

  useEffect(() => {
    if (!enabled) return undefined;

    const gw = new GatewayClient();
    gwRef.current = gw;

    const offState = gw.onState((s) => {
      dispatch({
        type: "connection/state",
        state:
          s === "open"
            ? "open"
            : s === "connecting"
            ? "connecting"
            : s === "error"
            ? "error"
            : s === "closed"
            ? "closed"
            : "idle",
      });
    });

    const offAny = gw.onAny((event) => {
      const action = dispatchFor(event);
      if (action) dispatch(action);
    });

    let cancelled = false;
    (async () => {
      try {
        dispatch({ type: "connection/state", state: "connecting" });
        await gw.connect();
        if (cancelled) return;
        // Bootstrap order: when a stored session id was supplied we
        // resume into it; otherwise we spin up a fresh session. The page
        // mirrors the active id back into the URL — see ChatUIPage for
        // the matching redirect — but we read the *initial* target only.
        if (targetResumeId) {
          await resumeInto(gw, targetResumeId, profile ?? null, dispatch);
          if (cancelled) return;
          sessionRef.current = targetResumeId;
        } else {
          const sid = await createInto(gw, profile ?? null, dispatch);
          if (cancelled) return;
          sessionRef.current = sid;
        }
      } catch (err) {
        if (cancelled) return;
        dispatch({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      }
    })();

    return () => {
      cancelled = true;
      offState();
      offAny();
      void gw.close();
      gwRef.current = null;
      sessionRef.current = null;
    };
  }, [targetResumeId, profile, enabled]);

  const submit = useCallback(
    async (
      text: string,
      attachments?: ReadonlyArray<ChatAttachment>,
    ): Promise<boolean> => {
      const trimmed = text.trim();
      if (!trimmed) return false;
      const gw = gwRef.current;
      const sessionId = sessionRef.current;
      if (!gw) {
        dispatch({
          type: "error",
          message: "Gateway not connected yet",
        });
        return false;
      }
      if (!sessionId) {
        dispatch({
          type: "error",
          message:
            "No active session; create one before sending a prompt",
        });
        return false;
      }
      dispatch({ type: "submit/started" });
      // Spread into a mutable array for the reducer (the union field is
      // declared mutable; the composer hands us a readonly tuple).
      dispatch({
        type: "user/submitted",
        text: trimmed,
        attachments: attachments ? [...attachments] : undefined,
      });
      try {
        await promptSubmit(gw, {
          session_id: sessionId,
          text: trimmed,
          surface,
          ...(profile ? { profile } : {}),
        });
        return true;
      } catch (err) {
        dispatch({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        });
        dispatch({ type: "submit/done", ok: false });
        return false;
      }
    },
    [surface, profile],
  );

  const interrupt = useCallback(async (): Promise<void> => {
    const gw = gwRef.current;
    const sessionId = sessionRef.current;
    if (!gw || !sessionId) return;
    try {
      await sessionInterrupt(gw, { session_id: sessionId });
    } catch (err) {
      dispatch({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  const resume = useCallback(
    async (sessionId: string): Promise<void> => {
      const gw = gwRef.current;
      if (!gw) {
        dispatch({
          type: "error",
          message: "Gateway not connected yet",
        });
        return;
      }
      if (gw.connectionState !== "open") {
        try {
          await gw.connect();
        } catch (err) {
          dispatch({
            type: "error",
            message:
              err instanceof Error ? err.message : String(err),
          });
          return;
        }
      }
      try {
        await resumeInto(gw, sessionId, profile ?? null, dispatch);
        sessionRef.current = sessionId;
      } catch (err) {
        dispatch({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [profile],
  );

  const newSession = useCallback(async (): Promise<void> => {
    const gw = gwRef.current;
    if (!gw) {
      dispatch({
        type: "error",
        message: "Gateway not connected yet",
      });
      return;
    }
    if (gw.connectionState !== "open") {
      try {
        await gw.connect();
      } catch (err) {
        dispatch({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        });
        return;
      }
    }
    try {
      const sid = await createInto(gw, profile ?? null, dispatch);
      sessionRef.current = sid;
    } catch (err) {
      dispatch({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [profile]);

  const attachImage = useCallback(
    async (file: File): Promise<ChatAttachment | null> => {
      const gw = gwRef.current;
      const sessionId = sessionRef.current;
      if (!gw) {
        dispatch({ type: "error", message: "Gateway not connected yet" });
        return null;
      }
      if (!sessionId) {
        dispatch({
          type: "error",
          message: "No active session; cannot attach image",
        });
        return null;
      }
      try {
        const base64 = await fileToBase64(file);
        const result: AttachedImageResult = await imageAttachBytes(gw, {
          session_id: sessionId,
          ...(profile ? { profile } : {}),
          content_base64: base64,
          filename: file.name || undefined,
          ext: guessExt(file),
        });
        if (!result.attached) {
          dispatch({
            type: "error",
            message:
              result.message ?? "Image attach failed (gateway refused).",
          });
          return null;
        }
        // Build a small data-URI thumbnail so the composer chip can show
        // an inline preview. The gateway never returns the bytes back, so
        // we keep a copy for the chip only (cleared on submit success).
        let dataUri: string | undefined;
        try {
          dataUri = await fileToDataUrl(file);
        } catch {
          dataUri = undefined;
        }
        const id =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? `att-${crypto.randomUUID()}`
            : `att-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        return {
          id,
          kind: "image",
          name: result.name ?? file.name ?? "image",
          mime: file.type || undefined,
          path: result.path ?? undefined,
          dataUri,
          width: result.width ?? undefined,
          height: result.height ?? undefined,
          tokenEstimate: result.token_estimate ?? undefined,
          bytes: result.bytes ?? file.size,
        };
      } catch (err) {
        // JSON-RPC error path — surface the gateway's message verbatim so
        // the user sees the actual failure (bad magic bytes, size limit,
        // unsupported type, etc.) instead of a generic "attach failed".
        // Re-throw so the composer's catch can render the same message
        // in the chip-side error banner; dispatch already lit the
        // shell-level error banner.
        const message =
          err instanceof Error ? err.message : "Image attach failed";
        dispatch({ type: "error", message });
        throw err instanceof Error ? err : new Error(message);
      }
    },
    [profile],
  );

  const attachPdf = useCallback(
    async (file: File): Promise<ChatAttachment | null> => {
      const gw = gwRef.current;
      const sessionId = sessionRef.current;
      if (!gw) {
        dispatch({ type: "error", message: "Gateway not connected yet" });
        return null;
      }
      if (!sessionId) {
        dispatch({
          type: "error",
          message: "No active session; cannot attach PDF",
        });
        return null;
      }
      try {
        const base64 = await fileToBase64(file);
        // pdf.attach returns a success envelope { attached, filename, ... }
        // when the gateway queues the pages. Failures arrive as a JSON-RPC
        // error frame (`_err(...)` on the server) which the shared transport
        // surfaces as a thrown `JsonRpcGatewayError`. We never see
        // `attached === false` from this gateway; the dead-code branch is
        // kept defensive so future gateways that DO return that shape still
        // surface a useful error instead of leaking the projection.
        const result = await pdfAttach(gw, {
          session_id: sessionId,
          ...(profile ? { profile } : {}),
          content_base64: base64,
          filename: file.name || undefined,
        });
        if (!result.attached) {
          // Defensive: contracts that return {attached: false} on failure.
          dispatch({
            type: "error",
            message:
              `PDF attach failed: ${result.filename || file.name || "document"}` +
              ` (${result.count} page(s) accepted).`,
          });
          return null;
        }
        const id =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? `att-${crypto.randomUUID()}`
            : `att-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        return {
          id,
          kind: "pdf",
          name: result.filename || file.name || "document.pdf",
          mime: file.type || "application/pdf",
          // Live-verified against the real gateway: ``count`` is the
          // running session-wide attachment total (image + PDF pages),
          // not the number of pages in *this* PDF. The page count for
          // the chip must come from ``pages_attached`` / ``pages.length``.
          // See tui_gateway/methods_prompt.py ``pdf_attach``.
          pageCount:
            (typeof result.pages_attached === "number" && result.pages_attached) ||
            (Array.isArray(result.pages) && result.pages.length) ||
            (typeof result.count === "number" ? result.count : 0),
          // The contract doesn't surface a single path back; pages have paths
          // but the chip is intentionally text-only. Leave path undefined
          // (the gateway already queued it for the next turn).
          bytes: file.size,
        };
      } catch (err) {
        // JSON-RPC error path — surface the gateway's message verbatim so
        // the user sees the actual failure ("PDF too large", "not a PDF",
        // "session not found", etc.) rather than a generic "attach failed".
        // Re-throw so the composer's catch surfaces the same message in
        // the chip-side error banner; the dispatch above already lit up
        // the shell-level error banner too.
        const message =
          err instanceof Error ? err.message : "PDF attach failed";
        dispatch({ type: "error", message });
        throw err instanceof Error ? err : new Error(message);
      }
    },
    [profile],
  );

  return useMemo(
    () => ({ state, submit, interrupt, resume, newSession, attachImage, attachPdf }),
    [state, submit, interrupt, resume, newSession, attachImage, attachPdf],
  );
}

/* -------------------------------------------------------------------------- */
/* Bootstrap helpers (shared between mount and imperatives)                    */
/* -------------------------------------------------------------------------- */

async function createInto(
  gw: GatewayClientType,
  profile: string | null,
  dispatch: (a: ChatUIAction) => void,
): Promise<string> {
  const result = await sessionCreate(gw, {
    profile: profile ?? null,
    source: "dashboard-chat-ui",
    close_on_disconnect: true,
  });
  const sessionId = result.session_id;
  dispatch({
    type: "session/created",
    sessionId,
    storedSessionId: result.stored_session_id,
  });
  if (Array.isArray(result.messages) && result.messages.length > 0) {
    dispatch({
      type: "session/resumed",
      sessionId,
      storedSessionId: result.stored_session_id,
      title: null,
      model: result.info?.model ?? null,
      provider: result.info?.provider ?? null,
      reasoningEffort: result.info?.reasoning_effort ?? null,
      history: result.messages,
      historySeq: 0,
    });
  }
  drainSince(gw, sessionId, 0, dispatch).catch(() => {});
  return sessionId;
}

async function resumeInto(
  gw: GatewayClientType,
  sessionId: string,
  profile: string | null,
  dispatch: (a: ChatUIAction) => void,
): Promise<void> {
  const result = await sessionResume(gw, {
    session_id: sessionId,
    profile: profile ?? null,
    lazy: false,
    omit_messages: false,
  });
  const messages = Array.isArray(result.messages) ? result.messages : [];
  dispatch({
    type: "session/resumed",
    sessionId: result.session_id,
    storedSessionId: result.stored_session_id ?? null,
    title: result.info?.title ?? null,
    model: result.info?.model ?? null,
    provider: result.info?.provider ?? null,
    reasoningEffort: result.info?.reasoning_effort ?? null,
    history: messages,
    historySeq: 0,
  });
  drainSince(gw, result.session_id, 0, dispatch).catch(() => {});
}

async function drainSince(
  gw: GatewayClientType,
  sessionId: string,
  lastSeen: number,
  dispatch: (a: ChatUIAction) => void,
): Promise<void> {
  try {
    const out = await sessionEventsSince(gw, {
      session_id: sessionId,
      last_seen: lastSeen,
    });
    if (Array.isArray(out.events)) {
      out.events.forEach((raw) => {
        const ev = raw as unknown as GatewayEvent;
        const action = dispatchFor(ev);
        if (action) dispatch(action);
      });
    }
  } catch {
    // Replay is best-effort; live events continue.
  }
}

/* -------------------------------------------------------------------------- */
/* File → base64 / data-URL helpers (browser FileReader)                       */
/* -------------------------------------------------------------------------- */

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("file read failed"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("file read did not return string"));
        return;
      }
      // Result is `data:<mime>;base64,<payload>` — strip the prefix.
      const idx = result.indexOf(",");
      resolve(idx >= 0 ? result.slice(idx + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("file read failed"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("file read did not return string"));
        return;
      }
      resolve(result);
    };
    reader.readAsDataURL(file);
  });
}

const EXT_BY_MIME: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
};

function guessExt(file: File): string | undefined {
  // Prefer the file name extension; fall back to the MIME map.
  const dot = file.name.lastIndexOf(".");
  if (dot > 0 && dot < file.name.length - 1) {
    return file.name.slice(dot + 1).toLowerCase();
  }
  if (file.type) return EXT_BY_MIME[file.type];
  return undefined;
}

/* -------------------------------------------------------------------------- */
/* Selectors re-exported for the page                                         */
/* -------------------------------------------------------------------------- */

export function isAssistantTurnPublic(
  turn: ChatUITurn,
): turn is ChatUIAssistantTurn {
  return turn.role === "assistant";
}
