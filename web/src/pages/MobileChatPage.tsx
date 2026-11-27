/**
 * MobileChatPage — touch-first chat client for Hermes (MVP).
 *
 * Replaces the raw xterm PTY that the dashboard /chat tab renders on a phone.
 * This page is a dedicated mobile surface (route: /m): a full-screen overlay
 * that talks to the SAME tui_gateway JSON-RPC WebSocket the desktop app uses,
 * but presents native chat UX instead of a terminal.
 *
 *   session.create ──► { session_id }
 *   prompt.submit { session_id, text }            (after input.detect_drop)
 *   events: message.start / message.delta / reasoning.delta / tool.* /
 *           message.complete / session.info / error
 *
 * Why detect_drop first: the TUI runs `input.detect_drop` on the composer text
 * before `prompt.submit` so a bare file/image path gets rewritten into the
 * attach token the agent understands. We mirror that so an uploaded photo is
 * actually seen by the model (its server path is a real file the agent can
 * read/vision).
 */
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  Camera,
  Check,
  ChevronLeft,
  Copy,
  Eraser,
  History,
  Loader2,
  MessageSquarePlus,
  RefreshCw,
  Send,
  Settings,
  Sparkles,
  X,
} from "lucide-react";

import { Markdown } from "@/components/Markdown";
import {
  type ConnectionState,
  GatewayClient,
} from "@/lib/gatewayClient";
import { uploadChatImage } from "@/lib/chatImagePaste";
import { copyTextToClipboard } from "@/lib/clipboard";
import { api, type SessionInfo } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type MsgRole = "user" | "assistant" | "system" | "error";

interface ChatMsg {
  id: string;
  role: MsgRole;
  text: string;
  reasoning?: string;
  tools?: string[];
  inProgress: boolean;
  status?: "streaming" | "complete" | "error";
}

interface PendingImage {
  dataUrl: string;
  path: string;
  name: string;
  bytes: number;
}

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: "offline",
  connecting: "connecting…",
  open: "live",
  closed: "closed",
  error: "error",
};

function uid(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

function coerceText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((v) =>
        typeof v === "string"
          ? v
          : v && typeof v === "object"
            ? (v as Record<string, unknown>).text ?? ""
            : "",
      )
      .join("");
  }
  if (value && typeof value === "object") {
    const row = value as Record<string, unknown>;
    return typeof row.text === "string"
      ? row.text
      : typeof row.output_text === "string"
        ? row.output_text
        : "";
  }
  return "";
}

/** An assistant bubble that is still empty (no text, no reasoning, no tools)
 *  is invisible — the standalone "thinking…" row handles that window instead.
 *  An *in-progress* assistant message with no text yet is shown as a live
 *  "…" caret so the bubble never collapses into a blank region mid-stream. */
function assistantEmpty(m: ChatMsg): boolean {
  return m.role === "assistant" && !m.text && !m.reasoning && !m.tools?.length;
}

/** Map a persisted session's messages (from session.resume or the REST
 *  /api/sessions/<id>/messages endpoint) into renderable ChatMsg rows.
 *  Only user/assistant/system text surfaces as bubbles; tool rows and
 *  empty-content tool-call stubs are dropped for a clean mobile read. */
function mapHistoryMessages(items: unknown[] | undefined): ChatMsg[] {
  if (!Array.isArray(items)) return [];
  const out: ChatMsg[] = [];
  for (const raw of items) {
    const m = (raw ?? {}) as Record<string, unknown>;
    const roleRaw = typeof m.role === "string" ? m.role : "assistant";

    // Drop raw tool rows (the ⚙ flood) and empty tool-call stubs.
    if (roleRaw === "tool") continue;

    const role: MsgRole =
      roleRaw === "user" ||
      roleRaw === "assistant" ||
      roleRaw === "system" ||
      roleRaw === "error"
        ? roleRaw
        : "assistant";

    let text = coerceText(m.content);
    if (!text && role === "assistant") text = coerceText(m.rendered);
    if (!text) continue; // skip tool-call assistant stubs with empty content
    // Drop system-injected context-compaction handoffs — never render them as bubbles.
    if (isInjectedSystemNote(text)) continue;
    out.push({ id: uid(role), role, text, inProgress: false, status: "complete" });
  }
  return out;
}

/**
 * True if a message is a system-injected note that should never surface as a
 * user/assistant bubble (context compactions, model-switch notices, etc.).
 * Matches the `[SYSTEM: ...]` / `[CONTEXT COMPACTION ...]` / `[OUT-OF-BAND ...]`
 * markers the harness prepends mid-turn.
 */
function isInjectedSystemNote(text: string): boolean {
  const t = text.trimStart();
  return (
    t.startsWith("[CONTEXT COMPACTION") ||
    t.startsWith("[OUT-OF-BAND USER MESSAGE") ||
    t.startsWith("[System:") ||
    t.startsWith("[SYSTEM:")
  );
}

/* sessionStorage keys — the active session survives a page refresh so Cmd+R
   no longer dumps the user into an empty new chat. */
const SS_SESSION_KEY = "hermes.m.activeSession";
const SS_TITLE_KEY = "hermes.m.activeTitle";
const SS_MOBILE_SESSION_KEY = "hermes.m.mobileSession";
const BUILD_TAG = "build 2026-09-02.2 · picker-and-composer-heal";

function ssGet(key: string): string {
  try {
    return sessionStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function ssSet(key: string, value: string) {
  try {
    if (value) sessionStorage.setItem(key, value);
    else sessionStorage.removeItem(key);
  } catch {
    /* private mode — ignore */
  }
}

/**
 * Clean a session title/preview for display: strip raw internal references
 * (`@session:profile/id` leaks backend IDs into the UI) and markdown
 * punctuation that renders as litter in a plain-text row.
 */
function cleanPreview(raw: string | undefined | null): string {
  if (!raw) return "";
  let t = raw.replace(/@session:\S+/g, "").trim();
  t = t
    .replace(/[`*_#>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return t;
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function BubbleShell({
  role,
  failed = false,
  children,
}: {
  role: MsgRole;
  failed?: boolean;
  children: ReactNode;
}) {
  if (role === "system" || role === "error") {
    return (
      <div className="my-2 flex justify-center px-4">
        <p
          className={cn(
            "max-w-[85%] rounded-lg border border-border/60 px-3 py-2 text-center text-[0.8rem] leading-snug",
            role === "error" ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {children}
        </p>
      </div>
    );
  }

  const isUser = role === "user";
  const userFailed = isUser && failed;
  // User bubble: Nous-blue-tinted bubble, LIGHT text (matches the desktop
  // app's `userBubble: #07162c`). The message body renders through <Markdown>,
  // whose root sets `text-foreground` → `color:var(--midground)` (accent blue).
  // On the dark bubble that's fine, but we still override `--midground` to the
  // light foreground so Markdown text reads as plain light gray like on
  // desktop. The bubble background/border come from the mobile palette vars.
  // A FAILED send desaturates the fill and takes a red border so it can't be
  // mistaken for a delivered message.
  const userStyle = {
    background: userFailed
      ? "color-mix(in srgb, #f85149 14%, var(--background-base, #0d1117))"
      : "var(--mobile-user-bubble, var(--midground-base))",
    border: userFailed
      ? "1px solid color-mix(in srgb, #f85149 55%, transparent)"
      : "1px solid var(--mobile-user-bubble-border, transparent)",
    color: "var(--foreground-base, #e6edf3)",
    ["--midground" as string]: "var(--foreground-base, #e6edf3)",
  };
  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[86%] rounded-2xl px-4 py-2.5 text-[0.95rem] leading-relaxed",
          isUser
            ? "rounded-br-md"
            : "rounded-bl-md border border-border/40 bg-card",
        )}
        style={isUser ? userStyle : undefined}
      >
        {children}
      </div>
    </div>
  );
}

function CopyButton({ text, label = "copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onCopy = useCallback(() => {
    void copyTextToClipboard(text).then((copied) => {
      if (!copied) return;
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1600);
    });
  }, [text]);

  return (
    <button
      type="button"
      onClick={onCopy}
      aria-label={label}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[0.72rem] text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-success" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
      <span className="hidden min-[400px]:inline">
        {copied ? "copied" : label}
      </span>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */

export default function MobileChatPage() {
  const gwRef = useRef<GatewayClient | null>(null);
  const sessionIdRef = useRef<string>("");
  const mountedRef = useRef(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const [state, setState] = useState<ConnectionState>("idle");
  // Server-side reap (ws_orphan_reap) can leave the WS "open" while the
  // session underneath is gone. "live" then lies; show a warning state
  // until a send/reattach mints a fresh session id.
  const [sessionDetached, setSessionDetached] = useState(false);
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelOptions, setModelOptions] = useState<Awaited<ReturnType<typeof api.getModelOptions>> | null>(null);
  const [modelInfo, setModelInfo] = useState<Awaited<ReturnType<typeof api.getModelInfo>> | null>(null);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [composer, setComposer] = useState("");
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  // Soft-keyboard inset: iOS overlays the keyboard without shrinking the layout
  // viewport, so a `fixed` bottom composer hides under it. The overlay itself
  // is sized PURELY with CSS (inset-0) — never JS-resized — because in the iOS
  // standalone app visualViewport.height/offsetTop can report stale-shrunk
  // values (sized for a keyboard that already closed), which drew the whole
  // overlay shoved-up with a dead band above the home indicator. Instead, the
  // measured shrink is applied ONLY as bottom padding under the composer while
  // the keyboard is actually open.
  const [kbInset, setKbInset] = useState(0);

  // Past-chat access.
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [sessionErr, setSessionErr] = useState<string | null>(null);
  const [sessionFilter, setSessionFilter] = useState("");
  const [activeTitle, setActiveTitle] = useState("");

  // Settings sheet.
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [thinkingOpen, setThinkingOpen] = useState<boolean>(() => {
    try {
      return localStorage.getItem("hermes.mobile.thinking-open") === "1";
    } catch {
      return false;
    }
  });

  const onToggleThinking = useCallback(() => {
    setThinkingOpen((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("hermes.mobile.thinking-open", next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLDivElement | null>(null);
  const lastAssistantIdRef = useRef<string | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy, kbInset]);

  // Track the soft-keyboard as an inset ONLY. The overlay stays CSS-sized
  // (inset-0); this measures how much the keyboard occludes the bottom so the
  // composer can be lifted clear of it. Two rules that keep iOS from breaking:
  //  (1) when the keyboard opens, iOS pans the LAYOUT page up to reveal the
  //      focused input — undo that pan FIRST (scrollTo(0,0)) so we don't
  //      double-lift the composer (pan + inset = fly up). We pin whenever the
  //      keyboard is present, not only when scrollY!==0, because the pan can
  //      hide in visualViewport.offsetTop with scrollY still 0.
  //  (2) only apply a non-zero inset while the keyboard is genuinely open
  //      (shrink > 120px); stale-shrunk standalone values must never move it.
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const measure = () => {
      // PIN: any pan — vertical (keyboard reveal) or HORIZONTAL (wide content
      // making the page wider than the screen, which iOS happily scrolls and
      // drags the fixed overlay sideways, leaving void bars at the edges) —
      // gets snapped back to (0,0) immediately.
      if (window.scrollX !== 0 || window.scrollY !== 0) {
        window.scrollTo(0, 0);
      }
      const keyboardOpen = window.innerHeight - vv.height > 120;
      if (keyboardOpen) {
        // Undo iOS's focus-scroll pan FIRST, then measure the real occlusion.
        if (window.scrollY !== 0 || window.scrollX !== 0 || vv.offsetTop > 1) {
          window.scrollTo(0, 0);
        }
        const inset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        setKbInset(inset);
      } else {
        setKbInset(0);
        // Keyboard dismissed: snap the page back down if iOS left a residual
        // pan (the "stuck pushed-up" standalone quirk).
        if (window.scrollY !== 0 || window.scrollX !== 0) {
          window.scrollTo(0, 0);
        }
        if (vv.offsetTop > 1) {
          document.body.style.transform = "translateZ(0)";
          requestAnimationFrame(() => {
            document.body.style.transform = "";
          });
        }
      }
    };
    measure();
    vv.addEventListener("resize", measure);
    vv.addEventListener("scroll", measure);
    window.addEventListener("resize", measure);
    // iOS standalone: visualViewport events STOP firing after sheet dismissals
    // (file picker, share sheet), leaving the composer parked at a stale inset.
    // Watchdog: re-measure on visibility changes and on a short post-interaction
    // cadence so a stale viewport self-heals instead of waiting for an event
    // that never comes.
    const heal = () => requestAnimationFrame(measure);
    document.addEventListener("visibilitychange", heal);
    const healTimer = window.setInterval(heal, 1500);
    return () => {
      document.removeEventListener("visibilitychange", heal);
      window.clearInterval(healTimer);
      vv.removeEventListener("resize", measure);
      vv.removeEventListener("scroll", measure);
      window.removeEventListener("resize", measure);
    };
  }, []);

  // Keep the document from scrolling behind the overlay. NOTE: we deliberately
  // do NOT set body { position: fixed; top: 0 } here — that makes BODY the
  // containing block for the overlay's position:fixed, so when iOS pans the
  // page to reveal the keyboard (and keeps a residual pan afterwards on
  // standalone), the whole app rides up with the body and stays shoved-up
  // (top hidden under the Dynamic Island + a gap above the home indicator).
  // With plain overflow:hidden the overlay stays glued to the SCREEN and the
  // vv-based scrollTo(0,0) below snaps any residual pan back.
  useEffect(() => {
    const html = document.documentElement;
    const body = document.body;
    const prev = {
      htmlOverflow: html.style.overflow,
      bodyOverflow: body.style.overflow,
      htmlOverscroll: html.style.overscrollBehavior,
      bodyOverscroll: body.style.overscrollBehavior,
      htmlHeight: html.style.height,
      bodyHeight: body.style.height,
    };
    html.style.overflow = "hidden";
    body.style.overflow = "hidden";
    // Kill iOS rubber-banding: with scroll chaining allowed, flinging past the
    // top/bottom of the message list scrolls the DOCUMENT behind the overlay,
    // which bounces (overscroll) and re-rasterizes mid-motion (looks like the
    // app blurs itself). contain + none removes both.
    html.style.overscrollBehavior = "none";
    body.style.overscrollBehavior = "none";
    html.style.height = "100%";
    body.style.height = "100%";
    return () => {
      html.style.overflow = prev.htmlOverflow;
      body.style.overflow = prev.bodyOverflow;
      html.style.overscrollBehavior = prev.htmlOverscroll;
      body.style.overscrollBehavior = prev.bodyOverscroll;
      html.style.height = prev.htmlHeight;
      body.style.height = prev.bodyHeight;
    };
  }, []);

  // Kill the iOS form-assistant bar (↑↓ + ✓ above the keyboard) for good:
  // iOS shows it whenever the DOCUMENT contains more than one focusable TEXT
  // control. The file input is already on-demand, but the dashboard shell
  // behind the overlay renders its own inputs — some AFTER our mount — so a
  // one-time sweep is not enough. Watch the DOM continuously and disable any
  // text-entry control that is NOT inside our overlay. (Buttons/selects are
  // harmless: the bar's prev/next cycles text fields only.)
  useEffect(() => {
    const overlay = document.querySelector('[data-mobile-overlay="1"]');
    if (!overlay) return;
    const touched: HTMLInputElement[] = [];
    const sweep = () => {
      const controls = document.querySelectorAll<HTMLInputElement>(
        "input[type=text], input[type=search], input[type=email], input[type=url], input[type=tel], input[type=password], input[type=number], input:not([type]), textarea",
      );
      controls.forEach((el) => {
        if (overlay.contains(el) || el.disabled) return;
        el.disabled = true;
        touched.push(el);
      });
    };
    sweep();
    const mo = new MutationObserver(sweep);
    mo.observe(document.body, { childList: true, subtree: true });
    return () => {
      mo.disconnect();
      touched.forEach((el) => (el.disabled = false));
    };
  }, []);

  // iOS standalone viewport-heal (CONSERVATIVE): WebKit can leave the PWA
  // canvas stuck shrunken after a keyboard cycle. The display off→on flip
  // forces a re-measure BUT blanks the app for a frame — visibly janky if it
  // fires routinely. So: require the shrink to persist across two checks
  // 400ms apart (transient keyboard-close animation no longer triggers it),
  // and only flip when the keyboard is fully closed.
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    let maxVH = window.innerHeight;
    let pendingShrink = false;
    let t: ReturnType<typeof setTimeout> | null = null;
    const heal = () => {
      const el = document.querySelector<HTMLElement>('[data-mobile-overlay="1"]');
      if (!el) return;
      const scroller = el.querySelector<HTMLElement>(".flex-1.overflow-y-auto");
      const st = scroller ? scroller.scrollTop : 0;
      el.style.display = "none";
      void el.offsetHeight; // sync reflow, no paint between
      el.style.display = "";
      if (scroller) scroller.scrollTop = st;
    };
    const check = () => {
      maxVH = Math.max(maxVH, vv.height);
      const keyboardClosed = window.innerHeight - vv.height <= 120;
      if (maxVH - vv.height > 4 && keyboardClosed) {
        if (pendingShrink) {
          pendingShrink = false;
          heal(); // shrink persisted across two checks — genuinely stuck
        } else {
          pendingShrink = true;
          t = setTimeout(check, 400); // confirm before acting
        }
      } else {
        pendingShrink = false;
      }
    };
    const onResize = () => {
      if (t) clearTimeout(t);
      t = setTimeout(check, 250);
    };
    vv.addEventListener("resize", onResize);
    return () => {
      if (t) clearTimeout(t);
      vv.removeEventListener("resize", onResize);
    };
  }, []);

  // Block iOS page zoom entirely: since iOS 10, Safari ignores
  // user-scalable=no, and `gesturestart` events do NOT fire in the standalone
  // web view — so block multi-touch at the raw touchmove level instead, and
  // auto-heal with a viewport-meta flip (re-applying the meta forces Safari
  // to snap visual-viewport scale back to 1) if a zoom ever lands.
  useEffect(() => {
    const prevent = (e: Event) => e.preventDefault();
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 1) e.preventDefault();
    };
    document.addEventListener("gesturestart", prevent, { passive: false });
    document.addEventListener("touchmove", onTouchMove, { passive: false });
    const meta = document.querySelector<HTMLMetaElement>('meta[name="viewport"]');
    const vv = window.visualViewport;
    let healTimer: ReturnType<typeof setTimeout> | null = null;
    const healZoom = () => {
      if (!meta || !vv || vv.scale <= 1.01) return;
      // Flip a dummy attribute in and out: Safari re-applies the viewport on
      // content change, which resets zoom to 1.
      const original = meta.getAttribute("content") ?? "";
      meta.setAttribute("content", `${original},`);
      healTimer = setTimeout(() => meta.setAttribute("content", original), 60);
    };
    vv?.addEventListener("resize", healZoom);
    const body = document.body;
    const prevTouch = body.style.touchAction;
    body.style.touchAction = "pan-y";
    return () => {
      document.removeEventListener("gesturestart", prevent);
      document.removeEventListener("touchmove", onTouchMove);
      vv?.removeEventListener("resize", healZoom);
      if (healTimer) clearTimeout(healTimer);
      body.style.touchAction = prevTouch;
    };
  }, []);

  const appendMsg = useCallback((msg: ChatMsg) => {
    // Drop system-injected context-compaction handoffs and out-of-band markers
    // — they're never meant to render as bubbles.
    if (isInjectedSystemNote(msg.text)) return;
    setMessages((prev) => [...prev, msg]);
  }, []);

  const patchLastAssistant = useCallback(
    (patch: (m: ChatMsg) => ChatMsg) => {
      setMessages((prev) => {
        const id = lastAssistantIdRef.current;
        if (!id) return prev;
        return prev.map((m) => (m.id === id ? patch(m) : m));
      });
    },
    [],
  );

  /* ---- connect + create session ---- */
  useEffect(() => {
    mountedRef.current = true;
    let disposed = false;

    const gw = new GatewayClient();
    gwRef.current = gw;
    setState("connecting");

    gw
      .connect()
      .then(async () => {
        if (disposed) return;
        setState("open");
        // Refresh persistence: if this tab had an active session, resume it
        // (loads full history) instead of silently starting an empty chat.
        // Cold boot re-attaches ONLY to the phone's own last session. Sessions opened
// from the drawer that belong to Telegram/desktop/TUI are volatile on purpose.
const prevSid = ssGet(SS_MOBILE_SESSION_KEY);
        if (prevSid) {
          try {
            const res = await gw.request<{ session_id?: string; messages?: unknown[] }>(
              "session.resume",
              { session_id: prevSid, cols: 80 },
            );
            if (disposed) return;
            // session.resume returns the LIVE in-memory id in session_id and
            // the persisted state.db id in stored_session_id. Keep the live id
            // for prompt.submit targeting, but fetch REST history with the
            // persisted id (REST /messages only knows the stored row).
            sessionIdRef.current = res?.session_id || prevSid;
            const storedId =
              (res as { stored_session_id?: string } | null)?.stored_session_id || prevSid;
            const savedTitle = ssGet(SS_TITLE_KEY);
            if (savedTitle) setActiveTitle(savedTitle);
            let hist: ChatMsg[] = [];
            try {
              // REST /messages is oldest-first — render as-is so the chat
              // reads top→bottom with new messages appended at the bottom.
              const msgsRes = await api.getSessionMessages(storedId);
              hist = mapHistoryMessages(msgsRes.messages || []);
            } catch {
              hist = mapHistoryMessages(res?.messages);
            }
            setMessages(hist);
            console.info("[m] boot resume ok:", sessionIdRef.current, "msgs:", hist.length);
            return;
          } catch (e) {
            // Resume failed (session gone, gateway hiccup) — fall through to a
            // fresh session so the page is still usable.
            console.warn("[m] boot resume failed:", (e as Error)?.message);
            ssSet(SS_MOBILE_SESSION_KEY, "");
          }
        }
        try {
          const res = await gw.request<{ session_id: string }>("session.create", {
            source: "tool",
          });
          if (disposed) return;
          sessionIdRef.current = res?.session_id ?? "";
        } catch {
          if (!disposed) setBootError("failed to create session");
        }
      })
      .catch((e: Error) => {
        if (disposed) return;
        setState("error");
        setBootError(e.message || "connection failed");
      });

    gw.onState((s) => {
      if (mountedRef.current) setState(s);
      // Auto-reconnect: iOS suspends the PWA's socket when the app
      // backgrounds or the screen locks; the shared client intentionally
      // leaves reconnection to the page owner ("outer connection owner
      // decides"). Without this, every background/lock kills the session
      // until a manual reload. On reconnect, re-run the boot resume so the
      // live session reattaches and the transcript reloads.
      if (s === "closed" && !disposed) {
        reconnectTimerRef.current && clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = setTimeout(() => {
          if (disposed || mountedRef.current === false) return;
          if (gwRef.current !== gw) return;
          setState("connecting");
          gw.connect().catch(() => {
            // Retry with backoff; give up quietly after ~5 tries — the
            // user can always pull-to-refresh.
            reconnectAttemptsRef.current += 1;
            if (reconnectAttemptsRef.current <= 5) {
              reconnectTimerRef.current = setTimeout(() => {
                if (!disposed && gwRef.current === gw) {
                  gw.connect().catch(() => {});
                }
              }, Math.min(15000, 1000 * 2 ** reconnectAttemptsRef.current));
            }
          });
        }, 800);
      }
      if (s === "open") {
        reconnectAttemptsRef.current = 0;
        // Reattach to the previous session after any reconnect.
        const prevSid = ssGet(SS_SESSION_KEY);
        if (prevSid && sessionIdRef.current !== prevSid) {
          void gw
            .request<{ session_id?: string }>("session.resume", {
              session_id: prevSid,
              cols: 80,
            })
            .then((res) => {
              sessionIdRef.current = res?.session_id || prevSid;
            })
            .catch(() => {});
        }
      }
    });

    gw.on("message.start", () => {
      setBusy(true);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.inProgress) return prev;
        const id = uid("assistant");
        lastAssistantIdRef.current = id;
        return [
          ...prev,
          { id, role: "assistant", text: "", inProgress: true, status: "streaming" },
        ];
      });
    });

    gw.on("message.delta", (ev) => {
      const payload = ev?.payload as { text?: unknown } | undefined;
      const text = coerceText(payload?.text);
      if (!text) return;
      setMessages((prev) => {
        const id = lastAssistantIdRef.current;
        if (!id) {
          const nid = uid("assistant");
          lastAssistantIdRef.current = nid;
          return [
            ...prev,
            { id: nid, role: "assistant", text, inProgress: true, status: "streaming" },
          ];
        }
        return prev.map((m) =>
          m.id === id
            ? { ...m, text: m.text + text, inProgress: true, status: "streaming" }
            : m,
        );
      });
    });

    gw.on("reasoning.delta", (ev) => {
      const payload = ev?.payload as { text?: unknown } | undefined;
      const text = coerceText(payload?.text);
      if (!text) return;
      setMessages((prev) => {
        const id = lastAssistantIdRef.current;
        if (!id) {
          const nid = uid("assistant");
          lastAssistantIdRef.current = nid;
          return [
            ...prev,
            {
              id: nid,
              role: "assistant",
              text: "",
              reasoning: text,
              inProgress: true,
              status: "streaming",
            },
          ];
        }
        return prev.map((m) =>
          m.id === id ? { ...m, reasoning: (m.reasoning ?? "") + text } : m,
        );
      });
    });

    gw.on("tool.start", (ev) => {
      const payload = ev?.payload as { name?: unknown; tool?: unknown } | undefined;
      const name = coerceText(payload?.name) || coerceText(payload?.tool);
      if (!name) return;
      patchLastAssistant((m) => ({
        ...m,
        tools: [...new Set([...(m.tools ?? []), name])],
      }));
    });

    gw.on("message.complete", (ev) => {
      const payload = ev?.payload as Record<string, unknown> | undefined;
      const finalText = coerceText(payload?.text) || coerceText(payload?.rendered);
      const isError = payload?.status === "error";
      const errText =
        coerceText(payload?.error) || finalText || "Hermes reported an error";

      patchLastAssistant((m) => ({
        ...m,
        ...(finalText ? { text: finalText } : {}),
        inProgress: false,
        status: isError ? "error" : "complete",
        ...(isError ? { role: "error", text: errText } : {}),
      }));
      setBusy(false);
    });

    gw.on("session.info", (ev) => {
      const p = ev?.payload as Record<string, unknown> | undefined;
      if (p?.model) setModel(coerceText(p.model));
      if (p?.provider) setProvider(coerceText(p.provider));
      if (typeof p?.running === "boolean" && !p.running) setBusy(false);
    });

    gw.on("error", (ev) => {
      setBusy(false);
      const payload = ev?.payload as Record<string, unknown> | undefined;
      const msg =
        coerceText(payload?.message) || coerceText(payload?.text) || "error";
      appendMsg({
        id: uid("error"),
        role: "error",
        text: msg,
        inProgress: false,
        status: "error",
      });
    });

    return () => {
      disposed = true;
      mountedRef.current = false;
      try {
        gw.close();
      } catch {
        /* ignore */
      }
    };
  }, [appendMsg, patchLastAssistant]);

  /* ---- send ---- */
  const doSend = useCallback(async () => {
    const sid = sessionIdRef.current;
    const gw = gwRef.current;
    const text = composer.trim();
    // Photo-only sends are valid: bail only when there's no text AND no images.
    if (!sid || !gw || busy || (!text && pendingImages.length === 0)) return;

    const body = [...pendingImages.map((im) => `@image:${im.path}`), text].join(
      "\n\n",
    );

    appendMsg({
      id: uid("user"),
      role: "user",
      text: pendingImages.length > 0 && !text ? "📷 image" : text,
      inProgress: false,
      status: "complete",
    });

    // Persist the active session so a refresh resumes it instead of wiping.
    ssSet(SS_MOBILE_SESSION_KEY, sid);
    ssSet(SS_TITLE_KEY, activeTitle);

    setComposer("");
    if (composerRef.current) composerRef.current.innerText = "";
    setPendingImages([]);
    setBusy(true);
    const aid = uid("assistant");
    lastAssistantIdRef.current = aid;
    setMessages((prev) => [
      ...prev,
      { id: aid, role: "assistant", text: "", inProgress: true, status: "streaming" },
    ]);

    try {
      let submitText = body;
      if (pendingImages.length > 0) {
        try {
          const det = await gw.request<{ matched?: boolean; text?: string }>(
            "input.detect_drop",
            { session_id: sid, text: body },
          );
          if (det?.matched && det.text) submitText = det.text;
        } catch {
          /* fall back to body */
        }
      }
      await gw.request("prompt.submit", { session_id: sid, text: submitText }).catch(
        async (firstErr: unknown) => {
          // Server-side reap (ws_orphan_reap) can leave the WS "open" while
          // the session underneath is gone. One silent recovery: re-attach
          // (session.resume mints a fresh live id), then retry once.
          const storedSid = ssGet(SS_SESSION_KEY) || sid;
          await gw.request("session.resume", { session_id: storedSid, cols: 80 });
          const res = await gw.request<{ session_id?: string }>(
            "session.resume",
            { session_id: storedSid, cols: 80 },
          );
          const newSid = res?.session_id || storedSid;
          sessionIdRef.current = newSid;
          ssSet(SS_SESSION_KEY, newSid);
          setSessionDetached(false);
          await gw.request("prompt.submit", { session_id: newSid, text: submitText });
          void firstErr; // recovered — surface nothing
        },
      );
    } catch (e) {
      const err = e as Error;
      setSessionDetached(true);
      patchLastAssistant((m) => ({
        ...m,
        inProgress: false,
        status: "error",
        role: "error",
        text: `Send failed — tap your message to retry. (${err.message})`,
      }));
      // Mark the user bubble failed + tappable for one-tap resend.
      setMessages((prev) => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "user") {
            next[i] = { ...next[i], status: "error" };
            failedUserMsgRef.current = { id: next[i].id, body };
            break;
          }
        }
        return next;
      });
      setBusy(false);
    }
  }, [composer, pendingImages, busy, activeTitle, appendMsg, patchLastAssistant]);

  // One-tap retry: tapping a failed user bubble resends its original body.
  const failedUserMsgRef = useRef<{ id: string; body: string } | null>(null);
  const retryFailedSend = useCallback(
    async (msgId: string) => {
      const failed = failedUserMsgRef.current;
      if (!failed || failed.id !== msgId || busy) return;
      failedUserMsgRef.current = null;
      const gw = gwRef.current;
      const sid = sessionIdRef.current;
      if (!gw || !sid) return;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId ? { ...m, status: "complete" as const } : m,
        ),
      );
      setBusy(true);
      const aid = uid("assistant");
      lastAssistantIdRef.current = aid;
      setMessages((prev) => [
        ...prev,
        { id: aid, role: "assistant", text: "", inProgress: true, status: "streaming" },
      ]);
      try {
        await gw.request("prompt.submit", { session_id: sid, text: failed.body });
      } catch (e) {
        const err = e as Error;
        setSessionDetached(true);
        patchLastAssistant((m) => ({
          ...m,
          inProgress: false,
          status: "error",
          role: "error",
          text: `Send failed — tap your message to retry. (${err.message})`,
        }));
        setMessages((prev) =>
          prev.map((m) =>
            m.id === msgId ? { ...m, status: "error" as const } : m,
          ),
        );
        failedUserMsgRef.current = { id: msgId, body: failed.body };
        setBusy(false);
      }
    },
    [busy, patchLastAssistant],
  );

  const onAttach = useCallback(
    async (file: File | null) => {
      if (!file || !file.type.startsWith("image/")) return;
      setUploading(true);
      try {
        const dataUrl = await fileToDataUrl(file);
        const res = await uploadChatImage(file);
        setPendingImages((prev) => [
          ...prev,
          { dataUrl, path: res.path, name: res.name, bytes: res.bytes },
        ]);
      } catch (e) {
        const err = e as Error;
        appendMsg({
          id: uid("error"),
          role: "error",
          text: `upload failed: ${err.message}`,
          inProgress: false,
          status: "error",
        });
      } finally {
        setUploading(false);
      }
    },
    [appendMsg],
  );

  // On-demand file picker: creating the <input type="file"> only when the
  // camera button is tapped keeps it OUT of the DOM the rest of the time, so
  // iOS never sees a second form field (that's what brought back the
  // form-assistant bar with the up/down arrows above the keyboard).
  const pickImage = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.multiple = true;
    let settled = false;
    input.addEventListener("change", () => {
      settled = true;
      const files = Array.from(input.files ?? []);
      // iOS standalone PWA: the native picker sheet can stay on screen while
      // the <input> element exists in the DOM. Detach + blur immediately.
      input.value = "";
      input.remove();
      if (files.length === 0) return;
      // Upload sequentially, yielding to the main thread between files —
      // 20 back-to-back base64 conversions of iPhone photos both spike memory
      // and freeze repaints (picker sheet looks "stuck open").
      (async () => {
        for (let i = 0; i < files.length; i++) {
          await new Promise((r) => requestAnimationFrame(() => r(null)));
          await onAttach(files[i]);
        }
      })();
    });
    // Sheet dismissed without picking anything — detach the input too.
    input.addEventListener("cancel", () => {
      if (!settled) {
        input.value = "";
        input.remove();
      }
    });
    input.click();
  }, [onAttach]);

  const startNewChat = useCallback(async () => {
    setMessages([]);
    lastAssistantIdRef.current = null;
    setComposer("");
    if (composerRef.current) composerRef.current.innerText = "";
    setPendingImages([]);
    setBusy(false);
    setActiveTitle("");
    ssSet(SS_MOBILE_SESSION_KEY, "");
    ssSet(SS_TITLE_KEY, "");
    try {
      const res = await gwRef.current?.request<{ session_id?: string }>(
        "session.create",
        { source: "tool" },
      );
      if (res?.session_id) sessionIdRef.current = res.session_id;
    } catch {
      /* keep the old sid; next send will surface any real error */
    }
  }, []);

  const openHistory = useCallback(async () => {
    setHistoryOpen(true);
    setLoadingSessions(true);
    setSessionErr(null);
    try {
      const res = await api.getSessions(30, 0, "", "recent");
      setSessions(res.sessions);
    } catch (e) {
      setSessionErr((e as Error).message || "failed to load chats");
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  const resumeSession = useCallback(
    async (id: string, title: string) => {
      const gw = gwRef.current;
      if (!gw) return;
      setHistoryOpen(false);
      setActiveTitle(title);
      // Persist only phone-owned sessions across launches. Opening a session from another
// surface (Telegram/desktop/TUI) stays volatile: it renders now, but the next cold
// boot re-attaches to the phone's own last session instead of hijacking that chat.
ssSet(SS_MOBILE_SESSION_KEY, id);
      ssSet(SS_TITLE_KEY, title);
      setBusy(false);
      setMessages([]);
      lastAssistantIdRef.current = null;
      sessionIdRef.current = id;
      try {
        const res = await gw.request<{
          session_id?: string;
          messages?: unknown[];
        }>("session.resume", {
          session_id: id,
          cols: 80,
        });
        if (res?.session_id) sessionIdRef.current = res.session_id;

        // The authoritative transcript is the REST /messages endpoint — the
        // resume payload only carries a live tool-event feed. Always fetch the
        // text history for display; fall back to the resume payload only if the
        // REST read fails. REST /messages is oldest-first — render as-is so the
        // chat reads top→bottom with new messages appended at the bottom.
        let hist: ChatMsg[] = [];
        try {
          const msgsRes = await api.getSessionMessages(id);
          hist = mapHistoryMessages(msgsRes.messages || []);
        } catch {
          hist = mapHistoryMessages(res?.messages);
        }
        setMessages(hist);
      } catch (e) {
        appendMsg({
          id: uid("error"),
          role: "error",
          text: `resume failed: ${(e as Error).message}`,
          inProgress: false,
          status: "error",
        });
      }
    },
    [appendMsg],
  );

  const stop = useCallback(() => {
    const gw = gwRef.current;
    const sid = sessionIdRef.current;
    if (gw && sid) {
      void gw.request("session.interrupt", { session_id: sid }).catch(() => {});
    }
    patchLastAssistant((m) => ({ ...m, inProgress: false, status: "complete" }));
    setBusy(false);
  }, [patchLastAssistant]);

  /* ---- refresh transcript ---- */
  // Re-pulls the authoritative REST history for the CURRENT session. Use
  // case: resuming a past chat can land on a stale snapshot (session was
  // active on another surface — desktop, another device — and REST served
  // the page before those turns persisted). The refresh button re-fetches
  // and keeps scroll pinned to the newest message.
  const [refreshing, setRefreshing] = useState(false);
  const refreshTranscript = useCallback(async () => {
    const storedId = ssGet(SS_SESSION_KEY);
    if (!storedId || refreshing) return;
    setRefreshing(true);
    try {
      const msgsRes = await api.getSessionMessages(storedId);
      const hist = mapHistoryMessages(msgsRes.messages || []);
      if (hist.length) setMessages(hist);
    } catch {
      /* keep current view; a failed refresh shouldn't blank the chat */
    } finally {
      setRefreshing(false);
    }
  }, [refreshing]);

  /* ---- manual reconnect (header refresh button) ---- */
  const manualReconnect = useCallback(() => {
    const gw = gwRef.current;
    if (!gw) return;
    reconnectAttemptsRef.current = 0;
    setState("connecting");
    gw.connect()
      .then(() => {
        if (!mountedRef.current) return;
        const prevSid = ssGet(SS_SESSION_KEY);
        if (prevSid) {
          void gw
            .request<{ session_id?: string }>("session.resume", { session_id: prevSid, cols: 80 })
            .then((res) => {
              sessionIdRef.current = res?.session_id || prevSid;
            })
            .catch(() => {});
        }
      })
      .catch(() => {
        if (mountedRef.current) setState("error");
      });
  }, []);

  /* ---- model picker ---- */
  const openModelPicker = useCallback(() => {
    setModelPickerOpen(true);
    void api.getModelInfo().then(setModelInfo).catch(() => {});
    void api.getModelOptions().then(setModelOptions).catch(() => {});
  }, []);

  const switchModel = useCallback(
    async (prov: string, mdl: string) => {
      setModelSwitching(true);
      try {
        // Hot-swap the LIVE session: config.set with key="model" +
        // session_id (same mechanism as the desktop /model command). The
        // gateway splices a model-switch marker into the live history, so
        // the next turn runs on the new model with full conversation
        // context — no fresh chat needed.
        const gw = gwRef.current;
        const sid = sessionIdRef.current;
        if (!gw || !sid) throw new Error("not connected");
        await gw.request("config.set", {
          key: "model",
          session_id: sid,
          value: `${prov}/${mdl}`,
        });
        setModel(mdl);
        setProvider(prov);
        setModelPickerOpen(false);
      } catch {
        /* leave picker open; unchanged model signals the failure */
      } finally {
        setModelSwitching(false);
      }
    },
    [],
  );

  /* ---- render ---- */
  const showEmpty = messages.length === 0;
  const last = messages[messages.length - 1];
  const eff = !last?.inProgress;

  return createPortal(
    <div
      data-mobile-overlay="1"
      className="fixed inset-0 z-[200] flex flex-col overflow-hidden bg-background-base"
      style={{
        // Opaque app background — with the translucent status bar the page
        // content extends under it; without an explicit fill iOS can blend
        // the underlying snapshot (the "blurred top" artifact). Size is PURE
        // CSS (inset-0): never JS-resized, so stale standalone viewport
        // values can't shove the app up or shrink it.
        background: "var(--background-base)",
        paddingLeft: "env(safe-area-inset-left)",
        paddingRight: "env(safe-area-inset-right)",
        // Keyboard inset as bottom PADDING on the container (not margin on the
        // composer): it shrinks the flex content area so the last child (the
        // composer) is always pinned just above the keyboard. A margin on the
        // composer lets iOS's focus pan push it out of view.
        paddingBottom: kbInset,
      }}
    >
      {/* Header */}
      <header className="flex shrink-0 items-center gap-2 border-b border-border/40 px-4 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3">
        <Sparkles className="h-5 w-5 text-primary" aria-hidden />
        <div className="min-w-0 flex-1 overflow-hidden">
          <button
            type="button"
            onClick={openModelPicker}
            className="block w-full min-w-0 text-left"
          >
            <h1 className="truncate text-[1.05rem] font-bold tracking-[0.02em] text-foreground">
              {activeTitle || "Hermes"}
            </h1>
            <span className="block truncate text-[0.75rem] text-muted-foreground">
              {model || "—"}
              {provider ? ` · ${provider}` : ""} ⌄
            </span>
          </button>
        </div>
        {state !== "open" && (
          <button
            type="button"
            onClick={manualReconnect}
            aria-label="Reconnect"
            className="flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[0.72rem] text-warning"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            {STATE_LABEL[state]}
          </button>
        )}
        {state === "open" && (
          <span
            className="flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[0.72rem]"
            style={{ color: sessionDetached ? "var(--warning, #d29922)" : "var(--success, #3fb950)" }}
          >
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: sessionDetached ? "var(--warning, #d29922)" : "var(--success, #3fb950)" }}
            />
            {sessionDetached ? "reattach…" : "live"}
          </span>
        )}
        <button
          type="button"
          onClick={() => void refreshTranscript()}
          aria-label="Refresh messages"
          title="Refresh messages"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
        >
          <RefreshCw className={cn("h-5 w-5", refreshing && "animate-spin")} />
        </button>
        <button
          type="button"
          onClick={() => void openHistory()}
          aria-label="Past chats"
          title="Past chats"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
        >
          <History className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={() => void startNewChat()}
          aria-label="New chat"
          title="New chat"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
        >
          <MessageSquarePlus className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          aria-label="Settings"
          title="Settings"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
        >
          <Settings className="h-5 w-5" />
        </button>
      </header>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto overflow-x-hidden overscroll-contain [touch-action:pan-y] py-4"
        style={{ WebkitOverflowScrolling: "touch" }}
      >
        {showEmpty ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
            <Sparkles className="h-8 w-8 text-muted-foreground/60" />
            <p className="max-w-[16rem] text-[0.9rem] text-muted-foreground">
              Ask anything. Type below, or tap the camera to send a photo.
            </p>
            {/* Build tag: lets Eric (and me) verify at a glance which bundle
                the phone is actually running — iOS caches PWAs aggressively. */}
            <p className="text-[0.65rem] text-muted-foreground/40 select-none">
              {BUILD_TAG}
            </p>
            {bootError && (
              <p className="text-[0.78rem] text-destructive">{bootError}</p>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {messages.map((m) => {
              if (assistantEmpty(m) && m.inProgress) {
                return (
                  <div key={m.id} className="flex flex-col">
                    <BubbleShell role={m.role}>
                      <span
                        aria-hidden
                        className="inline-block h-[1em] w-[0.5em] animate-pulse align-[-0.15em]"
                        style={{
                          backgroundColor: "var(--midground)",
                          opacity: 0.6,
                        }}
                      />
                    </BubbleShell>
                  </div>
                );
              }
              if (assistantEmpty(m)) return null;
              const thinkingLive =
                m.role === "assistant" &&
                m.inProgress &&
                !m.text &&
                !!m.reasoning;
              return (
                <div key={m.id} className="flex flex-col">
                  <BubbleShell role={m.role} failed={m.status === "error"}>
                    {m.role === "user" && m.status === "error" && (
                      <button
                        type="button"
                        onClick={() => void retryFailedSend(m.id)}
                        className="mb-1 w-full text-left text-[0.7rem] font-medium text-destructive"
                      >
                        not delivered — tap to retry ↻
                      </button>
                    )}
                    {m.role === "assistant" && m.reasoning && (
                      <details
                        className="mb-2"
                        // While the model is actively thinking (no answer text
                        // yet) the reasoning renders expanded + live; once the
                        // answer starts it collapses to a tappable "thinking"
                        // chip. Default-open only affects the first mount; the
                        // controlled toggle below keeps user choice after.
                        open={thinkingLive ? true : undefined}
                        key={thinkingLive ? "live" : "done"}
                      >
                        <summary className="flex cursor-pointer items-center gap-1.5 text-[0.7rem] text-muted-foreground">
                          {thinkingLive && (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          )}
                          thinking
                        </summary>
                        <div className="mt-1 max-h-48 overflow-y-auto border-l-2 border-border/50 pl-2 text-[0.82rem] text-muted-foreground">
                          <Markdown content={m.reasoning} streaming={thinkingLive} />
                        </div>
                      </details>
                    )}
                    <Markdown content={m.text} streaming={m.inProgress} codeCopy />
                    {m.tools && m.tools.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1 text-[0.7rem] text-muted-foreground">
                        {m.tools.map((t) => (
                          <span key={t} className="rounded border border-border/60 px-1.5 py-0.5">
                            ⚙ {t}
                          </span>
                        ))}
                      </div>
                    )}
                  </BubbleShell>
                  {m.role === "assistant" && !m.inProgress && eff && (
                    <div className="flex justify-start px-4 pt-1">
                      <CopyButton text={m.text} label="copy" />
                      {m.status === "error" && (
                        <span className="ml-1 self-center text-[0.72rem] text-destructive">
                          failed
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Standalone "thinking…" pill only while the assistant bubble hasn't
            taken any content yet — once it has (text/reasoning/tool chips),
            the bubble itself shows the in-progress caret and a second working
            indicator would just read as a duplicate stacked bubble. */}
        {busy &&
          !messages.some((m) => m.role === "assistant" && !assistantEmpty(m)) && (
            <div className="flex justify-start px-4 pt-3">
              <div className="inline-flex items-center gap-2 rounded-2xl rounded-bl-md border border-border/50 bg-card px-4 py-2 text-[0.82rem] text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                thinking…
              </div>
            </div>
          )}
      </div>

      {/* Pending image thumbnails */}
      {pendingImages.length > 0 && (
        <div className="flex shrink-0 gap-2 overflow-x-auto px-4 pb-2 scrollbar-none">
          {pendingImages.map((im, i) => (
            <div key={i} className="relative shrink-0">
              <img
                src={im.dataUrl}
                alt={im.name}
                className="h-16 w-16 rounded-lg border border-border object-cover"
              />
              <button
                type="button"
                onClick={() =>
                  setPendingImages((prev) => prev.filter((_, j) => j !== i))
                }
                aria-label="Remove image"
                className="absolute -right-1.5 -top-1.5 rounded-full bg-background-base p-0.5 text-foreground shadow"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Composer — the overlay container's bottom padding (kbInset, applied
          in the root style above) lifts this above the soft keyboard. Keep the
          safe-area bottom padding here for when the keyboard is closed. */}
      <div
        className="shrink-0 border-t border-border/40 bg-background-base px-4 pt-2.5"
        style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}
      >
        <div className="flex items-end gap-2">
          <button
            type="button"
            onClick={pickImage}
            disabled={uploading}
            aria-label="Attach photo"
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border/60 text-muted-foreground transition-colors hover:text-foreground active:text-foreground disabled:opacity-50"
          >
            {uploading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Camera className="h-5 w-5" />}
          </button>

          <div className="min-w-0 flex-1 rounded-2xl border border-border/60 bg-card px-3 focus-within:border-border">
          <div
            ref={composerRef}
            contentEditable
            suppressContentEditableWarning
            role="textbox"
            aria-multiline="true"
            aria-label="Message Hermes"
            data-placeholder="Message Hermes…"
            onInput={(e) => setComposer((e.target as HTMLDivElement).innerText)}
            onPaste={(e) => {
              // Paste as plain text so iOS doesn't inject styled spans that
              // break innerText tracking.
              e.preventDefault();
              const text = e.clipboardData.getData("text/plain");
              document.execCommand("insertText", false, text);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void doSend();
              }
            }}
            // inputmode flicker (last documented PWA lever against the iOS
            // form-assistant bar): on focus, briefly claim "no keyboard", then
            // restore text mode ~80ms in. The window can suppress the
            // accessory bar (↑↓/✓) entirely; inputmode is only valid on
            // input/textarea, so it lives on the wrapper and is mirrored onto
            // the contenteditable via a data attr + focus handler.
            data-inputmode-flicker="1"
            onFocus={(e) => {
              const el = e.currentTarget as HTMLElement & {
                inputmode?: string;
              };
              try {
                el.setAttribute("inputmode", "none");
                window.setTimeout(() => {
                  el.setAttribute("inputmode", "text");
                }, 80);
              } catch {
                /* attribute tricks are best-effort */
              }
            }}
            autoCapitalize="sentences"
            autoCorrect="on"
            spellCheck
            className="max-h-[7.5rem] w-full overflow-y-auto whitespace-pre-wrap py-2.5 text-[16px] leading-normal text-foreground outline-none placeholder:text-muted-foreground empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground"
          />
          </div>

          {busy ? (
            <button
              type="button"
              onClick={stop}
              aria-label="Stop"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-destructive text-destructive-foreground"
            >
              <Eraser className="h-5 w-5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void doSend()}
              disabled={!composer.trim() && pendingImages.length === 0}
              aria-label="Send"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl disabled:opacity-30"
              style={{
                backgroundColor: "var(--midground-base)",
                color: "var(--background-base)",
              }}
            >
              <Send className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* Model picker sheet */}
      {modelPickerOpen && (
        <div
          className="absolute inset-0 z-40 flex flex-col bg-background-base"
          style={{ background: "var(--background-base)" }}
          data-testid="mobile-model-sheet"
        >
          <header className="flex shrink-0 items-center gap-2 border-b border-border/40 px-3 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3">
            <button
              type="button"
              onClick={() => setModelPickerOpen(false)}
              aria-label="Back"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <h2 className="flex-1 text-[1rem] font-bold text-foreground">Model</h2>
            {modelSwitching && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          </header>
          <div className="flex-1 overflow-y-auto overscroll-contain px-3 py-3">
            {!modelOptions && (
              <p className="px-2 py-6 text-center text-[0.85rem] text-muted-foreground">loading…</p>
            )}
            {modelOptions?.providers
              ?.filter((p) => p.authenticated !== false && (p.models?.length ?? 0) > 0)
              .map((p) => (
                <div key={p.slug} className="mb-4">
                  <h3 className="px-2 pb-1 text-[0.72rem] font-semibold uppercase tracking-wide text-muted-foreground">
                    {p.name}
                  </h3>
                  {p.models?.map((m) => {
                    const current =
                      modelInfo?.provider === p.slug && modelInfo?.model === m;
                    return (
                      <button
                        key={m}
                        type="button"
                        disabled={modelSwitching}
                        onClick={() => void switchModel(p.slug, m)}
                        className={cn(
                          "flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-[0.92rem]",
                          current
                            ? "bg-card text-foreground"
                            : "text-muted-foreground active:bg-card/60",
                        )}
                      >
                        <span className="truncate">{m}</span>
                        {current && <Check className="h-4 w-4 shrink-0 text-success" />}
                      </button>
                    );
                  })}
                </div>
              ))}
          </div>
          <p className="shrink-0 px-4 pb-[max(0.5rem,env(safe-area-inset-bottom))] pt-1 text-center text-[0.7rem] text-muted-foreground">
            applies next message · chat keeps its history
          </p>
        </div>
      )}

      {/* Past chats sheet */}
      {historyOpen && (
        <div
          className="absolute inset-0 z-40 flex flex-col bg-background-base"
          style={{ background: "var(--background-base)" }}
          data-testid="mobile-chats-sheet"
        >
          <header className="flex shrink-0 items-center gap-2 border-b border-border/40 px-3 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3">
            <button
              type="button"
              onClick={() => setHistoryOpen(false)}
              aria-label="Back"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <h2 className="flex-1 text-[1rem] font-bold text-foreground">Chats</h2>
          </header>
          <div
            className="flex-1 overflow-y-auto"
            style={{ WebkitOverflowScrolling: "touch" }}
          >
            {loadingSessions ? (
              <div className="flex justify-center py-12">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : sessionErr ? (
              <p className="px-4 py-8 text-center text-[0.85rem] text-destructive">
                {sessionErr}
              </p>
            ) : !sessions?.length ? (
              <p className="px-4 py-8 text-center text-[0.85rem] text-muted-foreground">
                No past chats yet.
              </p>
            ) : (
              <>
                <input
                  type="search"
                  value={sessionFilter}
                  onChange={(e) => setSessionFilter(e.target.value)}
                  placeholder="Search chats"
                  aria-label="Search chats"
                  className="mb-1 block w-full border-b border-border/40 bg-transparent px-4 py-2.5 text-[16px] text-foreground outline-none placeholder:text-muted-foreground/60"
                  data-testid="mobile-chats-search"
                />
                {(() => {
                  const needle = sessionFilter.trim().toLowerCase();
                  const visible = needle
                    ? sessions.filter((s) =>
                        (s.title || s.preview || "").toLowerCase().includes(needle),
                      )
                    : sessions;
                  if (!visible.length) {
                    return (
                      <p className="px-4 py-8 text-center text-[0.85rem] text-muted-foreground">
                        No matches for “{sessionFilter}”.
                      </p>
                    );
                  }
                  return (
                    <ul className="divide-y divide-border/40" data-testid="mobile-chats-list">
                      {visible.map((s) => (
                        <li key={s.id}>
                          <button
                            type="button"
                            onClick={() => void resumeSession(s.id, s.title || "")}
                            data-testid="mobile-chat-row"
                            className="w-full px-4 py-3 text-left"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="min-w-0 flex-1 truncate text-[0.9rem] font-medium text-foreground">
                                {cleanPreview(s.title) ||
                                  cleanPreview(s.preview) ||
                                  "Untitled"}
                              </span>
                              {s.message_count > 0 && (
                                <span className="shrink-0 text-[0.68rem] text-muted-foreground">
                                  {s.message_count}{" "}
                                  {s.message_count === 1 ? "msg" : "msgs"} ·{" "}
                                  {timeAgo(s.last_active)}
                                </span>
                              )}
                            </div>
                            {cleanPreview(s.preview) && (
                              <p className="mt-0.5 truncate text-[0.78rem] text-muted-foreground">
                                {cleanPreview(s.preview)}
                              </p>
                            )}
                          </button>
                        </li>
                      ))}
                    </ul>
                  );
                })()}
              </>
            )}
          </div>
        </div>
      )}

      {/* Settings sheet */}
      {settingsOpen && (
        <div
          className="absolute inset-0 z-40 flex flex-col bg-background-base"
          style={{ background: "var(--background-base)" }}
          data-testid="mobile-settings-sheet"
        >
          <header className="flex shrink-0 items-center gap-2 border-b border-border/40 px-3 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3">
            <button
              type="button"
              onClick={() => setSettingsOpen(false)}
              aria-label="Back"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground active:text-foreground"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <h2 className="flex-1 text-[1rem] font-bold text-foreground">Settings</h2>
          </header>
          <div
            className="flex-1 overflow-y-auto p-4 space-y-4"
            style={{ WebkitOverflowScrolling: "touch" }}
          >
            {/* Palette note */}
            <div className="rounded-xl border border-border/40 bg-card p-3.5">
              <p className="text-[0.78rem] leading-relaxed text-muted-foreground">
                Hermes mobile uses the scoped Nous-dark palette.
              </p>
            </div>

            {/* Show thinking by default toggle */}
            <div className="flex items-center justify-between rounded-xl border border-border/40 bg-card p-3.5">
              <span className="text-[0.88rem] font-medium text-foreground">
                Show thinking by default
              </span>
              <button
                type="button"
                role="switch"
                aria-checked={thinkingOpen}
                onClick={onToggleThinking}
                aria-label="Show thinking by default"
                className={cn(
                  "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none",
                  thinkingOpen ? "bg-primary" : "bg-muted",
                )}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out",
                    thinkingOpen ? "translate-x-5" : "translate-x-0",
                  )}
                />
              </button>
            </div>

            {/* Version line */}
            <div className="rounded-xl border border-border/40 bg-card p-3.5">
              <p className="text-[0.78rem] text-muted-foreground select-none">
                {BUILD_TAG}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>,
    document.body,
  );
}
