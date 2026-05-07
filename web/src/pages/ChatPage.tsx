/**
 * ChatPage — embeds `hermes --tui` inside the dashboard.
 *
 *   <div host> (dashboard chrome)                                         .
 *     └─ <div wrapper> (rounded, dark bg, padded — the "terminal window"  .
 *         look that gives the page a distinct visual identity)            .
 *         └─ @xterm/xterm Terminal (WebGL renderer, Unicode 11 widths)    .
 *              │ onData      keystrokes → WebSocket → PTY master          .
 *              │ onResize    terminal resize → `\x1b[RESIZE:cols;rows]`   .
 *              │ write(data) PTY output bytes → VT100 parser              .
 *              ▼                                                          .
 *     WebSocket /api/pty?token=<session>                                  .
 *          ▼                                                              .
 *     FastAPI pty_ws  (hermes_cli/web_server.py)                          .
 *          ▼                                                              .
 *     POSIX PTY → `node ui-tui/dist/entry.js` → tui_gateway + AIAgent     .
 */

import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { WebglAddon } from "@xterm/addon-webgl";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { cn } from "@/lib/utils";
import { Copy, Headphones, Mic, PanelRight, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useSearchParams } from "react-router-dom";

import { ChatSidebar } from "@/components/ChatSidebar";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import {
  DEFAULT_HANDS_FREE_MIC_STATS,
  HANDS_FREE_INITIAL_NOISE_FLOOR,
  HANDS_FREE_MAX_SPOKEN_CHARS,
  HANDS_FREE_MAX_UTTERANCE_MS,
  HANDS_FREE_METER_UPDATE_MS,
  HANDS_FREE_MIN_RECORDING_MS,
  HANDS_FREE_NOISE_ALPHA,
  HANDS_FREE_SILENCE_MS,
  HANDS_FREE_SILENT_WAV,
  HANDS_FREE_TRUNCATION_NOTICE,
  handsFreeMeterPercent,
  handsFreeSilenceThreshold,
  handsFreeSpeechThreshold,
  handsFreeStateLabel,
  handsFreeStatusText,
  isHandsFreeDisableCommand,
  isHandsFreeStopCommand,
  rmsFromTimeDomain,
  sanitizeHandsFreeSpeechText,
  splitHandsFreeSpeech,
  takeReadyHandsFreeSpeechChunks,
  type HandsFreeMicStats,
  type HandsFreeState,
} from "@/lib/handsFreeVoice";
import { PluginSlot } from "@/plugins";

function buildWsUrl(
  token: string,
  resume: string | null,
  channel: string,
): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams({ token, channel });
  if (resume) qs.set("resume", resume);
  return `${proto}//${window.location.host}/api/pty?${qs.toString()}`;
}

function buildEventsWsUrl(token: string, channel: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams({ token, channel });
  return `${proto}//${window.location.host}/api/events?${qs.toString()}`;
}

// Channel id ties this chat tab's PTY child (publisher) to its sidebar
// (subscriber).  Generated once per mount so a tab refresh starts a fresh
// channel — the previous PTY child terminates with the old WS, and its
// channel auto-evicts when no subscribers remain.
function generateChannelId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `chat-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

// Colors for the terminal body.  Matches the dashboard's dark teal canvas
// with cream foreground — we intentionally don't pick monokai or a loud
// theme, because the TUI's skin engine already paints the content; the
// terminal chrome just needs to sit quietly inside the dashboard.
const TERMINAL_THEME = {
  background: "#0d2626",
  foreground: "#f0e6d2",
  cursor: "#f0e6d2",
  cursorAccent: "#0d2626",
  selectionBackground: "#f0e6d244",
};

/**
 * CSS width for xterm font tiers.
 *
 * Prefer the terminal host's `clientWidth` — Chrome DevTools device mode often
 * keeps `window.innerWidth` at the full desktop value while the *drawn* layout
 * is phone-sized, which made us pick desktop font sizes (~14px) and look huge.
 */
function terminalTierWidthPx(host: HTMLElement | null): number {
  if (typeof window === "undefined") return 1280;
  const fromHost = host?.clientWidth ?? 0;
  if (fromHost > 2) return Math.round(fromHost);
  const doc = document.documentElement?.clientWidth ?? 0;
  const vv = window.visualViewport;
  const inner = window.innerWidth;
  const vvw = vv?.width ?? inner;
  const layout = Math.min(inner, vvw, doc > 0 ? doc : inner);
  return Math.max(1, Math.round(layout));
}

function terminalFontSizeForWidth(layoutWidthPx: number): number {
  if (layoutWidthPx < 300) return 7;
  if (layoutWidthPx < 360) return 8;
  if (layoutWidthPx < 420) return 9;
  if (layoutWidthPx < 520) return 10;
  if (layoutWidthPx < 720) return 11;
  if (layoutWidthPx < 1024) return 12;
  return 14;
}

function terminalLineHeightForWidth(layoutWidthPx: number): number {
  return layoutWidthPx < 1024 ? 1.02 : 1.15;
}

function clipboardCanReadText(): boolean {
  return !!navigator.clipboard?.readText && window.isSecureContext;
}

function copyText(text: string): void {
  if (navigator.clipboard?.writeText && window.isSecureContext) {
    navigator.clipboard.writeText(text).catch((err) => {
      console.warn("[dashboard clipboard] direct copy failed:", err.message);
    });
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
  } catch (err) {
    console.warn("[dashboard clipboard] fallback copy failed:", err);
  } finally {
    textarea.remove();
  }
}

function preferredAudioMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/aac",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

function extensionForAudioType(mimeType: string): string {
  const mediaType = mimeType.split(";", 1)[0].trim().toLowerCase();
  if (mediaType === "audio/mp4") return "mp4";
  if (mediaType === "audio/aac") return "aac";
  if (mediaType === "audio/ogg") return "ogg";
  if (mediaType === "audio/wav" || mediaType === "audio/wave") return "wav";
  return "webm";
}

async function transcribeBrowserAudio(blob: Blob, token: string): Promise<string> {
  const type = blob.type || "audio/webm";
  const qs = new URLSearchParams({
    filename: `dashboard-voice.${extensionForAudioType(type)}`,
  });
  const resp = await fetch(`/api/voice/transcribe?${qs.toString()}`, {
    method: "POST",
    headers: {
      "Content-Type": type,
      "X-Hermes-Session-Token": token,
    },
    body: blob,
  });

  let payload: {
    detail?: string;
    error?: string;
    filtered?: boolean;
    provider?: string;
    success?: boolean;
    transcript?: string;
  } = {};
  try {
    payload = await resp.json();
  } catch {
    // Keep the HTTP error below useful when the server returns non-JSON.
  }

  if (!resp.ok) {
    throw new Error(payload.detail || payload.error || `transcription failed (${resp.status})`);
  }
  if (payload.success === false) {
    throw new Error(payload.error || "transcription failed");
  }
  return payload.transcript ?? "";
}

async function synthesizeBrowserSpeech(text: string, token: string): Promise<Blob> {
  const resp = await fetch("/api/voice/synthesize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Hermes-Session-Token": token,
    },
    body: JSON.stringify({ text }),
  });

  if (!resp.ok) {
    let payload: { detail?: string; error?: string } = {};
    try {
      payload = await resp.json();
    } catch {
      // Keep the fallback HTTP status below useful for non-JSON errors.
    }
    throw new Error(payload.detail || payload.error || `speech synthesis failed (${resp.status})`);
  }

  return resp.blob();
}

interface WakeLockSentinelLike {
  release(): Promise<void>;
}

type WakeLockNavigator = Navigator & {
  wakeLock?: {
    request(type: "screen"): Promise<WakeLockSentinelLike>;
  };
};

interface RpcEventFrame {
  method?: string;
  params?: {
    type?: string;
    payload?: {
      text?: string;
      rendered?: string;
    };
  };
}

function setAudioSource(audio: HTMLAudioElement, src: string, volume: number): void {
  audio.preload = "auto";
  audio.src = src;
  audio.volume = volume;
}

function clearAudioSource(audio: HTMLAudioElement): void {
  audio.pause();
  audio.onended = null;
  audio.onerror = null;
  audio.removeAttribute("src");
  audio.load();
}

function rewindAudio(audio: HTMLAudioElement): void {
  audio.currentTime = 0;
}

export default function ChatPage({ isActive = true }: { isActive?: boolean }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const ptyBusyRef = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recorderStreamRef = useRef<MediaStream | null>(null);
  const recorderChunksRef = useRef<BlobPart[]>([]);
  const [recording, setRecording] = useState(false);
  const [transcribingVoice, setTranscribingVoice] = useState(false);
  const handsFreeStreamRef = useRef<MediaStream | null>(null);
  const handsFreeAudioContextRef = useRef<AudioContext | null>(null);
  const handsFreeAnalyserRef = useRef<AnalyserNode | null>(null);
  const handsFreeSamplesRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const handsFreeVadRafRef = useRef<number>(0);
  const runHandsFreeVadRef = useRef<() => void>(() => {});
  const handsFreeRecorderRef = useRef<MediaRecorder | null>(null);
  const handsFreeChunksRef = useRef<BlobPart[]>([]);
  const handsFreeSpeechStartedAtRef = useRef<number>(0);
  const handsFreeLastVoiceAtRef = useRef<number>(0);
  const handsFreeNoiseFloorRef = useRef<number>(HANDS_FREE_INITIAL_NOISE_FLOOR);
  const handsFreeVoiceActiveSinceRef = useRef<number>(0);
  const handsFreeMeterUpdatedAtRef = useRef<number>(0);
  const handsFreeAudioRef = useRef<HTMLAudioElement | null>(null);
  const handsFreeAudioUrlRef = useRef<string | null>(null);
  const handsFreePlaybackResolveRef = useRef<(() => void) | null>(null);
  const handsFreePlaybackInterruptedRef = useRef(false);
  const handsFreeWakeLockRef = useRef<WakeLockSentinelLike | null>(null);
  const handsFreeDeltaBufferRef = useRef("");
  const handsFreeSpeechQueueRef = useRef<string[]>([]);
  const handsFreeSpeechWorkerRunningRef = useRef(false);
  const handsFreeTurnCompleteRef = useRef(false);
  const handsFreeStreamedAnyRef = useRef(false);
  const handsFreeSpokenCharsRef = useRef(0);
  const handsFreeSpeechTruncatedRef = useRef(false);
  const [handsFreeEnabled, setHandsFreeEnabled] = useState(false);
  const handsFreeEnabledRef = useRef(false);
  const [handsFreeState, setHandsFreeState] = useState<HandsFreeState>("off");
  const handsFreeStateRef = useRef<HandsFreeState>("off");
  const [handsFreeMicStats, setHandsFreeMicStats] = useState<HandsFreeMicStats>(
    DEFAULT_HANDS_FREE_MIC_STATS,
  );
  // Exposed to the main metrics-sync effect so it can refit the terminal
  // the moment `isActive` flips back to true (display:none → display:flex
  // collapses the host's box, so ResizeObserver never fires on return).
  const syncMetricsRef = useRef<(() => void) | null>(null);
  const [searchParams] = useSearchParams();
  // Lazy-init: the missing-token check happens at construction so the effect
  // body doesn't have to setState (React 19's set-state-in-effect rule).
  const [banner, setBanner] = useState<string | null>(() =>
    typeof window !== "undefined" && !window.__HERMES_SESSION_TOKEN__
      ? "Session token unavailable. Open this page through `hermes dashboard`, not directly."
      : null,
  );
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Raw state for the mobile side-sheet + a derived value that force-
  // closes whenever the chat tab isn't active.  The *derived* value is
  // what side-effects (body-scroll lock, keydown listener, portal render)
  // key on — that way switching to another tab triggers the effect's
  // cleanup, releasing the scroll-lock on /sessions etc.  Returning to
  // /chat re-runs the effect (derived flips back to true) and re-locks.
  // Keying on the raw state would leak the body.overflow="hidden" across
  // tabs because the dep wouldn't change on tab switch.
  const [mobilePanelOpenRaw, setMobilePanelOpenRaw] = useState(false);
  const mobilePanelOpen = isActive && mobilePanelOpenRaw;
  const { setEnd } = usePageHeader();
  const { t } = useI18n();
  const closeMobilePanel = useCallback(() => setMobilePanelOpenRaw(false), []);
  const modelToolsLabel = useMemo(
    () => `${t.app.modelToolsSheetTitle} ${t.app.modelToolsSheetSubtitle}`,
    [t.app.modelToolsSheetSubtitle, t.app.modelToolsSheetTitle],
  );
  const [portalRoot] = useState<HTMLElement | null>(() =>
    typeof document !== "undefined" ? document.body : null,
  );
  const [narrow, setNarrow] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(max-width: 1023px)").matches
      : false,
  );

  const resumeRef = useRef<string | null>(searchParams.get("resume"));
  const channel = useMemo(() => generateChannelId(), []);

  const setHandsFreeEnabledValue = useCallback((value: boolean) => {
    handsFreeEnabledRef.current = value;
    setHandsFreeEnabled(value);
  }, []);

  const setHandsFreeStateValue = useCallback((value: HandsFreeState) => {
    handsFreeStateRef.current = value;
    setHandsFreeState(value);
  }, []);

  useEffect(() => {
    const mql = window.matchMedia("(max-width: 1023px)");
    const sync = () => setNarrow(mql.matches);
    sync();
    mql.addEventListener("change", sync);
    return () => mql.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    if (!mobilePanelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeMobilePanel();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [mobilePanelOpen, closeMobilePanel]);

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMobilePanelOpenRaw(false);
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    // When hidden (non-chat tab) we must not register the header button —
    // another page owns the header's end slot at that point.
    if (!isActive) {
      setEnd(null);
      return;
    }
    if (!narrow) {
      setEnd(null);
      return;
    }
    setEnd(
      <Button
        ghost
        onClick={() => setMobilePanelOpenRaw(true)}
        aria-expanded={mobilePanelOpen}
        aria-controls="chat-side-panel"
        className={cn(
          "shrink-0 rounded border border-current/20",
          "px-2 py-1 text-[0.65rem] font-medium tracking-wide normal-case",
          "text-midground/80 hover:text-midground hover:bg-midground/5",
        )}
      >
        <span className="inline-flex items-center gap-1.5">
          <PanelRight className="h-3 w-3 shrink-0" />
          {modelToolsLabel}
        </span>
      </Button>,
    );
    return () => setEnd(null);
  }, [isActive, narrow, mobilePanelOpen, modelToolsLabel, setEnd]);

  const handleCopyLast = () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Send the slash as a burst, wait long enough for Ink's tokenizer to
    // emit a keypress event for each character (not coalesce them into a
    // paste), then send Return as its own event.  The timing here is
    // empirical — 100ms is safely past Node's default stdin coalescing
    // window and well inside UI responsiveness.
    ws.send("/copy");
    setTimeout(() => {
      const s = wsRef.current;
      if (s && s.readyState === WebSocket.OPEN) s.send("\r");
    }, 100);
    setCopyState("copied");
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
    copyResetRef.current = setTimeout(() => setCopyState("idle"), 1500);
    termRef.current?.focus();
  };

  const restoreTerminalInput = useCallback(() => {
    requestAnimationFrame(() => {
      syncMetricsRef.current?.();
      requestAnimationFrame(() => {
        syncMetricsRef.current?.();
        termRef.current?.focus();
      });
    });
  }, []);

  const showVoiceBanner = useCallback((message: string) => {
    setBanner(message);
    setRecording(false);
    setTranscribingVoice(false);
    restoreTerminalInput();
  }, [restoreTerminalInput]);

  const injectTranscript = useCallback((text: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showVoiceBanner("Chat websocket is not connected.");
      return false;
    }
    ws.send(text);
    setTimeout(() => {
      const s = wsRef.current;
      if (s && s.readyState === WebSocket.OPEN) s.send("\r");
    }, 100);
    termRef.current?.focus();
    return true;
  }, [showVoiceBanner]);

  const stopBrowserRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    if (recorder.state !== "inactive") recorder.stop();
  }, []);

  const startBrowserRecording = useCallback(async () => {
    if (!window.__HERMES_SESSION_TOKEN__) {
      showVoiceBanner("Session token unavailable. Reload the dashboard.");
      return;
    }
    if (!window.isSecureContext) {
      showVoiceBanner("Browser microphone requires HTTPS or localhost.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      showVoiceBanner("Browser microphone recording is unavailable in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = preferredAudioMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderChunksRef.current = [];
      recorderRef.current = recorder;
      recorderStreamRef.current = stream;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recorderChunksRef.current.push(event.data);
      };

      recorder.onerror = () => {
        showVoiceBanner("Browser microphone recording failed.");
      };

      recorder.onstop = () => {
        const chunks = recorderChunksRef.current;
        const token = window.__HERMES_SESSION_TOKEN__;
        recorderStreamRef.current?.getTracks().forEach((track) => track.stop());
        recorderStreamRef.current = null;
        recorderRef.current = null;
        setRecording(false);

        if (!token || chunks.length === 0) return;
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        setTranscribingVoice(true);
        transcribeBrowserAudio(blob, token)
          .then((transcript) => {
            const clean = transcript.trim();
            if (clean) injectTranscript(clean);
          })
          .catch((err: Error) => {
            showVoiceBanner(err.message || "Voice transcription failed.");
          })
          .finally(() => {
            setTranscribingVoice(false);
            restoreTerminalInput();
          });
      };

      recorder.start();
      setBanner(null);
      setRecording(true);
      termRef.current?.focus();
    } catch (err) {
      const message = err instanceof Error ? err.message : "microphone permission denied";
      recorderStreamRef.current?.getTracks().forEach((track) => track.stop());
      recorderStreamRef.current = null;
      recorderRef.current = null;
      showVoiceBanner(`Microphone unavailable: ${message}`);
    }
  }, [injectTranscript, restoreTerminalInput, showVoiceBanner]);

  const handleVoiceButton = useCallback(() => {
    if (recording) {
      stopBrowserRecording();
      return;
    }
    if (!transcribingVoice) void startBrowserRecording();
  }, [recording, startBrowserRecording, stopBrowserRecording, transcribingVoice]);

  const stopHandsFreePlayback = useCallback((interrupted = false) => {
    if (interrupted) handsFreePlaybackInterruptedRef.current = true;
    const resolve = handsFreePlaybackResolveRef.current;
    handsFreePlaybackResolveRef.current = null;
    if (resolve) resolve();

    const audio = handsFreeAudioRef.current;
    if (audio) {
      clearAudioSource(audio);
    }

    if (handsFreeAudioUrlRef.current) {
      URL.revokeObjectURL(handsFreeAudioUrlRef.current);
      handsFreeAudioUrlRef.current = null;
    }
  }, []);

  const resetHandsFreeSpeechPipeline = useCallback(() => {
    handsFreeDeltaBufferRef.current = "";
    handsFreeSpeechQueueRef.current = [];
    handsFreeTurnCompleteRef.current = false;
    handsFreeStreamedAnyRef.current = false;
    handsFreePlaybackInterruptedRef.current = false;
    handsFreeSpokenCharsRef.current = 0;
    handsFreeSpeechTruncatedRef.current = false;
    stopHandsFreePlayback();
  }, [stopHandsFreePlayback]);

  const releaseHandsFreeWakeLock = useCallback(() => {
    const sentinel = handsFreeWakeLockRef.current;
    handsFreeWakeLockRef.current = null;
    if (sentinel) void sentinel.release().catch(() => {});
  }, []);

  const requestHandsFreeWakeLock = useCallback(async () => {
    if (typeof document === "undefined" || document.visibilityState !== "visible") {
      return;
    }
    const wakeLock = (navigator as WakeLockNavigator).wakeLock;
    if (!wakeLock || handsFreeWakeLockRef.current) {
      return;
    }
    try {
      handsFreeWakeLockRef.current = await wakeLock.request("screen");
    } catch {
      // Wake lock is unavailable on some browsers or power modes. Voice still works.
    }
  }, []);

  const cleanupHandsFreeResources = useCallback(() => {
    if (handsFreeVadRafRef.current) {
      cancelAnimationFrame(handsFreeVadRafRef.current);
      handsFreeVadRafRef.current = 0;
    }

    const recorder = handsFreeRecorderRef.current;
    if (recorder) {
      recorder.ondataavailable = null;
      recorder.onerror = null;
      recorder.onstop = null;
      if (recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch {
          // Ignore recorder shutdown races during mode teardown.
        }
      }
    }
    handsFreeRecorderRef.current = null;
    handsFreeChunksRef.current = [];
    handsFreeDeltaBufferRef.current = "";
    handsFreeSpeechQueueRef.current = [];
    handsFreeTurnCompleteRef.current = false;
    handsFreeStreamedAnyRef.current = false;
    handsFreeSpokenCharsRef.current = 0;
    handsFreeSpeechTruncatedRef.current = false;

    handsFreeStreamRef.current?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });
    handsFreeStreamRef.current = null;
    handsFreeAnalyserRef.current = null;
    handsFreeSamplesRef.current = null;

    const audioContext = handsFreeAudioContextRef.current;
    handsFreeAudioContextRef.current = null;
    if (audioContext && audioContext.state !== "closed") {
      void audioContext.close().catch(() => {});
    }

    stopHandsFreePlayback();
    releaseHandsFreeWakeLock();
  }, [releaseHandsFreeWakeLock, stopHandsFreePlayback]);

  const disableHandsFree = useCallback((message?: string) => {
    setHandsFreeEnabledValue(false);
    cleanupHandsFreeResources();
    setHandsFreeStateValue("off");
    if (message) setBanner(message);
    restoreTerminalInput();
  }, [
    cleanupHandsFreeResources,
    restoreTerminalInput,
    setHandsFreeEnabledValue,
    setHandsFreeStateValue,
  ]);

  const primeHandsFreeAudio = useCallback(async () => {
    const audio = handsFreeAudioRef.current ?? new Audio();
    handsFreeAudioRef.current = audio;
    setAudioSource(audio, HANDS_FREE_SILENT_WAV, 0);
    try {
      await audio.play();
      audio.pause();
      rewindAudio(audio);
    } catch {
      // Browsers may still allow later playback after the user gesture that
      // enabled hands-free mode; surface a real error only if TTS playback fails.
    } finally {
      clearAudioSource(audio);
      audio.volume = 1;
    }
  }, []);

  const transcribeHandsFreeBlob = useCallback(async (blob: Blob) => {
    const token = window.__HERMES_SESSION_TOKEN__;
    if (!token || !handsFreeEnabledRef.current) {
      return;
    }

    setHandsFreeStateValue("transcribing");
    try {
      const transcript = (await transcribeBrowserAudio(blob, token)).trim();
      if (!handsFreeEnabledRef.current) {
        return;
      }
      if (!transcript) {
        setHandsFreeStateValue("listening");
        return;
      }
      if (isHandsFreeDisableCommand(transcript)) {
        disableHandsFree("Hands-free voice mode stopped.");
        return;
      }
      if (isHandsFreeStopCommand(transcript)) {
        resetHandsFreeSpeechPipeline();
        setHandsFreeStateValue("listening");
        restoreTerminalInput();
        return;
      }
      resetHandsFreeSpeechPipeline();
      if (injectTranscript(transcript)) {
        setHandsFreeStateValue("thinking");
      } else {
        setHandsFreeStateValue("listening");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Voice transcription failed.";
      if (handsFreeEnabledRef.current) {
        setBanner(message);
        setHandsFreeStateValue("listening");
      }
    } finally {
      restoreTerminalInput();
    }
  }, [
    disableHandsFree,
    injectTranscript,
    resetHandsFreeSpeechPipeline,
    restoreTerminalInput,
    setHandsFreeStateValue,
  ]);

  const stopHandsFreeRecording = useCallback(() => {
    const recorder = handsFreeRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }, []);

  const startHandsFreeRecording = useCallback(() => {
    const stream = handsFreeStreamRef.current;
    if (!stream || handsFreeRecorderRef.current) {
      return;
    }

    const mimeType = preferredAudioMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const now = performance.now();
    handsFreeChunksRef.current = [];
    handsFreeRecorderRef.current = recorder;
    handsFreeSpeechStartedAtRef.current = now;
    handsFreeLastVoiceAtRef.current = now;
    handsFreeVoiceActiveSinceRef.current = 0;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) handsFreeChunksRef.current.push(event.data);
    };

    recorder.onerror = () => {
      if (handsFreeEnabledRef.current) {
        setBanner("Browser microphone recording failed.");
        setHandsFreeStateValue("listening");
      }
      handsFreeRecorderRef.current = null;
      handsFreeChunksRef.current = [];
    };

    recorder.onstop = () => {
      const chunks = handsFreeChunksRef.current;
      const type = recorder.mimeType || "audio/webm";
      handsFreeRecorderRef.current = null;
      handsFreeChunksRef.current = [];
      if (!handsFreeEnabledRef.current) {
        return;
      }
      if (chunks.length === 0) {
        setHandsFreeStateValue("listening");
        return;
      }
      void transcribeHandsFreeBlob(new Blob(chunks, { type }));
    };

    recorder.start();
    setHandsFreeStateValue("recording");
  }, [setHandsFreeStateValue, transcribeHandsFreeBlob]);

  const runHandsFreeVad = useCallback(() => {
    if (!handsFreeEnabledRef.current) {
      return;
    }

    const audioContext = handsFreeAudioContextRef.current;
    if (audioContext?.state === "suspended") {
      void audioContext.resume().catch(() => {});
    }

    const analyser = handsFreeAnalyserRef.current;
    const samples = handsFreeSamplesRef.current;
    const state = handsFreeStateRef.current;
    // Half-duplex MVP: only open a recording while the user turn is expected.
    // During thinking/speaking, ambient speech must not interrupt Hermes.
    if (
      analyser &&
      samples &&
      (state === "listening" ||
        state === "recording")
    ) {
      analyser.getByteTimeDomainData(samples);
      const rms = rmsFromTimeDomain(samples);
      const now = performance.now();
      const speechThreshold = handsFreeSpeechThreshold(handsFreeNoiseFloorRef.current);
      const silenceThreshold = handsFreeSilenceThreshold(handsFreeNoiseFloorRef.current);
      if (now - handsFreeMeterUpdatedAtRef.current >= HANDS_FREE_METER_UPDATE_MS) {
        handsFreeMeterUpdatedAtRef.current = now;
        setHandsFreeMicStats({
          level: rms,
          noise: handsFreeNoiseFloorRef.current,
          speaking: rms >= speechThreshold,
          threshold: speechThreshold,
        });
      }

      if (state === "listening") {
        if (rms < speechThreshold) {
          handsFreeVoiceActiveSinceRef.current = 0;
          handsFreeNoiseFloorRef.current =
            handsFreeNoiseFloorRef.current * (1 - HANDS_FREE_NOISE_ALPHA) + rms * HANDS_FREE_NOISE_ALPHA;
        } else {
          startHandsFreeRecording();
        }
      } else if (state === "recording") {
        if (rms >= silenceThreshold) {
          handsFreeLastVoiceAtRef.current = now;
        }

        const elapsed = now - handsFreeSpeechStartedAtRef.current;
        const silentFor = now - handsFreeLastVoiceAtRef.current;
        if (
          elapsed >= HANDS_FREE_MAX_UTTERANCE_MS ||
          (elapsed >= HANDS_FREE_MIN_RECORDING_MS && silentFor >= HANDS_FREE_SILENCE_MS)
        ) {
          stopHandsFreeRecording();
        }
      }
    }

    handsFreeVadRafRef.current = requestAnimationFrame(() => runHandsFreeVadRef.current());
  }, [startHandsFreeRecording, stopHandsFreeRecording]);

  useEffect(() => {
    runHandsFreeVadRef.current = runHandsFreeVad;
  }, [runHandsFreeVad]);

  const startHandsFreeMode = useCallback(async () => {
    if (!window.__HERMES_SESSION_TOKEN__) {
      showVoiceBanner("Session token unavailable. Reload the dashboard.");
      return;
    }
    if (!window.isSecureContext) {
      showVoiceBanner("Browser microphone requires HTTPS or localhost.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      showVoiceBanner("Browser microphone recording is unavailable in this browser.");
      return;
    }
    if (recording || transcribingVoice) {
      showVoiceBanner("Stop the current voice recording before enabling hands-free mode.");
      return;
    }

    try {
      await primeHandsFreeAudio();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) {
        stream.getTracks().forEach((track) => track.stop());
        showVoiceBanner("Browser audio analysis is unavailable in this browser.");
        return;
      }
      const audioContext = new AudioContextCtor();
      await audioContext.resume();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      stream.getAudioTracks().forEach((track) => {
        track.onended = () => {
          if (handsFreeEnabledRef.current) {
            disableHandsFree("Microphone stream ended. Restart hands-free voice mode.");
          }
        };
      });

      cleanupHandsFreeResources();
      handsFreeStreamRef.current = stream;
      handsFreeAudioContextRef.current = audioContext;
      handsFreeAnalyserRef.current = analyser;
      handsFreeSamplesRef.current = new Uint8Array(analyser.fftSize);
      handsFreeNoiseFloorRef.current = HANDS_FREE_INITIAL_NOISE_FLOOR;
      handsFreeVoiceActiveSinceRef.current = 0;
      handsFreeMeterUpdatedAtRef.current = 0;
      setHandsFreeMicStats({
        ...DEFAULT_HANDS_FREE_MIC_STATS,
        threshold: handsFreeSpeechThreshold(HANDS_FREE_INITIAL_NOISE_FLOOR),
      });
      setBanner(null);
      setHandsFreeEnabledValue(true);
      setHandsFreeStateValue("listening");
      void requestHandsFreeWakeLock();
      handsFreeVadRafRef.current = requestAnimationFrame(() => runHandsFreeVadRef.current());
      termRef.current?.focus();
    } catch (err) {
      cleanupHandsFreeResources();
      const message = err instanceof Error ? err.message : "microphone permission denied";
      showVoiceBanner(`Microphone unavailable: ${message}`);
    }
  }, [
    cleanupHandsFreeResources,
    disableHandsFree,
    primeHandsFreeAudio,
    requestHandsFreeWakeLock,
    recording,
    setHandsFreeEnabledValue,
    setHandsFreeStateValue,
    showVoiceBanner,
    transcribingVoice,
  ]);

  const playHandsFreeSpeechChunk = useCallback(async (text: string, token: string) => {
    const blob = await synthesizeBrowserSpeech(text, token);
    if (!handsFreeEnabledRef.current || handsFreePlaybackInterruptedRef.current) {
      return;
    }

    const audio = handsFreeAudioRef.current ?? new Audio();
    handsFreeAudioRef.current = audio;
    const url = URL.createObjectURL(blob);
    handsFreeAudioUrlRef.current = url;
    setAudioSource(audio, url, 1);

    await new Promise<void>((resolve, reject) => {
      handsFreePlaybackResolveRef.current = resolve;
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error("Browser audio playback failed."));
      const playPromise = audio.play();
      if (playPromise) {
        playPromise.catch(reject);
      }
    });
  }, []);

  const drainHandsFreeSpeechQueue = useCallback(async () => {
    const token = window.__HERMES_SESSION_TOKEN__;
    if (!token || handsFreeSpeechWorkerRunningRef.current) {
      return;
    }

    handsFreeSpeechWorkerRunningRef.current = true;
    try {
      while (
        handsFreeEnabledRef.current &&
        !handsFreePlaybackInterruptedRef.current &&
        handsFreeSpeechQueueRef.current.length > 0
      ) {
        const chunk = handsFreeSpeechQueueRef.current.shift();
        if (!chunk) break;
        setHandsFreeStateValue("speaking");
        await playHandsFreeSpeechChunk(chunk, token);
        stopHandsFreePlayback();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Voice playback failed.";
      if (handsFreeEnabledRef.current && !handsFreePlaybackInterruptedRef.current) {
        setBanner(message);
      }
    } finally {
      handsFreeSpeechWorkerRunningRef.current = false;
      stopHandsFreePlayback();

      if (
        handsFreeEnabledRef.current &&
        !handsFreePlaybackInterruptedRef.current &&
        handsFreeSpeechQueueRef.current.length > 0
      ) {
        void drainHandsFreeSpeechQueue();
      } else if (
        handsFreeEnabledRef.current &&
        !handsFreePlaybackInterruptedRef.current &&
        handsFreeTurnCompleteRef.current &&
        !handsFreeDeltaBufferRef.current.trim()
      ) {
        setHandsFreeStateValue("listening");
        restoreTerminalInput();
      } else if (
        handsFreeEnabledRef.current &&
        !handsFreePlaybackInterruptedRef.current &&
        handsFreeStateRef.current === "speaking"
      ) {
        setHandsFreeStateValue("thinking");
      }
    }
  }, [
    playHandsFreeSpeechChunk,
    restoreTerminalInput,
    setHandsFreeStateValue,
    stopHandsFreePlayback,
  ]);

  const queueHandsFreeSpeechChunks = useCallback((chunks: string[]) => {
    const cleanChunks: string[] = [];
    for (const chunk of chunks) {
      if (handsFreeSpeechTruncatedRef.current) break;
      const clean = sanitizeHandsFreeSpeechText(chunk);
      if (!clean) continue;

      if (clean.includes(HANDS_FREE_TRUNCATION_NOTICE)) {
        cleanChunks.push(clean);
        handsFreeSpokenCharsRef.current = HANDS_FREE_MAX_SPOKEN_CHARS;
        handsFreeSpeechTruncatedRef.current = true;
        break;
      }

      const remaining = HANDS_FREE_MAX_SPOKEN_CHARS - handsFreeSpokenCharsRef.current;
      if (clean.length <= remaining) {
        cleanChunks.push(clean);
        handsFreeSpokenCharsRef.current += clean.length;
        continue;
      }

      if (remaining > 80) {
        const slice = clean.slice(0, remaining);
        const boundary = Math.max(
          slice.lastIndexOf("."),
          slice.lastIndexOf("!"),
          slice.lastIndexOf("?"),
          slice.lastIndexOf(","),
          slice.lastIndexOf(";"),
          slice.lastIndexOf(":"),
          slice.lastIndexOf(" "),
        );
        const trimmed = slice.slice(0, boundary > remaining * 0.6 ? boundary + 1 : remaining).trim();
        cleanChunks.push(`${trimmed} ${HANDS_FREE_TRUNCATION_NOTICE}`);
      } else {
        cleanChunks.push(HANDS_FREE_TRUNCATION_NOTICE);
      }
      handsFreeSpokenCharsRef.current = HANDS_FREE_MAX_SPOKEN_CHARS;
      handsFreeSpeechTruncatedRef.current = true;
      break;
    }
    if (cleanChunks.length === 0) return;
    handsFreeSpeechQueueRef.current.push(...cleanChunks);
    void drainHandsFreeSpeechQueue();
  }, [drainHandsFreeSpeechQueue]);

  const flushHandsFreeDeltaBuffer = useCallback((force: boolean) => {
    const { chunks, rest } = takeReadyHandsFreeSpeechChunks(
      handsFreeDeltaBufferRef.current,
      force,
    );
    handsFreeDeltaBufferRef.current = rest;
    queueHandsFreeSpeechChunks(chunks);
  }, [queueHandsFreeSpeechChunks]);

  const handleHandsFreeAssistantStart = useCallback(() => {
    if (!handsFreeEnabledRef.current) return;
    handsFreeDeltaBufferRef.current = "";
    handsFreeSpeechQueueRef.current = [];
    handsFreeTurnCompleteRef.current = false;
    handsFreeStreamedAnyRef.current = false;
    handsFreePlaybackInterruptedRef.current = false;
    handsFreeSpokenCharsRef.current = 0;
    handsFreeSpeechTruncatedRef.current = false;
  }, []);

  const handleHandsFreeAssistantDelta = useCallback((text: string) => {
    if (!handsFreeEnabledRef.current) {
      return;
    }
    if (handsFreeStateRef.current !== "thinking" && handsFreeStateRef.current !== "speaking") {
      return;
    }
    if (!text) {
      return;
    }

    handsFreeStreamedAnyRef.current = true;
    handsFreeDeltaBufferRef.current += text;
    flushHandsFreeDeltaBuffer(false);
  }, [flushHandsFreeDeltaBuffer]);

  const handleHandsFreeAssistantComplete = useCallback((text: string) => {
    if (!handsFreeEnabledRef.current) {
      return;
    }
    if (handsFreeStateRef.current !== "thinking" && handsFreeStateRef.current !== "speaking") {
      return;
    }
    handsFreeTurnCompleteRef.current = true;
    if (handsFreeStreamedAnyRef.current) {
      flushHandsFreeDeltaBuffer(true);
      void drainHandsFreeSpeechQueue();
      return;
    }
    const fallbackChunks = splitHandsFreeSpeech(text);
    if (fallbackChunks.length > 0) {
      queueHandsFreeSpeechChunks(fallbackChunks);
    } else {
      setHandsFreeStateValue("listening");
      restoreTerminalInput();
    }
  }, [
    drainHandsFreeSpeechQueue,
    flushHandsFreeDeltaBuffer,
    queueHandsFreeSpeechChunks,
    restoreTerminalInput,
    setHandsFreeStateValue,
  ]);

  const handleHandsFreeButton = useCallback(() => {
    if (handsFreeEnabledRef.current) {
      disableHandsFree();
      return;
    }
    void startHandsFreeMode();
  }, [disableHandsFree, startHandsFreeMode]);

  useEffect(() => {
    const token = window.__HERMES_SESSION_TOKEN__;
    if (!token || !channel) {
      return;
    }

    const ws = new WebSocket(buildEventsWsUrl(token, channel));
    let unmounting = false;

    ws.addEventListener("error", () => {
      if (handsFreeEnabledRef.current && !unmounting) {
        setBanner("Hands-free events feed disconnected.");
      }
    });

    ws.addEventListener("close", (ev) => {
      if (unmounting || !handsFreeEnabledRef.current || ev.code === 1000) {
        return;
      }
      setBanner("Hands-free events feed disconnected.");
    });

    ws.addEventListener("message", (ev) => {
      let frame: RpcEventFrame;
      try {
        frame = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (frame.method !== "event" || !frame.params?.type) {
        return;
      }
      const eventType = frame.params.type;
      if (eventType === "message.start") {
        ptyBusyRef.current = true;
      } else if (
        eventType === "message.complete" ||
        eventType === "approval.request" ||
        eventType === "clarify.request" ||
        eventType === "error"
      ) {
        ptyBusyRef.current = false;
      }
      if (
        handsFreeEnabledRef.current &&
        handsFreeStateRef.current === "thinking" &&
        (eventType === "approval.request" ||
          eventType === "clarify.request" ||
          eventType === "error")
      ) {
        setHandsFreeStateValue("listening");
      }
      if (eventType === "message.start") {
        handleHandsFreeAssistantStart();
        return;
      }
      if (eventType === "message.delta") {
        handleHandsFreeAssistantDelta(frame.params.payload?.text || "");
        return;
      }
      if (eventType !== "message.complete") {
        return;
      }
      const payload = frame.params.payload;
      const text = payload?.text || payload?.rendered || "";
      if (text) handleHandsFreeAssistantComplete(text);
    });

    return () => {
      unmounting = true;
      ptyBusyRef.current = false;
      ws.close();
    };
  }, [
    channel,
    handleHandsFreeAssistantDelta,
    handleHandsFreeAssistantComplete,
    handleHandsFreeAssistantStart,
    setHandsFreeStateValue,
  ]);

  useEffect(() => {
    return () => {
      handsFreeEnabledRef.current = false;
      cleanupHandsFreeResources();
    };
  }, [cleanupHandsFreeResources]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!handsFreeEnabledRef.current) return;
      if (document.visibilityState === "visible") {
        void requestHandsFreeWakeLock();
        const audioContext = handsFreeAudioContextRef.current;
        if (audioContext?.state === "suspended") {
          void audioContext.resume().catch(() => {});
        }
        return;
      }

      releaseHandsFreeWakeLock();
      setBanner("Keep the dashboard visible for reliable hands-free voice mode.");
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, [releaseHandsFreeWakeLock, requestHandsFreeWakeLock]);

  useEffect(() => {
    const handlePageHide = () => {
      if (!handsFreeEnabledRef.current) return;
      disableHandsFree("Hands-free voice mode paused by the browser. Restart it when the page is active.");
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [disableHandsFree]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const token = window.__HERMES_SESSION_TOKEN__;
    // Banner already initialised above; just bail before wiring xterm/WS.
    if (!token) {
      return;
    }

    const tierW0 = terminalTierWidthPx(host);
    const term = new Terminal({
      allowProposedApi: true,
      cursorBlink: true,
      fontFamily:
        "'JetBrains Mono', 'Cascadia Mono', 'Fira Code', 'MesloLGS NF', 'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace",
      fontSize: terminalFontSizeForWidth(tierW0),
      lineHeight: terminalLineHeightForWidth(tierW0),
      letterSpacing: 0,
      fontWeight: "400",
      fontWeightBold: "700",
      macOptionIsMeta: true,
      scrollback: 0,
      theme: TERMINAL_THEME,
    });
    termRef.current = term;

    // --- Clipboard integration ---------------------------------------
    //
    // Three independent paths all route to the system clipboard:
    //
    //   1. **Selection → Ctrl+C (or Cmd+C on macOS).**  Ink's own handler
    //      in useInputHandlers.ts turns Ctrl+C into a copy when the
    //      terminal has a selection, then emits an OSC 52 escape.  Our
    //      OSC 52 handler below decodes that escape and writes to the
    //      browser clipboard — so the flow works just like it does in
    //      `hermes --tui`.
    //
    //   2. **Ctrl/Cmd+Shift+C.**  Belt-and-suspenders shortcut that
    //      operates directly on xterm's selection, useful if the TUI
    //      ever stops listening (e.g. overlays / pickers) or if the user
    //      has selected with the mouse outside of Ink's selection model.
    //
    //   3. **Ctrl/Cmd+Shift+V.**  Reads the system clipboard and feeds
    //      it to the terminal as keyboard input.  xterm's paste() wraps
    //      it with bracketed-paste if the host has that mode enabled.
    //
    // OSC 52 reads (terminal asking to read the clipboard) are not
    // supported — that would let any content the TUI renders exfiltrate
    // the user's clipboard.
    term.parser.registerOscHandler(52, (data) => {
      // Format: "<targets>;<base64 | '?'>"
      const semi = data.indexOf(";");
      if (semi < 0) return false;
      const payload = data.slice(semi + 1);
      if (payload === "?" || payload === "") return false; // read/clear — ignore
      try {
        const binary = atob(payload);
        const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
        const text = new TextDecoder("utf-8").decode(bytes);
        copyText(text);
      } catch {
        console.warn("[dashboard clipboard] malformed OSC 52 payload");
      }
      return true;
    });

    const isMac =
      typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);

    term.attachCustomKeyEventHandler((ev) => {
      if (ev.type !== "keydown") return true;

      // Copy: Cmd+C on macOS, Ctrl+Shift+C on other platforms. Bare Ctrl+C
      // is never forwarded as a raw terminal signal in the dashboard: when
      // Hermes is busy we route it through /stop, otherwise we swallow it so
      // an idle chat tab cannot accidentally close its embedded PTY.
      // Paste: Cmd+V on macOS, Ctrl+Shift+V on others.  On non-secure
      // dashboard origins (for example http://100.x.y.z over Tailscale),
      // browsers block navigator.clipboard.readText(); in that case we let
      // xterm's native paste event handle the key instead of swallowing it.
      const copyModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;
      const pasteModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;
      const bareCtrlC =
        ev.ctrlKey &&
        !ev.shiftKey &&
        !ev.altKey &&
        !ev.metaKey &&
        ev.key.toLowerCase() === "c";

      if (copyModifier && ev.key.toLowerCase() === "c") {
        const sel = term.getSelection();
        if (sel) {
          // Direct writeText inside the keydown handler preserves the user
          // gesture — async round-trips through OSC 52 can lose activation
          // and fail with "Document is not focused".
          copyText(sel);
          // Clear xterm.js's highlight after copy (matches gnome-terminal).
          term.clearSelection();
          ev.preventDefault();
          return false;
        }
      }

      if (bareCtrlC) {
        const sel = term.getSelection();
        if (sel) {
          copyText(sel);
          term.clearSelection();
        } else if (ptyBusyRef.current) {
          const s = wsRef.current;
          if (s && s.readyState === WebSocket.OPEN) {
            s.send("/stop");
            setTimeout(() => {
              const live = wsRef.current;
              if (live && live.readyState === WebSocket.OPEN) live.send("\r");
            }, 100);
          }
        }
        ev.preventDefault();
        return false;
      }

      if (pasteModifier && ev.key.toLowerCase() === "v") {
        if (!clipboardCanReadText()) return true;
        navigator.clipboard
          .readText()
          .then((text) => {
            if (text) term.paste(text);
          })
          .catch((err) => {
            console.warn("[dashboard clipboard] paste failed:", err.message);
          });
        ev.preventDefault();
        return false;
      }

      return true;
    });

    const fit = new FitAddon();
    fitRef.current = fit;
    term.loadAddon(fit);

    const unicode11 = new Unicode11Addon();
    term.loadAddon(unicode11);
    term.unicode.activeVersion = "11";

    term.loadAddon(new WebLinksAddon());

    term.open(host);

    // WebGL draws from a texture atlas sized with device pixels. On phones and
    // in DevTools device mode that often produces *visually* much larger cells
    // than `fontSize` suggests — users see "huge" text even at 7–9px settings.
    // The canvas/DOM renderer tracks `fontSize` faithfully; use it for narrow
    // hosts.  Wide layouts still get WebGL for crisp box-drawing.
    const useWebgl = terminalTierWidthPx(host) >= 768;
    if (useWebgl) {
      try {
        const webgl = new WebglAddon();
        webgl.onContextLoss(() => webgl.dispose());
        term.loadAddon(webgl);
      } catch (err) {
        console.warn(
          "[hermes-chat] WebGL renderer unavailable; falling back to default",
          err,
        );
      }
    }

    // Initial fit + resize observer.  fit.fit() reads the container's
    // current bounding box and resizes the terminal grid to match.
    //
    // The subtle bit: the dashboard has CSS transitions on the container
    // (backdrop fade-in, rounded corners settling as fonts load).  If we
    // call fit() at mount time, the bounding box we measure is often 1-2
    // cell widths off from the final size.  ResizeObserver *does* fire
    // when the container settles, but if the pixel delta happens to be
    // smaller than one cell's width, fit() computes the same integer
    // (cols, rows) as before and doesn't emit onResize — so the PTY
    // never learns the final size.  Users see truncated long lines until
    // they resize the browser window.
    //
    // We force one extra fit + explicit RESIZE send after two animation
    // frames.  rAF→rAF guarantees one layout commit between the two
    // callbacks, giving CSS transitions and font metrics time to finalize
    // before we take the authoritative measurement.
    let hostSyncRaf = 0;
    const scheduleHostSync = () => {
      if (hostSyncRaf) return;
      hostSyncRaf = requestAnimationFrame(() => {
        hostSyncRaf = 0;
        syncTerminalMetrics();
      });
    };

    let metricsDebounce: ReturnType<typeof setTimeout> | null = null;
    const syncTerminalMetrics = () => {
      // display:none hosts have clientWidth/Height = 0, which fit() turns
      // into a 1x1 terminal.  Skip entirely while hidden; the visibility
      // effect below runs another fit as soon as the tab is shown again.
      if (!host.isConnected || host.clientWidth <= 0 || host.clientHeight <= 0) {
        return;
      }
      const w = terminalTierWidthPx(host);
      const nextSize = terminalFontSizeForWidth(w);
      const nextLh = terminalLineHeightForWidth(w);
      const fontChanged =
        term.options.fontSize !== nextSize ||
        term.options.lineHeight !== nextLh;
      if (fontChanged) {
        term.options.fontSize = nextSize;
        term.options.lineHeight = nextLh;
      }
      try {
        fit.fit();
      } catch {
        return;
      }
      if (fontChanged && term.rows > 0) {
        try {
          term.refresh(0, term.rows - 1);
        } catch {
          /* ignore */
        }
      }
      if (
        fontChanged &&
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN
      ) {
        wsRef.current.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
      }
    };
    syncMetricsRef.current = syncTerminalMetrics;

    const scheduleSyncTerminalMetrics = () => {
      if (metricsDebounce) clearTimeout(metricsDebounce);
      metricsDebounce = setTimeout(() => {
        metricsDebounce = null;
        syncTerminalMetrics();
      }, 60);
    };

    const ro = new ResizeObserver(() => scheduleHostSync());
    ro.observe(host);

    window.addEventListener("resize", scheduleSyncTerminalMetrics);
    window.visualViewport?.addEventListener("resize", scheduleSyncTerminalMetrics);
    window.visualViewport?.addEventListener("scroll", scheduleSyncTerminalMetrics);
    scheduleHostSync();
    requestAnimationFrame(() => scheduleHostSync());

    // Double-rAF authoritative fit.  On the second frame the layout has
    // committed at least once since mount; fit.fit() then reads the
    // stable container size.  We always send a RESIZE escape afterwards
    // (even if fit's cols/rows didn't change, so the PTY has the same
    // dims registered as our JS state — prevents a drift where Ink
    // thinks the terminal is one col bigger than what's on screen).
    let settleRaf1 = 0;
    let settleRaf2 = 0;
    settleRaf1 = requestAnimationFrame(() => {
      settleRaf1 = 0;
      settleRaf2 = requestAnimationFrame(() => {
        settleRaf2 = 0;
        syncTerminalMetrics();
      });
    });

    // WebSocket
    const url = buildWsUrl(token, resumeRef.current, channel);
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    // Suppress banner/terminal side-effects when cleanup() calls `ws.close()`
    // (React StrictMode remount, route change) so we never write to a
    // disposed xterm or setState on an unmounted tree.
    let unmounting = false;

    ws.onopen = () => {
      setBanner(null);
      // Send the initial RESIZE immediately so Ink has *a* size to lay
      // out against on its first paint.  The double-rAF block above will
      // follow up with the authoritative measurement — at worst Ink
      // reflows once after the PTY boots, which is imperceptible.
      ws.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        term.write(ev.data);
      } else {
        term.write(new Uint8Array(ev.data as ArrayBuffer));
      }
    };

    ws.onclose = (ev) => {
      wsRef.current = null;
      if (unmounting) {
        return;
      }
      if (ev.code === 4401) {
        setBanner("Auth failed. Reload the page to refresh the session token.");
        return;
      }
      if (ev.code === 4403) {
        setBanner("Chat is only reachable from localhost.");
        return;
      }
      if (ev.code === 1011) {
        // Server already wrote an ANSI error frame.
        return;
      }
      term.write("\r\n\x1b[90m[session ended]\x1b[0m\r\n");
    };

    // Keystrokes + mouse events → PTY, with cell-level dedup for motion.
    //
    // Ink enables `\x1b[?1003h` (any-motion tracking), which asks the
    // terminal to report every mouse-move as an SGR mouse event even with
    // no button held.  xterm.js happily emits one report per pixel of
    // mouse motion; without deduping, a casual mouse-over floods Ink with
    // hundreds of redraw-triggering reports and the UI goes laggy
    // (scrolling stutters, clicks land on stale positions by the time
    // Ink finishes processing the motion backlog).
    //
    // We keep track of the last cell we reported a motion for.  Press,
    // release, and wheel events always pass through; motion events only
    // pass through if the cell changed.  Parsing is cheap — SGR reports
    // are short literal strings.
    // eslint-disable-next-line no-control-regex -- intentional ESC byte in xterm SGR mouse report parser
    const SGR_MOUSE_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/;
    let lastMotionCell = { col: -1, row: -1 };
    let lastMotionCb = -1;
    const onDataDisposable = term.onData((data) => {
      if (ws.readyState !== WebSocket.OPEN) return;

      const m = SGR_MOUSE_RE.exec(data);
      if (m) {
        const cb = parseInt(m[1], 10);
        const col = parseInt(m[2], 10);
        const row = parseInt(m[3], 10);
        const released = m[4] === "m";
        // Motion events have bit 0x20 (32) set in the button code.
        // Wheel events have bit 0x40 (64); always forward wheel.
        const isMotion = (cb & 0x20) !== 0 && (cb & 0x40) === 0;
        const isWheel = (cb & 0x40) !== 0;
        if (isMotion && !isWheel && !released) {
          if (
            col === lastMotionCell.col &&
            row === lastMotionCell.row &&
            cb === lastMotionCb
          ) {
            return; // same cell + same button state; skip redundant report
          }
          lastMotionCell = { col, row };
          lastMotionCb = cb;
        } else {
          // Non-motion event (press, release, wheel) — reset dedup state
          // so the next motion after this always reports.
          lastMotionCell = { col: -1, row: -1 };
          lastMotionCb = -1;
        }
      }

      ws.send(data);
    });

    const onResizeDisposable = term.onResize(({ cols, rows }) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(`\x1b[RESIZE:${cols};${rows}]`);
      }
    });

    term.focus();

    return () => {
      unmounting = true;
      syncMetricsRef.current = null;
      onDataDisposable.dispose();
      onResizeDisposable.dispose();
      if (metricsDebounce) clearTimeout(metricsDebounce);
      window.removeEventListener("resize", scheduleSyncTerminalMetrics);
      window.visualViewport?.removeEventListener(
        "resize",
        scheduleSyncTerminalMetrics,
      );
      window.visualViewport?.removeEventListener(
        "scroll",
        scheduleSyncTerminalMetrics,
      );
      ro.disconnect();
      if (hostSyncRaf) cancelAnimationFrame(hostSyncRaf);
      if (settleRaf1) cancelAnimationFrame(settleRaf1);
      if (settleRaf2) cancelAnimationFrame(settleRaf2);
      recorderStreamRef.current?.getTracks().forEach((track) => track.stop());
      recorderStreamRef.current = null;
      recorderRef.current = null;
      ws.close();
      wsRef.current = null;
      term.dispose();
      termRef.current = null;
      fitRef.current = null;
      if (copyResetRef.current) {
        clearTimeout(copyResetRef.current);
        copyResetRef.current = null;
      }
    };
  }, [channel]);

  // When the user returns to the chat tab (isActive: false → true), the
  // terminal host just transitioned from display:none to display:flex.
  // ResizeObserver won't fire on that kind of style-driven box change —
  // xterm thinks its grid is still whatever it was when the tab was
  // hidden (or 0×0, if it was hidden before first fit).  Force a refit
  // after two animation frames so layout has committed.
  //
  // Focus handling: we only steal focus back into the terminal when
  // nothing else inside ChatPage was holding it (typically the first
  // activation after mount, where document.activeElement is <body>; or
  // a return after the user had been typing in the terminal, where
  // focus was already on the xterm textarea before the tab got hidden
  // and has since fallen back to <body>).  If the user had clicked
  // into the sidebar (model picker, tool-call entry) before switching
  // tabs, we must not yank focus away from wherever they left it when
  // they come back — that's a surprise and an a11y foot-gun.
  useEffect(() => {
    if (!isActive) return;
    let raf1 = 0;
    let raf2 = 0;
    raf1 = requestAnimationFrame(() => {
      raf1 = 0;
      raf2 = requestAnimationFrame(() => {
        raf2 = 0;
        syncMetricsRef.current?.();
        const host = hostRef.current;
        const active = typeof document !== "undefined"
          ? document.activeElement
          : null;
        const focusIsElsewhereInChatPage =
          active !== null &&
          active !== document.body &&
          host !== null &&
          !host.contains(active);
        if (!focusIsElsewhereInChatPage) {
          termRef.current?.focus();
        }
      });
    });
    return () => {
      if (raf1) cancelAnimationFrame(raf1);
      if (raf2) cancelAnimationFrame(raf2);
    };
  }, [isActive]);

  // Layout:
  //   outer flex column — sits inside the dashboard's content area
  //   row split — terminal pane (flex-1) + sidebar (fixed width, lg+)
  //   terminal wrapper — rounded, dark, padded — the "terminal window"
  //   controls row — normal-flow browser controls below xterm, so they
  //     never cover the TUI composer/cursor. Copy sends `/copy\n` to Ink,
  //     which emits OSC 52 → our clipboard handler.
  //   sidebar — ChatSidebar opens its own JSON-RPC sidecar; renders
  //     model badge, tool-call list, model picker. Best-effort: if the
  //     sidecar fails to connect the terminal pane keeps working.
  //
  // `normal-case` opts out of the dashboard's global `uppercase` rule on
  // the root `<div>` in App.tsx — terminal output must preserve case.
  //
  // Mobile model/tools sheet is portaled to `document.body` so it stacks
  // above the app sidebar (`z-50`) and mobile chrome (`z-40`).  The main
  // dashboard column uses `relative z-2`, which traps `position:fixed`
  // descendants below those layers (see Toast.tsx).
  const mobileModelToolsPortal =
    isActive &&
    narrow &&
    portalRoot &&
    createPortal(
      <>
        {mobilePanelOpen && (
          <Button
            ghost
            aria-label={t.app.closeModelTools}
            onClick={closeMobilePanel}
            className={cn(
              "fixed inset-0 z-[55] p-0 block",
              "bg-black/60 backdrop-blur-sm",
            )}
          />
        )}

        <div
          id="chat-side-panel"
          role="complementary"
          aria-label={modelToolsLabel}
          className={cn(
            "font-mondwest fixed top-0 right-0 z-[60] flex h-dvh max-h-dvh w-64 min-w-0 flex-col antialiased",
            "border-l border-current/20 text-midground",
            "bg-background-base/95 backdrop-blur-sm",
            "transition-transform duration-200 ease-out",
            "[background:var(--component-sidebar-background)]",
            "[clip-path:var(--component-sidebar-clip-path)]",
            "[border-image:var(--component-sidebar-border-image)]",
            mobilePanelOpen
              ? "translate-x-0"
              : "pointer-events-none translate-x-full",
          )}
        >
          <div
            className={cn(
              "flex h-14 shrink-0 items-center justify-between gap-2 border-b border-current/20 px-5",
            )}
          >
            <Typography
              className="font-bold text-[1.125rem] leading-[0.95] tracking-[0.0525rem] text-midground"
              style={{ mixBlendMode: "plus-lighter" }}
            >
              {t.app.modelToolsSheetTitle}
              <br />
              {t.app.modelToolsSheetSubtitle}
            </Typography>

            <Button
              ghost
              size="icon"
              onClick={closeMobilePanel}
              aria-label={t.app.closeModelTools}
              className="text-midground/70 hover:text-midground"
            >
              <X />
            </Button>
          </div>

          <div
            className={cn(
              "min-h-0 flex-1 overflow-y-auto overflow-x-hidden",
              "border-t border-current/10",
            )}
          >
            <ChatSidebar channel={channel} />
          </div>
        </div>
      </>,
      portalRoot,
    );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 normal-case">
      <PluginSlot name="chat:top" />
      {mobileModelToolsPortal}

      {banner && (
        <div className="border border-warning/50 bg-warning/10 text-warning px-3 py-2 text-xs tracking-wide">
          {banner}
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:gap-3">
        <div
          className={cn(
            "relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-lg",
            "p-2 sm:p-3",
          )}
          style={{
            backgroundColor: TERMINAL_THEME.background,
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
          }}
        >
          <div
            ref={hostRef}
            className="hermes-chat-xterm-host min-h-0 min-w-0 flex-1"
          />

          <div className="mt-2 flex shrink-0 flex-wrap items-center justify-between gap-2 sm:mt-3">
            <div className="flex min-w-0 flex-1 flex-wrap gap-1.5">
              <Button
                ghost
                onClick={handleVoiceButton}
                disabled={transcribingVoice || handsFreeEnabled}
                title={
                  recording
                    ? "Stop browser microphone recording and transcribe"
                    : "Record browser microphone and send transcript"
                }
                aria-label={
                  recording
                    ? "Stop browser microphone recording"
                    : "Record browser microphone"
                }
                className={cn(
                  "rounded border border-current/30",
                  "bg-black/20 backdrop-blur-sm",
                  "opacity-70 hover:opacity-100 hover:border-current/60",
                  "transition-opacity duration-150 normal-case font-normal tracking-normal",
                  "px-2 py-1 text-[0.65rem] sm:px-2.5 sm:py-1.5 sm:text-xs",
                  recording && "border-red-300/70 text-red-200 opacity-100",
                )}
                style={{ color: recording ? undefined : TERMINAL_THEME.foreground }}
              >
                <span className="inline-flex items-center gap-1.5">
                  <Mic className="h-3 w-3 shrink-0" />
                  <span className="hidden min-[400px]:inline tracking-wide">
                    {recording ? "stop voice" : transcribingVoice ? "transcribing" : "voice"}
                  </span>
                </span>
              </Button>

              <Button
                ghost
                onClick={handleHandsFreeButton}
                disabled={recording || transcribingVoice}
                title={
                  handsFreeEnabled
                    ? "Stop hands-free browser voice mode"
                    : "Start hands-free browser voice mode"
                }
                aria-label={
                  handsFreeEnabled
                    ? "Stop hands-free browser voice mode"
                    : "Start hands-free browser voice mode"
                }
                className={cn(
                  "rounded border border-current/30",
                  "bg-black/20 backdrop-blur-sm",
                  "opacity-70 hover:opacity-100 hover:border-current/60",
                  "transition-opacity duration-150 normal-case font-normal tracking-normal",
                  "px-2 py-1 text-[0.65rem] sm:px-2.5 sm:py-1.5 sm:text-xs",
                  handsFreeEnabled && "border-emerald-300/70 text-emerald-100 opacity-100",
                  handsFreeState === "recording" && "border-red-300/70 text-red-200",
                  handsFreeState === "speaking" && "border-sky-300/70 text-sky-100",
                )}
                style={{ color: handsFreeEnabled ? undefined : TERMINAL_THEME.foreground }}
              >
                <span className="inline-flex items-center gap-1.5">
                  <Headphones className="h-3 w-3 shrink-0" />
                  <span className="hidden min-[460px]:inline tracking-wide">
                    {handsFreeStateLabel(handsFreeState)}
                  </span>
                </span>
              </Button>

              {handsFreeEnabled && (
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded border border-current/25",
                    "bg-black/20 px-2 py-1 text-[0.65rem] font-normal tracking-normal backdrop-blur-sm",
                    "text-emerald-100 sm:px-2.5 sm:py-1.5 sm:text-xs",
                    handsFreeState === "recording" && "text-red-200",
                    handsFreeState === "speaking" && "text-sky-100",
                  )}
                  title={`mic ${handsFreeMicStats.level.toFixed(3)} / threshold ${handsFreeMicStats.threshold.toFixed(3)} / noise ${handsFreeMicStats.noise.toFixed(3)}`}
                >
                  <span>{handsFreeStatusText(handsFreeState)}</span>
                  <span className="relative h-1.5 w-12 overflow-hidden rounded bg-white/15">
                    <span
                      className={cn(
                        "absolute inset-y-0 left-0 rounded transition-[width] duration-100",
                        handsFreeMicStats.speaking ? "bg-red-300/90" : "bg-emerald-300/80",
                      )}
                      style={{ width: `${handsFreeMeterPercent(handsFreeMicStats)}%` }}
                    />
                    <span className="absolute inset-y-0 left-[62.5%] w-px bg-white/55" />
                  </span>
                </span>
              )}
            </div>

            <Button
              ghost
              onClick={handleCopyLast}
              title="Copy last assistant response as raw markdown"
              aria-label="Copy last assistant response"
              className={cn(
                "shrink-0 rounded border border-current/30",
                "bg-black/20 backdrop-blur-sm",
                "opacity-60 hover:opacity-100 hover:border-current/60",
                "transition-opacity duration-150 normal-case font-normal tracking-normal",
                "px-2 py-1 text-[0.65rem] sm:px-2.5 sm:py-1.5 sm:text-xs",
              )}
              style={{ color: TERMINAL_THEME.foreground }}
            >
              <span className="inline-flex items-center gap-1.5">
                <Copy className="h-3 w-3 shrink-0" />
                <span className="hidden min-[400px]:inline tracking-wide">
                  {copyState === "copied" ? "copied" : "copy last response"}
                </span>
              </span>
            </Button>
          </div>
        </div>

        {!narrow && (
          <div
            id="chat-side-panel"
            role="complementary"
            aria-label={modelToolsLabel}
            className="flex min-h-0 shrink-0 flex-col lg:h-full lg:w-80"
          >
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
              <ChatSidebar channel={channel} />
            </div>
          </div>
        )}
      </div>
      <PluginSlot name="chat:bottom" />
    </div>
  );
}

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
    webkitAudioContext?: typeof AudioContext;
  }
}
