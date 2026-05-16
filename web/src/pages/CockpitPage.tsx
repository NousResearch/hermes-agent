import {
  AlertTriangle,
  CheckCircle2,
  CircleStop,
  Download,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  Send,
  Settings2,
  ShieldCheck,
  Smartphone,
  Volume2,
  VolumeX,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { cn } from "@/lib/utils";

type CockpitStatus =
  | "idle"
  | "listening"
  | "transcribing"
  | "ready"
  | "thinking"
  | "working"
  | "waiting_for_approval"
  | "done"
  | "blocked"
  | "error";

type RunEvent = {
  event?: string;
  run_id?: string;
  timestamp?: number;
  delta?: string;
  output?: string;
  error?: string;
  tool?: string;
  preview?: string;
  duration?: number;
  choice?: string;
  choices?: string[];
  [key: string]: unknown;
};

const API_BASE_KEY = "hermes.cockpit.apiBase";
const API_KEY_KEY = "hermes.cockpit.apiKey";
const SESSION_KEY = "hermes.cockpit.sessionKey";
const HANDS_FREE_SECONDS = 8;

const DEFAULT_INSTRUCTIONS = `You are Hermes in Dalton's mobile truck cockpit. Be concise, mobile-friendly, and action-oriented. If Dalton asks you to send texts, emails, spend money, modify customer-facing systems, or take risky actions, draft first and require explicit approval unless a trusted rule already exists. Narrate blockers clearly.`;

function normalizeApiBase(value: string): string {
  const fallback = typeof window === "undefined" ? "" : window.location.origin;
  const raw = (value || fallback).trim() || fallback;
  const stripAppRoute = (path: string) => {
    const normalized = path.replace(/\/+$/, "");
    const marker = normalized.search(/\/(cockpit|dashboard|v1)(\/|$)/i);
    return marker >= 0 ? normalized.slice(0, marker) || "/" : normalized || "/";
  };
  try {
    const url = new URL(raw, fallback || undefined);
    // Users may save/copy the cockpit page URL or a full /v1 endpoint as the API
    // base. The frontend appends /v1 itself, so keep only the server origin/base.
    url.pathname = stripAppRoute(url.pathname);
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/+$/, "");
  } catch {
    return raw.replace(/\/+$/, "").replace(/\/(cockpit|dashboard|v1)(\/.*)?$/i, "");
  }
}

function storageValue(key: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return window.localStorage.getItem(key) ?? fallback;
}

function generateSessionKey(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `cockpit:${crypto.randomUUID()}`;
  }
  return `cockpit:${Date.now().toString(36)}:${Math.random().toString(36).slice(2)}`;
}

function isSecureMicContext(): boolean {
  return typeof window !== "undefined" && (window.isSecureContext || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
}

function pickAudioMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return undefined;
  for (const type of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return undefined;
}

function statusLabel(status: CockpitStatus): string {
  switch (status) {
    case "idle": return "Idle";
    case "listening": return "Listening";
    case "transcribing": return "Transcribing";
    case "ready": return "Ready";
    case "thinking": return "Thinking";
    case "working": return "Working";
    case "waiting_for_approval": return "Needs approval";
    case "done": return "Done";
    case "blocked": return "Blocked";
    case "error": return "Error";
  }
}

function eventSummary(event: RunEvent): string {
  const kind = event.event ?? "event";
  if (kind === "message.delta") return `Hermes: ${String(event.delta ?? "")}`;
  if (kind === "tool.started") return `Started ${String(event.tool ?? "tool")}${event.preview ? ` — ${String(event.preview)}` : ""}`;
  if (kind === "tool.completed") return `Completed ${String(event.tool ?? "tool")}${event.duration ? ` in ${event.duration}s` : ""}`;
  if (kind === "approval.request") return "Approval requested";
  if (kind === "approval.responded") return `Approval response: ${String(event.choice ?? "sent")}`;
  if (kind === "run.completed") return "Run completed";
  if (kind === "run.failed") return `Run failed: ${String(event.error ?? "unknown error")}`;
  if (kind === "run.cancelled") return "Run cancelled";
  if (kind === "reasoning.available") return "Reasoning available";
  return kind;
}

function compactForSpeech(text: string): string {
  const trimmed = text.replace(/\s+/g, " ").trim();
  if (trimmed.length <= 420) return trimmed;
  return `${trimmed.slice(0, 420)}…`;
}

export default function CockpitPage() {
  const eventStreamAbortRef = useRef<AbortController | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pushToTalkActiveRef = useRef(false);
  const handsFreeRef = useRef(false);
  const statusRef = useRef<CockpitStatus>("idle");

  const [apiBase, setApiBaseState] = useState(() => normalizeApiBase(storageValue(API_BASE_KEY, window.location.origin)));
  const setApiBase = useCallback((value: string) => setApiBaseState(normalizeApiBase(value)), []);
  const [apiKey, setApiKey] = useState(() => storageValue(API_KEY_KEY, ""));
  const [sessionKey, setSessionKey] = useState(() => storageValue(SESSION_KEY, generateSessionKey()));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [status, setStatus] = useState<CockpitStatus>("idle");
  const [handsFree, setHandsFree] = useState(false);
  const [voiceReplies, setVoiceReplies] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [typedInput, setTypedInput] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [response, setResponse] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [approvalEvent, setApprovalEvent] = useState<RunEvent | null>(null);
  const [lastTranscript, setLastTranscript] = useState("");
  const [installPrompt, setInstallPrompt] = useState<Event | null>(null);

  useEffect(() => { window.localStorage.setItem(API_BASE_KEY, apiBase); }, [apiBase]);
  useEffect(() => { window.localStorage.setItem(API_KEY_KEY, apiKey); }, [apiKey]);
  useEffect(() => { window.localStorage.setItem(SESSION_KEY, sessionKey); }, [sessionKey]);
  useEffect(() => { handsFreeRef.current = handsFree; }, [handsFree]);
  useEffect(() => { statusRef.current = status; }, [status]);

  useEffect(() => {
    const handler = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event);
    };
    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, []);

  const speak = useCallback((text: string) => {
    if (!voiceReplies || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const utterance = new SpeechSynthesisUtterance(compactForSpeech(text));
    utterance.rate = 1.02;
    utterance.pitch = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.resume();
    window.speechSynthesis.speak(utterance);
  }, [voiceReplies]);

  const updateStatus = useCallback((next: CockpitStatus, announce?: string) => {
    setStatus(next);
    statusRef.current = next;
    if (handsFreeRef.current && announce) speak(announce);
  }, [speak]);

  const apiHeaders = useCallback((json = true): HeadersInit => {
    const headers: Record<string, string> = { "X-Hermes-Session-Key": sessionKey };
    if (json) headers["Content-Type"] = "application/json";
    if (apiKey.trim()) headers.Authorization = `Bearer ${apiKey.trim()}`;
    return headers;
  }, [apiKey, sessionKey]);

  const resolvedInput = useMemo(() => (typedInput || transcript).trim(), [transcript, typedInput]);

  const closeEventSource = useCallback(() => {
    eventStreamAbortRef.current?.abort();
    eventStreamAbortRef.current = null;
  }, []);

  const stopMediaTracks = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  }, []);

  useEffect(() => () => {
    closeEventSource();
    if (recordTimerRef.current) clearTimeout(recordTimerRef.current);
    mediaRecorderRef.current?.stop();
    stopMediaTracks();
  }, [closeEventSource, stopMediaTracks]);

  const restartHandsFreeLater = useCallback((delay = 950) => {
    if (!handsFreeRef.current) return;
    window.setTimeout(() => {
      if (!handsFreeRef.current) return;
      if (["thinking", "working", "waiting_for_approval", "transcribing"].includes(statusRef.current)) return;
      void startRecording(true);
    }, delay);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const appendEvent = useCallback((event: RunEvent) => {
    setEvents((current) => [event, ...current].slice(0, 80));
    const kind = event.event;
    if (kind === "message.delta") {
      setStreamingText((text) => `${text}${String(event.delta ?? "")}`);
      updateStatus("working");
    } else if (kind === "tool.started") {
      updateStatus("working", `Working. Started ${String(event.tool ?? "a tool")}.`);
    } else if (kind === "approval.request") {
      setApprovalEvent(event);
      updateStatus("waiting_for_approval", "I need approval before continuing.");
      restartHandsFreeLater(700);
    } else if (kind === "approval.responded") {
      setApprovalEvent(null);
      updateStatus("working", "Approval sent. Continuing.");
    } else if (kind === "run.completed") {
      const output = String(event.output ?? "");
      setResponse(output);
      setStreamingText("");
      updateStatus("done", output ? `Done. ${output}` : "Done.");
      // In walkie/typed mode Dalton still expects the Voice replies toggle to
      // read the final answer aloud. Hands-free already speaks via updateStatus.
      if (output && !handsFreeRef.current) speak(output);
      closeEventSource();
      restartHandsFreeLater(1800);
    } else if (kind === "run.failed") {
      setError(String(event.error ?? "Run failed"));
      updateStatus("error", `Blocked. ${String(event.error ?? "Run failed")}`);
      closeEventSource();
      restartHandsFreeLater(1800);
    } else if (kind === "run.cancelled") {
      updateStatus("blocked", "Stopped.");
      closeEventSource();
      restartHandsFreeLater(1200);
    }
  }, [closeEventSource, restartHandsFreeLater, speak, updateStatus]);

  const startRun = useCallback(async (message?: string) => {
    const input = (message ?? resolvedInput).trim();
    if (!input) {
      setError("Say or type a command first.");
      updateStatus("blocked", "Say or type a command first.");
      return;
    }
    closeEventSource();
    setError(null);
    setApprovalEvent(null);
    setResponse("");
    setStreamingText("");
    setEvents([]);
    updateStatus("thinking", "Thinking.");

    try {
      const res = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/runs`, {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify({ input, instructions: DEFAULT_INSTRUCTIONS, session_id: sessionKey }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = (await res.json()) as { run_id?: string };
      if (!data.run_id) throw new Error("API did not return run_id");
      const activeRunId: string = data.run_id;
      setRunId(activeRunId);
      setTranscript("");
      setTypedInput("");
      updateStatus("working", "Working.");

      const streamAbort = new AbortController();
      eventStreamAbortRef.current = streamAbort;
      void (async () => {
        try {
          const streamRes = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/runs/${encodeURIComponent(activeRunId)}/events`, {
            headers: apiHeaders(false),
            signal: streamAbort.signal,
          });
          if (!streamRes.ok) throw new Error(`${streamRes.status}: ${await streamRes.text()}`);
          if (!streamRes.body) throw new Error("Event stream response had no body");

          const reader = streamRes.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const frames = buffer.split("\n\n");
            buffer = frames.pop() ?? "";
            for (const frame of frames) {
              const dataLines = frame.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart());
              if (dataLines.length === 0) continue;
              try { appendEvent(JSON.parse(dataLines.join("\n")) as RunEvent); }
              catch (parseError) { setError(`Bad event payload: ${String(parseError)}`); }
            }
          }
        } catch (streamError) {
          if (streamAbort.signal.aborted) return;
          setError(`Event stream disconnected: ${String(streamError)}`);
          updateStatus("blocked", "Event stream disconnected.");
          restartHandsFreeLater(1500);
        }
      })();
    } catch (runError) {
      setError(String(runError));
      updateStatus("error", "I could not start the run.");
      restartHandsFreeLater(1500);
    }
  }, [apiBase, apiHeaders, appendEvent, closeEventSource, resolvedInput, restartHandsFreeLater, sessionKey, updateStatus]);

  const sendApproval = useCallback(async (choice: "once" | "session" | "always" | "deny") => {
    if (!runId) return;
    setError(null);
    try {
      const res = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/runs/${encodeURIComponent(runId)}/approval`, {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify({ choice }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      setApprovalEvent(null);
      updateStatus(choice === "deny" ? "blocked" : "working", choice === "deny" ? "Denied." : "Approved.");
    } catch (approvalError) {
      setError(String(approvalError));
      updateStatus("error", "Approval failed.");
    }
  }, [apiBase, apiHeaders, runId, updateStatus]);

  const stopRun = useCallback(async () => {
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
    if (!runId) {
      closeEventSource();
      updateStatus("blocked", "Stopped.");
      return;
    }
    setError(null);
    try {
      const res = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/runs/${encodeURIComponent(runId)}/stop`, {
        method: "POST",
        headers: apiHeaders(false),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      closeEventSource();
      updateStatus("blocked", "Stopped.");
    } catch (stopError) {
      setError(String(stopError));
      updateStatus("error", "Stop failed.");
    }
  }, [apiBase, apiHeaders, closeEventSource, runId, updateStatus]);

  const handleVoiceCommand = useCallback((text: string): boolean => {
    const command = text.trim().toLowerCase().replace(/[.!?]$/g, "");
    if (!command) return false;
    if (["stop", "cancel", "cancel run", "abort", "interrupt"].includes(command)) {
      void stopRun();
      return true;
    }
    if (["repeat", "say that again", "read it again"].includes(command)) {
      speak(response || streamingText || statusLabel(statusRef.current));
      return true;
    }
    if (["clear", "reset"].includes(command)) {
      setTranscript("");
      setTypedInput("");
      setError(null);
      updateStatus("idle", "Cleared.");
      restartHandsFreeLater(500);
      return true;
    }
    if (approvalEvent) {
      if (["yes", "approve", "approved", "send it", "do it", "allow"].includes(command)) {
        void sendApproval("once");
        return true;
      }
      if (["no", "deny", "reject", "don't", "do not", "cancel"].includes(command)) {
        void sendApproval("deny");
        return true;
      }
    }
    return false;
  }, [approvalEvent, response, restartHandsFreeLater, sendApproval, speak, stopRun, streamingText, updateStatus]);

  const transcribeBlob = useCallback(async (blob: Blob) => {
    const form = new FormData();
    const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
    form.append("audio", blob, `cockpit.${ext}`);
    const res = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/cockpit/transcribe`, {
      method: "POST",
      headers: apiHeaders(false),
      body: form,
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    const data = (await res.json()) as { text?: string };
    return (data.text ?? "").trim();
  }, [apiBase, apiHeaders]);

  const processAudioBlob = useCallback(async (blob: Blob, autoSubmit: boolean) => {
    if (blob.size < 700) {
      if (autoSubmit) restartHandsFreeLater(300);
      return;
    }
    updateStatus("transcribing", "Transcribing.");
    setError(null);
    try {
      const text = await transcribeBlob(blob);
      setLastTranscript(text);
      if (!text) {
        updateStatus("listening", "I did not catch that.");
        if (autoSubmit) restartHandsFreeLater(600);
        return;
      }
      if (handleVoiceCommand(text)) return;
      setTranscript(text);
      setTypedInput("");
      if (autoSubmit) void startRun(text);
      else updateStatus("ready", "Ready to send.");
    } catch (transcribeError) {
      setError(String(transcribeError));
      updateStatus("error", "Transcription failed.");
      if (autoSubmit) restartHandsFreeLater(1400);
    }
  }, [handleVoiceCommand, restartHandsFreeLater, startRun, transcribeBlob, updateStatus]);

  async function startRecording(autoSubmit = false, maxSeconds = HANDS_FREE_SECONDS) {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("This browser cannot record microphone audio. Try Android Chrome over HTTPS or use typed input.");
      updateStatus("blocked", "Microphone recording is unavailable.");
      return;
    }
    if (!isSecureMicContext()) {
      setError("Microphone access requires HTTPS or localhost. Use the Tailscale HTTPS URL / serve route before using phone voice.");
      updateStatus("blocked", "Phone microphone requires HTTPS.");
      return;
    }
    if (mediaRecorderRef.current?.state === "recording") return;
    if (["thinking", "working", "transcribing"].includes(statusRef.current)) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];
      const mimeType = pickAudioMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        setIsRecording(false);
        if (recordTimerRef.current) {
          clearTimeout(recordTimerRef.current);
          recordTimerRef.current = null;
        }
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        audioChunksRef.current = [];
        stopMediaTracks();
        void processAudioBlob(blob, autoSubmit);
      };
      recorder.start(250);
      setIsRecording(true);
      updateStatus("listening", autoSubmit ? "Listening." : "Recording.");
      if (autoSubmit && maxSeconds > 0) {
        recordTimerRef.current = setTimeout(() => {
          if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
        }, maxSeconds * 1000);
      }
    } catch (recordError) {
      stopMediaTracks();
      setIsRecording(false);
      setError(String(recordError));
      updateStatus("error", "Could not access microphone.");
    }
  }

  const stopRecording = useCallback(() => {
    if (recordTimerRef.current) {
      clearTimeout(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
  }, []);

  const startPushToTalk = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (["thinking", "working", "transcribing"].includes(statusRef.current)) return;
    pushToTalkActiveRef.current = true;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    void startRecording(true, 60);
  };

  const finishPushToTalk = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (!pushToTalkActiveRef.current) return;
    pushToTalkActiveRef.current = false;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    stopRecording();
  };

  const toggleHandsFree = () => {
    const next = !handsFree;
    setHandsFree(next);
    handsFreeRef.current = next;
    if (next) {
      setVoiceReplies(true);
      speak("Hands-free mode on. I will listen in short turns. Say stop, repeat, clear, approve, or deny.");
      setTimeout(() => void startRecording(true), 350);
    } else {
      stopRecording();
      speak("Hands-free mode off.");
      updateStatus("idle");
    }
  };

  const installApp = async () => {
    const prompt = installPrompt as Event & { prompt?: () => Promise<void>; userChoice?: Promise<unknown> };
    if (prompt?.prompt) {
      await prompt.prompt();
      await prompt.userChoice?.catch(() => undefined);
      setInstallPrompt(null);
    }
  };

  const statusTone = {
    idle: "border-midground/20 bg-midground/5",
    listening: "border-sky-300/50 bg-sky-400/10",
    transcribing: "border-purple-300/50 bg-purple-400/10",
    ready: "border-emerald-300/50 bg-emerald-400/10",
    thinking: "border-amber-300/50 bg-amber-400/10",
    working: "border-blue-300/50 bg-blue-400/10",
    waiting_for_approval: "border-orange-300/60 bg-orange-400/10",
    done: "border-emerald-300/60 bg-emerald-400/10",
    blocked: "border-zinc-300/50 bg-zinc-400/10",
    error: "border-red-300/60 bg-red-400/10",
  } satisfies Record<CockpitStatus, string>;

  return (
    <main className="mx-auto flex h-full min-h-0 w-full max-w-5xl flex-col gap-3 overflow-y-auto pb-24 normal-case sm:gap-4">
      <section className={cn("rounded-3xl border p-4 shadow-2xl sm:p-5", statusTone[status])}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <Typography className="font-bold text-2xl tracking-wide text-midground sm:text-3xl">Hermes Cockpit</Typography>
            <p className="mt-1 text-sm opacity-70">Mobile truck mode: record, transcribe locally, watch work, approve safely.</p>
          </div>
          <Button ghost size="icon" aria-label="Cockpit settings" onClick={() => setSettingsOpen((v) => !v)}>
            <Settings2 className="h-5 w-5" />
          </Button>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatusPill label="Status" value={statusLabel(status)} hot={status === "listening" || status === "working"} />
          <StatusPill label="Run" value={runId ? runId.slice(0, 13) : "none"} />
          <StatusPill label="Mic" value={isSecureMicContext() ? "server Whisper" : "HTTPS needed"} hot={isSecureMicContext()} />
          <StatusPill label="Mode" value={handsFree ? "hands-free" : "walkie"} hot={handsFree} />
        </div>

        {settingsOpen && (
          <div className="mt-4 grid gap-3 rounded-2xl border border-current/15 bg-black/20 p-3 text-sm">
            <label className="grid gap-1">
              <span className="text-xs uppercase tracking-[0.15em] opacity-55">API base URL</span>
              <input className="rounded-xl border border-current/20 bg-black/30 px-3 py-2 outline-none" value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://desktop-vcb4ksf-1.tail87092b.ts.net:8642" />
            </label>
            <label className="grid gap-1">
              <span className="text-xs uppercase tracking-[0.15em] opacity-55">API key / bearer token (optional)</span>
              <input className="rounded-xl border border-current/20 bg-black/30 px-3 py-2 outline-none" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Only if API_SERVER_KEY is configured" type="password" />
            </label>
            <label className="grid gap-1">
              <span className="text-xs uppercase tracking-[0.15em] opacity-55">Session key</span>
              <div className="flex gap-2">
                <input className="min-w-0 flex-1 rounded-xl border border-current/20 bg-black/30 px-3 py-2 outline-none" value={sessionKey} onChange={(e) => setSessionKey(e.target.value)} />
                <Button ghost onClick={() => setSessionKey(generateSessionKey())}>Reset</Button>
              </div>
            </label>
          </div>
        )}
      </section>

      <section className="grid gap-3 rounded-3xl border border-current/15 bg-black/20 p-4 sm:p-5">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            onPointerDown={startPushToTalk}
            onPointerUp={finishPushToTalk}
            onPointerCancel={finishPushToTalk}
            onContextMenu={(event) => event.preventDefault()}
            disabled={status === "thinking" || status === "working" || status === "transcribing"}
            className={cn("min-h-16 flex-1 rounded-2xl text-base select-none touch-none sm:flex-none", isRecording && "border-sky-300/70 bg-sky-400/20")}
          >
            {isRecording ? <MicOff className="mr-2 h-5 w-5" /> : <Mic className="mr-2 h-5 w-5" />}
            {isRecording ? "Release to send" : "Hold to talk"}
          </Button>
          <Button ghost onClick={toggleHandsFree} className={cn("min-h-16 rounded-2xl", handsFree && "border-emerald-300/60 bg-emerald-400/15")}>
            <Radio className="mr-2 h-5 w-5" />
            Hands-free {handsFree ? "on" : "off"}
          </Button>
          <Button ghost onClick={() => setVoiceReplies((v) => !v)} className="min-h-16 rounded-2xl">
            {voiceReplies ? <Volume2 className="mr-2 h-5 w-5" /> : <VolumeX className="mr-2 h-5 w-5" />}
            Voice replies
          </Button>
          <Button ghost onClick={installApp} disabled={!installPrompt} className="min-h-16 rounded-2xl">
            <Download className="mr-2 h-5 w-5" />
            Install app
          </Button>
        </div>

        <textarea
          className="min-h-28 rounded-2xl border border-current/15 bg-black/25 p-3 text-base outline-none placeholder:opacity-40"
          value={typedInput || transcript}
          onChange={(e) => {
            setTypedInput(e.target.value);
            setTranscript("");
            setStatus("ready");
          }}
          placeholder="Say or type: Text Hannah…, Check FlipperForce…, Research this property…, Have Codex build…"
        />
        {lastTranscript && <p className="text-xs opacity-60">Last heard: “{lastTranscript}”</p>}

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void startRun()} disabled={!resolvedInput || status === "thinking" || status === "working" || status === "transcribing"} className="min-h-12 flex-1 rounded-2xl !normal-case !tracking-normal sm:flex-none">
            <Send className="mr-2 h-4 w-4 shrink-0" />
            <span className="whitespace-nowrap !normal-case !tracking-normal text-sm sm:text-base">Send<span className="hidden sm:inline"> to Hermes</span></span>
          </Button>
          <Button ghost onClick={stopRun} disabled={!runId && !isRecording} className="min-h-12 rounded-2xl">
            <CircleStop className="mr-2 h-4 w-4" />
            Stop
          </Button>
          <Button ghost onClick={() => speak(response || streamingText || "Nothing to repeat yet.")} className="min-h-12 rounded-2xl">
            <RefreshCw className="mr-2 h-4 w-4" />
            Repeat
          </Button>
        </div>
      </section>

      {approvalEvent && (
        <section className="rounded-3xl border border-orange-300/60 bg-orange-400/10 p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <ShieldCheck className="mt-1 h-6 w-6 shrink-0" />
            <div className="min-w-0 flex-1">
              <Typography className="font-bold text-xl">Approval needed</Typography>
              <p className="mt-1 text-sm opacity-75">Hermes paused before a risky action. Hands-free commands: “approve” or “deny”.</p>
              <pre className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap rounded-2xl bg-black/30 p-3 text-xs normal-case opacity-85">{JSON.stringify(approvalEvent, null, 2)}</pre>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Button onClick={() => void sendApproval("once")}><CheckCircle2 className="mr-2 h-4 w-4" />Once</Button>
            <Button ghost onClick={() => void sendApproval("session")}>Session</Button>
            <Button ghost onClick={() => void sendApproval("always")}>Always</Button>
            <Button ghost onClick={() => void sendApproval("deny")}><XCircle className="mr-2 h-4 w-4" />Deny</Button>
          </div>
        </section>
      )}

      {error && (
        <section className="flex gap-3 rounded-3xl border border-red-300/50 bg-red-400/10 p-4 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <p className="break-words">{error}</p>
        </section>
      )}

      <section className="grid gap-3 lg:grid-cols-[1fr_0.8fr]">
        <div className="rounded-3xl border border-current/15 bg-black/20 p-4 sm:p-5">
          <Typography className="font-bold text-xl">Hermes response</Typography>
          <div className="mt-3 min-h-40 whitespace-pre-wrap rounded-2xl bg-black/25 p-3 text-sm leading-relaxed normal-case">
            {response || streamingText || <span className="opacity-45">No response yet.</span>}
          </div>
        </div>

        <div className="rounded-3xl border border-current/15 bg-black/20 p-4 sm:p-5">
          <Typography className="font-bold text-xl">Live action stream</Typography>
          <div className="mt-3 grid max-h-96 gap-2 overflow-auto pr-1">
            {events.length === 0 ? (
              <p className="rounded-2xl bg-black/25 p-3 text-sm opacity-45">No events yet.</p>
            ) : (
              events.map((event, index) => (
                <div key={`${event.timestamp ?? index}-${index}`} className="rounded-2xl border border-current/10 bg-black/25 p-3 text-xs normal-case">
                  <div className="mb-1 flex items-center justify-between gap-2 opacity-60">
                    <span>{event.event ?? "event"}</span>
                    <span>{event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ""}</span>
                  </div>
                  <p className="break-words text-sm opacity-90">{eventSummary(event)}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-current/15 bg-black/20 p-4 text-xs leading-relaxed opacity-75 sm:p-5">
        <Typography className="mb-2 flex items-center gap-2 font-bold text-base"><Smartphone className="h-4 w-4" />App + hands-free notes</Typography>
        <ul className="list-disc space-y-1 pl-5 normal-case">
          <li>Use “Install app” or Chrome menu → Add to Home screen to save this cockpit like an app.</li>
          <li>Voice uses Android microphone recording plus server-side Whisper, not Chrome speech recognition.</li>
          <li>Hands-free records short turns, transcribes, submits after you stop talking, speaks back, then listens again.</li>
          <li>Phone microphone and PWA install require HTTPS or localhost; use a private Tailscale HTTPS route.</li>
          <li>It does not auto-approve risky actions. Say “approve” or “deny” when an approval card appears.</li>
        </ul>
      </section>
    </main>
  );
}

function StatusPill({ label, value, hot = false }: { label: string; value: string; hot?: boolean }) {
  return (
    <div className={cn("rounded-2xl border border-current/15 bg-black/20 p-3", hot && "border-emerald-300/50 bg-emerald-400/10")}>
      <div className="text-[0.65rem] uppercase tracking-[0.16em] opacity-45">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold normal-case">{value}</div>
    </div>
  );
}
