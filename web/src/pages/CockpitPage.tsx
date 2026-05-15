import {
  AlertTriangle,
  CheckCircle2,
  CircleStop,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  Send,
  Settings2,
  ShieldCheck,
  Volume2,
  VolumeX,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { cn } from "@/lib/utils";

type CockpitStatus =
  | "idle"
  | "listening"
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
  choices?: string[];
  [key: string]: unknown;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string };
  }>;
};

type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
};

const API_BASE_KEY = "hermes.cockpit.apiBase";
const API_KEY_KEY = "hermes.cockpit.apiKey";
const SESSION_KEY = "hermes.cockpit.sessionKey";

const DEFAULT_INSTRUCTIONS = `You are Hermes in Dalton's mobile truck cockpit. Be concise, mobile-friendly, and action-oriented. If Dalton asks you to send texts, emails, spend money, modify customer-facing systems, or take risky actions, draft first and require explicit approval unless a trusted rule already exists. Narrate blockers clearly.`;

function getSpeechRecognition(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as SpeechWindow;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
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

function statusLabel(status: CockpitStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "listening":
      return "Listening";
    case "ready":
      return "Ready";
    case "thinking":
      return "Thinking";
    case "working":
      return "Working";
    case "waiting_for_approval":
      return "Needs approval";
    case "done":
      return "Done";
    case "blocked":
      return "Blocked";
    case "error":
      return "Error";
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
  const recognitionCtor = useMemo(() => getSpeechRecognition(), []);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const eventStreamAbortRef = useRef<AbortController | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [apiBase, setApiBase] = useState(() => storageValue(API_BASE_KEY, window.location.origin));
  const [apiKey, setApiKey] = useState(() => storageValue(API_KEY_KEY, ""));
  const [sessionKey, setSessionKey] = useState(() => storageValue(SESSION_KEY, generateSessionKey()));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [status, setStatus] = useState<CockpitStatus>("idle");
  const [handsFree, setHandsFree] = useState(false);
  const [voiceReplies, setVoiceReplies] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [typedInput, setTypedInput] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [response, setResponse] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [approvalEvent, setApprovalEvent] = useState<RunEvent | null>(null);

  useEffect(() => {
    window.localStorage.setItem(API_BASE_KEY, apiBase);
  }, [apiBase]);

  useEffect(() => {
    window.localStorage.setItem(API_KEY_KEY, apiKey);
  }, [apiKey]);

  useEffect(() => {
    window.localStorage.setItem(SESSION_KEY, sessionKey);
  }, [sessionKey]);

  const speak = useCallback(
    (text: string) => {
      if (!voiceReplies || typeof window === "undefined" || !("speechSynthesis" in window)) return;
      const utterance = new SpeechSynthesisUtterance(compactForSpeech(text));
      utterance.rate = 1.02;
      utterance.pitch = 1;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    },
    [voiceReplies],
  );

  const updateStatus = useCallback(
    (next: CockpitStatus, announce?: string) => {
      setStatus(next);
      if (handsFree && announce) speak(announce);
    },
    [handsFree, speak],
  );

  const apiHeaders = useCallback(
    (json = true): HeadersInit => {
      const headers: Record<string, string> = {
        "X-Hermes-Session-Key": sessionKey,
      };
      if (json) headers["Content-Type"] = "application/json";
      if (apiKey.trim()) headers.Authorization = `Bearer ${apiKey.trim()}`;
      return headers;
    },
    [apiKey, sessionKey],
  );

  const resolvedInput = useMemo(() => {
    const spoken = `${transcript} ${interimTranscript}`.trim();
    return (typedInput || spoken).trim();
  }, [interimTranscript, transcript, typedInput]);

  const closeEventSource = useCallback(() => {
    eventStreamAbortRef.current?.abort();
    eventStreamAbortRef.current = null;
  }, []);

  useEffect(() => () => closeEventSource(), [closeEventSource]);

  const appendEvent = useCallback(
    (event: RunEvent) => {
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
      } else if (kind === "approval.responded") {
        setApprovalEvent(null);
        updateStatus("working", "Approval sent. Continuing.");
      } else if (kind === "run.completed") {
        const output = String(event.output ?? "");
        setResponse(output);
        setStreamingText("");
        updateStatus("done", output ? `Done. ${output}` : "Done.");
        closeEventSource();
      } else if (kind === "run.failed") {
        setError(String(event.error ?? "Run failed"));
        updateStatus("error", `Blocked. ${String(event.error ?? "Run failed")}`);
        closeEventSource();
      } else if (kind === "run.cancelled") {
        updateStatus("blocked", "Stopped.");
        closeEventSource();
      }
    },
    [closeEventSource, updateStatus],
  );

  const startRun = useCallback(
    async (message?: string) => {
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
          body: JSON.stringify({
            input,
            instructions: DEFAULT_INSTRUCTIONS,
            session_id: sessionKey,
          }),
        });
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        const data = (await res.json()) as { run_id?: string };
        if (!data.run_id) throw new Error("API did not return run_id");
        const activeRunId: string = data.run_id;
        setRunId(activeRunId);
        setTranscript("");
        setInterimTranscript("");
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
                const dataLines = frame
                  .split("\n")
                  .filter((line) => line.startsWith("data:"))
                  .map((line) => line.slice(5).trimStart());
                if (dataLines.length === 0) continue;
                try {
                  appendEvent(JSON.parse(dataLines.join("\n")) as RunEvent);
                } catch (parseError) {
                  setError(`Bad event payload: ${String(parseError)}`);
                }
              }
            }
          } catch (streamError) {
            if (streamAbort.signal.aborted) return;
            setError(`Event stream disconnected: ${String(streamError)}`);
            updateStatus("blocked", "Event stream disconnected.");
          }
        })();
      } catch (runError) {
        setError(String(runError));
        updateStatus("error", "I could not start the run.");
      }
    },
    [apiBase, apiHeaders, apiKey, appendEvent, closeEventSource, resolvedInput, sessionKey, updateStatus],
  );

  const sendApproval = useCallback(
    async (choice: "once" | "session" | "always" | "deny") => {
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
    },
    [apiBase, apiHeaders, runId, updateStatus],
  );

  const stopRun = useCallback(async () => {
    if (!runId) return;
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

  const handleVoiceCommand = useCallback(
    (text: string): boolean => {
      const command = text.trim().toLowerCase();
      if (!handsFree || !command) return false;
      if (["stop", "cancel", "cancel run", "abort"].includes(command)) {
        void stopRun();
        return true;
      }
      if (["repeat", "say that again", "read it again"].includes(command)) {
        speak(response || streamingText || statusLabel(status));
        return true;
      }
      if (["clear", "reset"].includes(command)) {
        setTranscript("");
        setInterimTranscript("");
        setTypedInput("");
        setError(null);
        updateStatus("idle", "Cleared.");
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
    },
    [approvalEvent, handsFree, response, sendApproval, speak, status, stopRun, streamingText, updateStatus],
  );

  const stopListening = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    recognitionRef.current?.stop();
    setIsListening(false);
    setInterimTranscript("");
    setStatus((current) => (current === "listening" ? "ready" : current));
  }, []);

  const startListening = useCallback(() => {
    if (!recognitionCtor) {
      setError("Browser speech recognition is unavailable. Use typed input or Chrome/Android.");
      updateStatus("blocked", "Speech recognition is unavailable in this browser.");
      return;
    }
    if (isListening) return;
    const recognition = new recognitionCtor();
    recognitionRef.current = recognition;
    recognition.continuous = handsFree;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) finalText += text;
        else interimText += text;
      }
      if (interimText) setInterimTranscript(interimText);
      if (finalText.trim()) {
        const commandHandled = handleVoiceCommand(finalText);
        if (!commandHandled) {
          setTranscript((current) => `${current} ${finalText}`.trim());
          if (handsFree) {
            if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = setTimeout(() => {
              void startRun(`${transcript} ${finalText}`.trim());
            }, 1400);
          }
        }
      }
    };
    recognition.onerror = (event) => {
      setError(`Speech recognition error: ${event.error ?? "unknown"}`);
      setIsListening(false);
      updateStatus("blocked", "Speech recognition stopped.");
    };
    recognition.onend = () => {
      setIsListening(false);
      if (handsFree && status !== "working" && status !== "thinking" && status !== "waiting_for_approval") {
        try {
          recognition.start();
          setIsListening(true);
          updateStatus("listening");
        } catch {
          // Browsers often require a fresh user gesture before restarting.
        }
      }
    };
    try {
      recognition.start();
      setIsListening(true);
      updateStatus("listening", "Listening.");
    } catch (listenError) {
      setError(String(listenError));
      setIsListening(false);
      updateStatus("error", "Could not start listening.");
    }
  }, [handleVoiceCommand, handsFree, isListening, recognitionCtor, startRun, status, transcript, updateStatus]);

  const toggleHandsFree = () => {
    const next = !handsFree;
    setHandsFree(next);
    if (next) {
      setVoiceReplies(true);
      speak("Hands-free mode on. Say a command, or say stop, repeat, clear, approve, or deny.");
      setTimeout(startListening, 250);
    } else {
      stopListening();
      speak("Hands-free mode off.");
    }
  };

  const statusTone = {
    idle: "border-midground/20 bg-midground/5",
    listening: "border-sky-300/50 bg-sky-400/10",
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
            <p className="mt-1 text-sm opacity-70">Mobile truck mode: talk, watch work, approve safely.</p>
          </div>
          <Button ghost size="icon" aria-label="Cockpit settings" onClick={() => setSettingsOpen((v) => !v)}>
            <Settings2 className="h-5 w-5" />
          </Button>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatusPill label="Status" value={statusLabel(status)} hot={status === "listening" || status === "working"} />
          <StatusPill label="Run" value={runId ? runId.slice(0, 13) : "none"} />
          <StatusPill label="Voice" value={recognitionCtor ? "available" : "typed fallback"} />
          <StatusPill label="Mode" value={handsFree ? "hands-free" : "manual"} hot={handsFree} />
        </div>

        {settingsOpen && (
          <div className="mt-4 grid gap-3 rounded-2xl border border-current/15 bg-black/20 p-3 text-sm">
            <label className="grid gap-1">
              <span className="text-xs uppercase tracking-[0.15em] opacity-55">API base URL</span>
              <input className="rounded-xl border border-current/20 bg-black/30 px-3 py-2 outline-none" value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://127.0.0.1:8642" />
            </label>
            <label className="grid gap-1">
              <span className="text-xs uppercase tracking-[0.15em] opacity-55">API key / bearer token (optional)</span>
              <input className="rounded-xl border border-current/20 bg-black/30 px-3 py-2 outline-none" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Only if api_server.api_key is configured" type="password" />
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
          <Button onClick={isListening ? stopListening : startListening} className="min-h-14 flex-1 rounded-2xl text-base sm:flex-none">
            {isListening ? <MicOff className="mr-2 h-5 w-5" /> : <Mic className="mr-2 h-5 w-5" />}
            {isListening ? "Stop listening" : "Push to talk"}
          </Button>
          <Button ghost onClick={toggleHandsFree} className={cn("min-h-14 rounded-2xl", handsFree && "border-emerald-300/60 bg-emerald-400/15")}>
            <Radio className="mr-2 h-5 w-5" />
            Hands-free {handsFree ? "on" : "off"}
          </Button>
          <Button ghost onClick={() => setVoiceReplies((v) => !v)} className="min-h-14 rounded-2xl">
            {voiceReplies ? <Volume2 className="mr-2 h-5 w-5" /> : <VolumeX className="mr-2 h-5 w-5" />}
            Voice replies
          </Button>
        </div>

        <textarea
          className="min-h-28 rounded-2xl border border-current/15 bg-black/25 p-3 text-base outline-none placeholder:opacity-40"
          value={typedInput || `${transcript}${interimTranscript ? ` ${interimTranscript}` : ""}`}
          onChange={(e) => {
            setTypedInput(e.target.value);
            setTranscript("");
            setInterimTranscript("");
            setStatus("ready");
          }}
          placeholder="Say or type: Text Hannah…, Check FlipperForce…, Research this property…, Have Codex build…"
        />

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void startRun()} disabled={!resolvedInput || status === "thinking" || status === "working"} className="min-h-12 flex-1 rounded-2xl sm:flex-none">
            <Send className="mr-2 h-4 w-4" />
            Send to Hermes
          </Button>
          <Button ghost onClick={stopRun} disabled={!runId || status === "done" || status === "idle"} className="min-h-12 rounded-2xl">
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
        <Typography className="mb-2 font-bold text-base">Hands-free safety rules</Typography>
        <ul className="list-disc space-y-1 pl-5 normal-case">
          <li>Hands-free mode may auto-submit your dictated command after a short pause.</li>
          <li>It does not auto-approve risky actions. Say “approve” or “deny” when an approval card appears.</li>
          <li>Browser speech recognition works best in Chrome/Android. iOS support is limited; typed fallback remains available.</li>
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
