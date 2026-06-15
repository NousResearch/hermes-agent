import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
  type ReactNode,
} from "react";
import {
  ArrowRight,
  CalendarDays,
  CheckCircle2,
  Clock3,
  CircleDot,
  Code2,
  Compass,
  Copy,
  Github,
  Globe,
  Inbox,
  Layers3,
  Loader2,
  Mic,
  MicOff,
  MessageSquare,
  NotebookPen,
  Radar,
  Sparkles,
  SquareActivity,
  Send,
  Terminal,
  Wrench,
  Zap,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import { GatewayClient } from "@/lib/gatewayClient";
import { cn } from "@/lib/utils";

type Tone = "good" | "warn" | "alert" | "neutral";

const toneStyles: Record<Tone, string> = {
  good: "border-[#4ade80]/18 bg-[#4ade80]/10 text-[#bff7d4]",
  warn: "border-[#fbbf24]/18 bg-[#fbbf24]/10 text-[#fde68a]",
  alert: "border-[#fb7185]/18 bg-[#fb7185]/10 text-[#fecdd3]",
  neutral: "border-white/10 bg-white/5 text-white/72",
};

const quickSignals = [
  { icon: Inbox, label: "Inbox triage", value: "12 unread", note: "Two messages can wait" },
  { icon: CalendarDays, label: "Calendar", value: "3 blocks", note: "One deep-work slot open" },
  { icon: SquareActivity, label: "Energy", value: "Steady", note: "Keep intensity modest" },
  { icon: Wrench, label: "Ops", value: "Ready", note: "No active incidents" },
];

const feedItems = [
  {
    time: "13:02",
    title: "Morning briefing captured",
    detail: "Today’s priorities are visible in one place, with notes and follow-ups attached.",
    tone: "good" as Tone,
  },
  {
    time: "11:40",
    title: "One task waiting",
    detail: "A message needs a direct reply before lunch; everything else can batch later.",
    tone: "warn" as Tone,
  },
  {
    time: "09:15",
    title: "System health check passed",
    detail: "Dashboard, sync layer, and automation hooks are all in the green.",
    tone: "good" as Tone,
  },
  {
    time: "08:10",
    title: "Review queued",
    detail: "A new note from yesterday is ready for review and conversion into actions.",
    tone: "neutral" as Tone,
  },
];

const sourceModules = [
  {
    title: "Notes",
    icon: NotebookPen,
    detail: "Capture, link, and surface context without hunting across apps.",
  },
  {
    title: "Tasks",
    icon: CheckCircle2,
    detail: "Short list, clear next action, no hidden backlog.",
  },
  {
    title: "Calendar",
    icon: CalendarDays,
    detail: "Appointments, travel, and time blocks in one view.",
  },
  {
    title: "Automation",
    icon: Zap,
    detail: "Small automations for capture, triage, and reminders.",
  },
  {
    title: "Research",
    icon: Compass,
    detail: "Quick access to reference material and decision notes.",
  },
  {
    title: "Systems",
    icon: Globe,
    detail: "Health, uptime, and status for the personal OS stack.",
  },
];

const skillLayers = [
  {
    level: "router",
    title: "Personal OS Router",
    detail: "Keyword / regex intake, intent split, and dispatch.",
    accent: "border-[#45d0ff]/18 bg-[#45d0ff]/10 text-[#b9f0ff]",
    chips: ["personal-os-skill-router"],
  },
  {
    level: "specialists",
    title: "Specialists",
    detail: "The workhorses that take a request and turn it into action.",
    accent: "border-white/10 bg-white/5 text-white/72",
    chips: ["dashboard-shell-design", "hermes-agent", "subagent-driven-development", "writing-plans"],
  },
  {
    level: "domain",
    title: "Domain skills",
    detail: "The application layers beneath the router.",
    accent: "border-white/10 bg-black/20 text-white/72",
    chips: ["obsidian", "google-workspace", "github-operations", "linear", "telegram-cron-ops"],
  },
];

const nodes = [
  { x: 50, y: 18, size: 10 },
  { x: 82, y: 34, size: 8 },
  { x: 20, y: 36, size: 9 },
  { x: 62, y: 68, size: 11 },
  { x: 28, y: 66, size: 7 },
  { x: 50, y: 50, size: 16 },
  { x: 50, y: 82, size: 8 },
  { x: 74, y: 14, size: 7 },
  { x: 12, y: 18, size: 7 },
  { x: 86, y: 78, size: 7 },
];

function ShellCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-[22px] border border-white/8 bg-[linear-gradient(180deg,rgba(16,18,24,0.96),rgba(8,10,14,0.98))] shadow-[0_20px_42px_rgba(0,0,0,0.36)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

function StatTile({ label, value, detail, tone }: { label: string; value: string; detail: string; tone: Tone }) {
  return (
    <ShellCard className="p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">{label}</div>
        <span className={cn("rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.24em]", toneStyles[tone])}>
          {tone}
        </span>
      </div>
      <div className="mt-4 flex items-end gap-2">
        <div className="text-[36px] font-semibold leading-none tracking-[-0.08em] text-white">{value}</div>
      </div>
      <div className="mt-3 text-[12px] leading-5 text-white/54">{detail}</div>
    </ShellCard>
  );
}

function SignalRow({ icon: Icon, label, value, note }: { icon: ComponentType<{ className?: string }>; label: string; value: string; note: string }) {
  return (
    <div className="rounded-[18px] border border-white/8 bg-black/20 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
            <Icon className="h-4 w-4 text-[#45d0ff]" />
            {label}
          </div>
          <div className="mt-3 text-[22px] font-semibold tracking-[-0.05em] text-white">{value}</div>
        </div>
        <ArrowRight className="mt-1 h-4 w-4 text-white/36" />
      </div>
      <div className="mt-2 text-[12px] leading-5 text-white/60">{note}</div>
    </div>
  );
}

function FeedRow({ time, title, detail, tone }: { time: string; title: string; detail: string; tone: Tone }) {
  return (
    <div className={cn("rounded-[18px] border p-4", toneStyles[tone])}>
      <div className="flex items-start gap-3">
        <span className={cn("mt-1 h-2.5 w-2.5 shrink-0 rounded-full", tone === "good" ? "bg-[#4ade80]" : tone === "warn" ? "bg-[#fbbf24]" : tone === "alert" ? "bg-[#fb7185]" : "bg-white/40")} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/36">
            <span>{time}</span>
            <span>·</span>
            <span>{tone}</span>
          </div>
          <div className="mt-1 text-[15px] font-semibold tracking-[-0.02em] text-white">{title}</div>
          <p className="mt-1 text-[12px] leading-5 text-white/62">{detail}</p>
        </div>
      </div>
    </div>
  );
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read recorded audio"));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(blob);
  });
}

function shortSessionId(sessionId: string | null): string {
  if (!sessionId) return "new session";
  if (sessionId.length <= 12) return sessionId;
  return `${sessionId.slice(0, 6)}…${sessionId.slice(-4)}`;
}

function VoiceConsole() {
  const navigate = useNavigate();
  const gateway = useMemo(() => new GatewayClient(), []);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [status, setStatus] = useState<"idle" | "recording" | "transcribing" | "sending" | "error">("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const stopTracks = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach((track) => track.stop());
  }, []);

  useEffect(() => {
    return () => {
      gateway.close();
      stopTracks();
      recorderRef.current = null;
    };
  }, [gateway, stopTracks]);

  const finishRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    recorderRef.current = null;
    stopTracks();

    const chunks = chunksRef.current;
    chunksRef.current = [];

    if (!chunks.length) {
      setStatus("error");
      setError("No audio was captured.");
      return;
    }

    const mimeType = recorder?.mimeType || chunks[0]?.type || "audio/webm";
    const blob = new Blob(chunks, { type: mimeType });

    setStatus("transcribing");
    setError(null);

    try {
      const dataUrl = await blobToDataUrl(blob);
      const result = await api.transcribeAudio(dataUrl, mimeType);
      const text = result.transcript.trim();

      if (!result.ok || !text) {
        throw new Error("Transcription returned no usable text.");
      }

      setTranscript(text);
      setStatus("sending");

      await gateway.connect();
      let sid = sessionId;
      if (!sid) {
        const created = await gateway.request<{ session_id: string }>("session.create", {});
        sid = created.session_id;
        setSessionId(sid);
      }

      await gateway.request("prompt.submit", { session_id: sid, text });
      setStatus("idle");
      navigate(`/chat?resume=${encodeURIComponent(sid)}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Voice submission failed";
      setStatus("error");
      setError(message);
    }
  }, [gateway, navigate, sessionId, stopTracks]);

  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setError("This browser cannot access the microphone.");
      return;
    }

    if (typeof MediaRecorder === "undefined") {
      setStatus("error");
      setError("This browser does not support audio recording.");
      return;
    }

    setError(null);
    setTranscript("");
    setStatus("recording");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const preferredMime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";

      const recorder = preferredMime ? new MediaRecorder(stream, { mimeType: preferredMime }) : new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) chunksRef.current.push(ev.data);
      };

      recorder.onerror = () => {
        setStatus("error");
        setError("Recording failed.");
        stopTracks();
      };

      recorder.onstop = () => {
        void finishRecording();
      };

      recorder.start();
    } catch (err) {
      stopTracks();
      const message = err instanceof Error ? err.message : "Could not start recording";
      setStatus("error");
      setError(message);
    }
  }, [finishRecording, stopTracks]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    setStatus("transcribing");
    try {
      recorder.stop();
    } catch {
      void finishRecording();
    }
  }, [finishRecording]);

  const toggleRecording = useCallback(() => {
    if (status === "recording") {
      stopRecording();
      return;
    }

    if (status === "transcribing" || status === "sending") return;
    void startRecording();
  }, [startRecording, status, stopRecording]);

  const copyTranscript = useCallback(async () => {
    if (!transcript.trim()) return;
    await navigator.clipboard.writeText(transcript);
  }, [transcript]);

  const clearTranscript = useCallback(() => {
    setTranscript("");
    setError(null);
    setStatus("idle");
  }, []);

  const buttonLabel =
    status === "recording"
      ? "Stop & send"
      : status === "transcribing"
        ? "Transcribing…"
        : status === "sending"
          ? "Sending…"
          : "Speak";

  const statusLabel =
    status === "recording"
      ? "recording"
      : status === "transcribing"
        ? "transcribing"
        : status === "sending"
          ? "submitting"
          : status === "error"
            ? "error"
            : "idle";

  const statusTone: Tone = status === "error" ? "alert" : status === "recording" || status === "sending" ? "warn" : transcript ? "good" : "neutral";

  return (
    <ShellCard className="p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
          <Mic className="h-4 w-4 text-[#45d0ff]" />
          voice bridge
        </div>
        <span className={cn("rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-[0.24em]", toneStyles[statusTone])}>
          {statusLabel}
        </span>
      </div>

      <p className="mt-3 text-[12px] leading-6 text-white/58">
        Speak naturally. The browser records audio, the backend transcribes it with local STT, and the result is submitted into Hermes.
      </p>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={toggleRecording}
          className={cn(
            "inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[11px] uppercase tracking-[0.22em] transition",
            status === "recording"
              ? "border-[#fb7185]/24 bg-[#fb7185]/10 text-[#fecdd3] hover:bg-[#fb7185]/16"
              : "border-[#45d0ff]/18 bg-[#45d0ff]/10 text-[#b9f0ff] hover:bg-[#45d0ff]/16",
          )}
        >
          {status === "recording" ? <MicOff className="h-4 w-4" /> : status === "transcribing" || status === "sending" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Mic className="h-4 w-4" />}
          {buttonLabel}
        </button>

        <button
          type="button"
          onClick={() => void copyTranscript()}
          disabled={!transcript.trim()}
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-[11px] uppercase tracking-[0.22em] text-white/72 transition hover:bg-white/8 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Copy className="h-4 w-4" />
          Copy
        </button>

        <button
          type="button"
          onClick={clearTranscript}
          disabled={!transcript && !error}
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-[11px] uppercase tracking-[0.22em] text-white/72 transition hover:bg-white/8 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Clear
        </button>
      </div>

      <div className="mt-4 rounded-[18px] border border-white/8 bg-black/20 p-4">
        <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.24em] text-white/34">
          <span>transcript</span>
          <span>{shortSessionId(sessionId)}</span>
        </div>
        <div className="mt-3 min-h-[88px] whitespace-pre-wrap text-[13px] leading-6 text-white/74">
          {transcript || "Tap Speak, say the task, and the result will appear here before it opens in chat."}
        </div>
      </div>

      {error ? <div className="mt-3 text-[12px] leading-6 text-[#fecdd3]">{error}</div> : null}

      <div className="mt-3 flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-white/34">
        <Send className="h-3.5 w-3.5" />
        auto-opens chat after submit
      </div>
    </ShellCard>
  );
}

function OrbField() {
  const spokes = useMemo(() => {
    const centre = { x: 50, y: 50 };
    return nodes
      .filter((node) => node.size >= 8)
      .map((node) => ({ x1: centre.x, y1: centre.y, x2: node.x, y2: node.y }));
  }, []);

  return (
    <div className="relative mx-auto flex aspect-square w-full max-w-[520px] items-center justify-center overflow-hidden rounded-full border border-white/8 bg-[radial-gradient(circle_at_50%_50%,rgba(69,208,255,0.18),rgba(124,240,216,0.09)_18%,rgba(251,191,36,0.03)_34%,rgba(0,0,0,0)_72%),linear-gradient(180deg,rgba(10,12,18,0.96),rgba(7,8,12,0.98))] shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_30px_90px_rgba(0,0,0,0.45)]">
      <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
        <defs>
          <radialGradient id="orb-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.18)" />
            <stop offset="30%" stopColor="rgba(69,208,255,0.16)" />
            <stop offset="100%" stopColor="rgba(69,208,255,0)" />
          </radialGradient>
        </defs>
        <circle cx="50" cy="50" r="48" fill="url(#orb-glow)" opacity="0.5" />
        {spokes.map((spoke, index) => (
          <line
            key={`${index}-${spoke.x2}`}
            x1={spoke.x1}
            y1={spoke.y1}
            x2={spoke.x2}
            y2={spoke.y2}
            stroke="rgba(255,255,255,0.12)"
            strokeWidth="0.4"
          />
        ))}
        {nodes.map((node, index) => (
          <g key={`${index}-${node.x}-${node.y}`}>
            <circle cx={node.x} cy={node.y} r={node.size / 2.8} fill={index === 5 ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.24)"} />
            <circle cx={node.x} cy={node.y} r={node.size / 1.8} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="0.35" />
          </g>
        ))}
        <circle cx="50" cy="50" r="12" fill="rgba(255,255,255,0.96)" opacity="0.8" />
        <circle cx="50" cy="50" r="20" fill="none" stroke="rgba(255,255,255,0.16)" strokeWidth="0.5" />
        <circle cx="50" cy="50" r="29" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" />
      </svg>

      <div className="relative z-10 flex flex-col items-center text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[#45d0ff]/20 bg-[#45d0ff]/10 px-3 py-1.5 text-[10px] uppercase tracking-[0.24em] text-[#b9f0ff]">
          <CircleDot className="h-3.5 w-3.5" />
          live command canvas
        </div>
        <div className="text-[clamp(36px,4.7vw,62px)] font-semibold tracking-[-0.09em] text-white">
          Personal OS
        </div>
        <div className="mt-2 max-w-[31rem] text-[14px] leading-7 text-white/60">
          One place for priorities, notes, calendar, tasks, and system status — structured like a cockpit rather than a document dump.
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2 text-[11px] uppercase tracking-[0.24em] text-white/40">
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-white/70">notes</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-white/70">tasks</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-white/70">calendar</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-white/70">systems</span>
        </div>
      </div>
    </div>
  );
}

export default function PersonalOSPage() {
  const { setTitle } = usePageHeader();
  const navigate = useNavigate();

  const launchActions = useMemo(
    () => [
      {
        label: "Open chat",
        detail: "Jump straight into the agent",
        icon: Terminal,
        onClick: () => navigate("/chat"),
      },
      {
        label: "Review sessions",
        detail: "Scan the last context trail",
        icon: MessageSquare,
        onClick: () => navigate("/sessions"),
      },
      {
        label: "Open docs",
        detail: "Read the operating notes",
        icon: NotebookPen,
        onClick: () => navigate("/docs"),
      },
    ],
    [navigate],
  );

  useLayoutEffect(() => {
    setTitle("Personal OS");
    return () => setTitle(null);
  }, [setTitle]);

  return (
    <div className="relative min-h-0 overflow-hidden rounded-none bg-[#05070b] text-white lg:rounded-[28px]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_14%,rgba(69,208,255,0.12),transparent_24%),radial-gradient(circle_at_78%_12%,rgba(124,240,216,0.08),transparent_24%),radial-gradient(circle_at_72%_86%,rgba(251,191,36,0.08),transparent_22%),linear-gradient(180deg,#05070b_0%,#080b11_100%)]" />
      <div className="absolute inset-0 opacity-[0.14] [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:30px_30px]" />

      <div className="relative mx-auto max-w-[1680px] px-4 py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-white/34">
              <span className="rounded-full border border-[#45d0ff]/18 bg-[#45d0ff]/10 px-2.5 py-1 text-[#9ee8ff]">mission control</span>
              <span>personal os</span>
              <span>·</span>
              <span>replacement view</span>
            </div>
            <h1 className="mt-3 text-[clamp(42px,5vw,76px)] font-semibold tracking-[-0.08em] text-white">
              Control Centre
            </h1>
            <p className="mt-3 max-w-3xl text-[15px] leading-7 text-white/62">
              A single cockpit for Richard’s context: priorities, tasks, systems, and the next action.
              Built as a shell now; data wiring can land behind the same layout later.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.24em] text-white/36">
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">router → specialists → feed-up</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">github as source of truth</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">voice on top</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap sm:justify-end">
            <StatTile label="Mode" value="Ready" detail="low-friction, high-signal" tone="good" />
            <StatTile label="Focus" value="78" detail="good enough to push" tone="warn" />
            <StatTile label="Queue" value="12" detail="two need action today" tone="alert" />
            <StatTile label="Latency" value="Low" detail="shell is lightweight" tone="neutral" />
          </div>
        </div>

        <ShellCard className="mb-4 overflow-hidden p-0">
          <div className="border-b border-white/8 bg-black/20 px-4 py-4 sm:px-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div className="min-w-0">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5">
                    <span className="h-3 w-3 rounded-full bg-[#fb7185]" />
                    <span className="h-3 w-3 rounded-full bg-[#fbbf24]" />
                    <span className="h-3 w-3 rounded-full bg-[#4ade80]" />
                  </div>
                  <div className="h-px flex-1 bg-white/8" />
                  <div className="text-[10px] uppercase tracking-[0.28em] text-white/30">desktop chrome</div>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                  <span className="rounded-full border border-[#45d0ff]/18 bg-[#45d0ff]/10 px-2.5 py-1 text-[#9ee8ff]">control centre</span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">github-backed memory</span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">vps execution</span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">voice first</span>
                </div>

                <div className="mt-3 text-[18px] font-semibold tracking-[-0.03em] text-white sm:text-[22px]">
                  Cleaner shell, faster launch.
                </div>
                <p className="mt-2 max-w-2xl text-[13px] leading-6 text-white/58">
                  The GUI now behaves like a desktop operating surface: launch actions on top, context in the middle, and status pinned to the side.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3 xl:min-w-[520px]">
                {launchActions.map((action) => {
                  const Icon = action.icon;
                  return (
                    <button
                      key={action.label}
                      type="button"
                      onClick={action.onClick}
                      className="group flex items-center gap-3 rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-left transition hover:border-[#45d0ff]/30 hover:bg-white/6"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
                        <Icon className="h-4 w-4 text-[#45d0ff]" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">launch</div>
                        <div className="mt-1 text-[13px] font-medium tracking-[-0.01em] text-white">{action.label}</div>
                        <div className="mt-1 text-[11px] leading-4 text-white/50">{action.detail}</div>
                      </div>
                      <ArrowRight className="h-4 w-4 shrink-0 text-white/30 transition group-hover:translate-x-0.5 group-hover:text-white/60" />
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 text-[10px] uppercase tracking-[0.24em] text-white/34 sm:px-5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">notes</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">tasks</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">calendar</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">systems</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">github</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">voice</span>
            </div>
            <div className="flex items-center gap-2 text-white/46">
              <span className="h-1.5 w-1.5 rounded-full bg-[#4ade80]" />
              live shell active
            </div>
          </div>
        </ShellCard>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[280px_minmax(0,1fr)_320px]">
          <ShellCard className="p-4">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
              <Terminal className="h-4 w-4 text-[#45d0ff]" />
              quick signals
            </div>
            <div className="mt-4 space-y-3">
              {quickSignals.map((item) => (
                <SignalRow key={item.label} {...item} />
              ))}
            </div>
          </ShellCard>

          <div className="space-y-4">
            <ShellCard className="p-4 sm:p-5">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0 max-w-2xl">
                  <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                    <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-white/70">today</span>
                    <span>·</span>
                    <span>13:02 GST</span>
                    <span>·</span>
                    <span>last sync 2m ago</span>
                  </div>
                  <h2 className="mt-3 text-[18px] font-semibold tracking-[-0.03em] text-white sm:text-[24px]">
                    Everything visible, nothing buried.
                  </h2>
                  <p className="mt-2 max-w-2xl text-[13px] leading-6 text-white/58 sm:text-[14px]">
                    This replaces the old personal OS landing page with a dashboard-style operating view.
                    It is intentionally more cockpit than notebook.
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <span className="inline-flex items-center gap-2 rounded-full border border-[#4ade80]/18 bg-[#4ade80]/10 px-3 py-1.5 text-[10px] uppercase tracking-[0.24em] text-[#bff7d4]">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    live shell
                  </span>
                  <span className="inline-flex items-center gap-2 rounded-full border border-[#45d0ff]/18 bg-[#45d0ff]/10 px-3 py-1.5 text-[10px] uppercase tracking-[0.24em] text-[#b9f0ff]">
                    <Sparkles className="h-3.5 w-3.5" />
                    hybrid build
                  </span>
                </div>
              </div>

              <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-[1.1fr_0.9fr]">
                <OrbField />

                <ShellCard className="h-full p-4">
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                    <MessageSquare className="h-4 w-4 text-[#45d0ff]" />
                    command feed
                  </div>
                  <div className="mt-4 space-y-3">
                    {feedItems.map((item) => (
                      <FeedRow key={`${item.time}-${item.title}`} {...item} />
                    ))}
                  </div>
                </ShellCard>
              </div>
            </ShellCard>

            <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              {sourceModules.map((module) => {
                const Icon = module.icon;
                return (
                  <ShellCard key={module.title} className="p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                        <Icon className="h-4 w-4 text-[#45d0ff]" />
                        {module.title}
                      </div>
                      <LayerBadge />
                    </div>
                    <div className="mt-3 text-[15px] font-semibold tracking-[-0.02em] text-white">{module.title}</div>
                    <p className="mt-2 text-[12px] leading-6 text-white/58">{module.detail}</p>
                  </ShellCard>
                );
              })}
            </section>

            <ShellCard className="p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                <Layers3 className="h-4 w-4 text-[#45d0ff]" />
                router map
              </div>
              <div className="mt-4 grid gap-3 xl:grid-cols-[1.1fr_0.9fr]">
                <div className="space-y-3">
                  <RouteLayer title="Router" summary="Keyword / regex intake" tone="good" note="`personal-os-skill-router` handles the first split." />
                  <RouteLayer title="Specialists" summary="Focused execution skills" tone="warn" note="`dashboard-shell-design`, `hermes-agent`, `subagent-driven-development`, `writing-plans`." />
                  <RouteLayer title="Domain skills" summary="App and knowledge layers" tone="neutral" note="`obsidian`, `google-workspace`, `github-operations`, `linear`, `telegram-cron-ops`." />
                </div>
                <div className="rounded-[20px] border border-white/8 bg-black/20 p-4">
                  <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">feed-up contract</div>
                  <div className="mt-3 text-[14px] leading-6 text-white/70">
                    Every route should return a compact summary, not a dump:
                  </div>
                  <div className="mt-3 space-y-2 text-[12px] leading-6 text-white/58">
                    <p><span className="text-white">Outcome:</span> what happened.</p>
                    <p><span className="text-white">Key facts:</span> only the important bits.</p>
                    <p><span className="text-white">Next actions:</span> the obvious follow-up.</p>
                    <p><span className="text-white">Risks / blockers:</span> anything that can stop progress.</p>
                  </div>
                </div>
              </div>
            </ShellCard>
          </div>

          <div className="space-y-4">
            <ShellCard className="p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                <Layers3 className="h-4 w-4 text-[#45d0ff]" />
                skill stack
              </div>
              <div className="mt-3 space-y-3">
                {skillLayers.map((layer, index) => (
                  <div key={layer.level} className="rounded-[20px] border border-white/8 bg-black/20 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                          <span className={cn("rounded-full border px-2 py-1", layer.accent)}>{String(index + 1).padStart(2, "0")}</span>
                          {layer.level}
                        </div>
                        <div className="mt-2 text-[15px] font-semibold tracking-[-0.02em] text-white">{layer.title}</div>
                        <p className="mt-1 text-[12px] leading-5 text-white/58">{layer.detail}</p>
                      </div>
                      <div className="h-8 w-8 rounded-2xl border border-white/8 bg-white/5" />
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {layer.chips.map((chip) => (
                        <span key={chip} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] uppercase tracking-[0.22em] text-white/68">
                          {chip}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 rounded-[20px] border border-[#45d0ff]/14 bg-[#45d0ff]/10 p-4 text-[12px] leading-6 text-white/70">
                Router in, concise summary out. The shell should show the handoff without exposing the whole internal prompt chain.
              </div>
            </ShellCard>

            <ShellCard className="p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                <Github className="h-4 w-4 text-[#45d0ff]" />
                github memory
              </div>
              <div className="mt-3 rounded-[20px] border border-white/8 bg-black/20 p-4 text-[13px] leading-6 text-white/72">
                <p className="font-medium text-white">GitHub stays canonical for memory, decisions, and config.</p>
                <p className="mt-3">
                  Treat <span className="text-[#b9f0ff]">richmale-cmyk/context</span> as the source of truth; the shell should surface it, not fork it.
                </p>
              </div>
            </ShellCard>

            <ShellCard className="p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                <Clock3 className="h-4 w-4 text-[#45d0ff]" />
                today’s order
              </div>
              <div className="mt-4 space-y-3">
                <OrderItem index="01" title="Surface priorities" detail="Show the top items first, not a long backlog." />
                <OrderItem index="02" title="Keep the cockpit dark" detail="High contrast, low glare, no spreadsheet feel." />
                <OrderItem index="03" title="Wire live data next" detail="Notes, tasks, calendar, systems, and automations." />
              </div>
            </ShellCard>

            <ShellCard className="p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                <Radar className="h-4 w-4 text-[#45d0ff]" />
                status
              </div>
              <div className="mt-4 space-y-2">
                <MiniStatus label="Capture" value="on" tone="good" />
                <MiniStatus label="Sync" value="stable" tone="neutral" />
                <MiniStatus label="Review" value="pending" tone="warn" />
                <MiniStatus label="Automation" value="ready" tone="good" />
              </div>
            </ShellCard>

            <VoiceConsole />
          </div>
        </section>
      </div>
    </div>
  );
}

function LayerBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] uppercase tracking-[0.22em] text-white/58">
      <Code2 className="h-3.5 w-3.5 text-[#45d0ff]" />
      layer
    </span>
  );
}

function RouteLayer({
  title,
  summary,
  note,
  tone,
}: {
  title: string;
  summary: string;
  note: string;
  tone: Tone;
}) {
  return (
    <div className={cn("rounded-[20px] border p-4", toneStyles[tone])}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">{title}</div>
          <div className="mt-2 text-[15px] font-semibold tracking-[-0.02em] text-white">{summary}</div>
        </div>
        <span className="rounded-full border border-white/10 bg-black/20 px-2 py-1 text-[10px] uppercase tracking-[0.22em] text-white/58">
          {tone}
        </span>
      </div>
      <p className="mt-2 text-[12px] leading-5 text-white/60">{note}</p>
    </div>
  );
}

function OrderItem({ index, title, detail }: { index: string; title: string; detail: string }) {
  return (
    <div className="rounded-[18px] border border-white/8 bg-black/20 p-4">
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-[10px] font-semibold tracking-[0.24em] text-[#b9f0ff]">
          {index}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[14px] font-semibold tracking-[-0.02em] text-white">{title}</div>
          <p className="mt-1 text-[12px] leading-5 text-white/58">{detail}</p>
        </div>
      </div>
    </div>
  );
}

function MiniStatus({ label, value, tone }: { label: string; value: string; tone: Tone }) {
  return (
    <div className={cn("flex items-center justify-between rounded-[18px] border px-4 py-3", toneStyles[tone])}>
      <div className="text-[10px] uppercase tracking-[0.24em] text-white/36">{label}</div>
      <div className="text-[12px] font-medium uppercase tracking-[0.18em] text-white/82">{value}</div>
    </div>
  );
}
