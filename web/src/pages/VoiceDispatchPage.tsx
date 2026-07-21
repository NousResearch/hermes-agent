import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Mic,
  Radio,
  RefreshCw,
  ShieldCheck,
  Volume2,
  WifiOff,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api, type VoiceApprovalChoice, type VoiceRealtimeConfig, type VoiceRunStatus } from "@/lib/api";
import {
  TERMINAL_VOICE_DELEGATE_STATUSES,
  buildVoiceDelegateStatusNotification,
  summarizeVoiceDelegateStatus,
} from "@/lib/voiceDispatchNotifications";
import { float32ToPcm16Base64, pcm16Base64ToFloat32, resampleFloat32 } from "@/lib/voiceAudio";
import { PluginSlot } from "@/plugins";

const XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime";
const MAX_EVENT_LOG = 16;

type ConnectionState = "idle" | "connecting" | "connected" | "error";

type VoiceEvent = {
  id: string;
  timestamp: string;
  type: string;
};

type VoiceTranscriptItem = {
  id: string;
  kind: "assistant" | "user" | "run" | "approval" | "system";
  title?: string;
  body: string;
  timestamp: string;
};

type RealtimeEvent = {
  type?: string;
  error?: { message?: string } | string;
  item?: Record<string, unknown>;
  call_id?: string;
  item_id?: string;
  name?: string;
  arguments?: string | Record<string, unknown>;
  delta?: string;
  [key: string]: unknown;
};

type FunctionCall = {
  callId: string;
  name: string;
  args: Record<string, unknown>;
};

type InputAudioPipeline = {
  stream: MediaStream;
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
};

type OutputAudioPipeline = {
  context: AudioContext;
};

const VOICE_APPROVAL_CHOICES: VoiceApprovalChoice[] = ["once", "session", "deny"];

function voiceApprovalChoices(status: VoiceRunStatus | null): VoiceApprovalChoice[] {
  const rawChoices = status?.approval_request?.choices;
  const allowed = rawChoices?.filter((choice): choice is VoiceApprovalChoice =>
    VOICE_APPROVAL_CHOICES.includes(choice as VoiceApprovalChoice),
  ) ?? VOICE_APPROVAL_CHOICES;
  return Array.from(new Set(allowed.length ? allowed : VOICE_APPROVAL_CHOICES));
}

function approvalDetailRows(status: VoiceRunStatus | null): Array<{ label: string; value: string }> {
  const request = status?.approval_request;
  if (!request) return [];
  const rows: Array<{ label: string; value: string }> = [];
  if (typeof request.description === "string" && request.description.trim()) {
    rows.push({ label: "Description", value: request.description.trim() });
  }
  if (typeof request.command === "string" && request.command.trim()) {
    rows.push({ label: "Command", value: request.command.trim() });
  }
  if (typeof request.pattern_key === "string" && request.pattern_key.trim()) {
    rows.push({ label: "Pattern", value: request.pattern_key.trim() });
  }
  if (Array.isArray(request.pattern_keys) && request.pattern_keys.length > 0) {
    rows.push({ label: "Matched policies", value: request.pattern_keys.join(", ") });
  }
  return rows;
}

function approvalTranscriptBody(status: VoiceRunStatus): string {
  const summary = summarizeVoiceDelegateStatus(status);
  const rows = approvalDetailRows(status);
  if (!rows.length) return summary;
  return [summary, "", ...rows.map((row) => `${row.label}: ${row.value}`)].join("\n");
}

function formatEpochSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "Not minted";
  const millis = value > 10_000_000_000 ? value : value * 1000;
  return new Date(millis).toLocaleString();
}

function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function eventErrorMessage(event: RealtimeEvent): string | null {
  if (!event.error) return null;
  if (typeof event.error === "string") return event.error;
  return event.error.message ?? null;
}

function parseFunctionArgs(value: unknown): Record<string, unknown> {
  if (!value) return {};
  if (typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>;
  if (typeof value !== "string") return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function functionCallFromEvent(event: RealtimeEvent): FunctionCall | null {
  if (event.type === "response.function_call_arguments.done") {
    const name = typeof event.name === "string" ? event.name : "";
    const callId = typeof event.call_id === "string" ? event.call_id : typeof event.item_id === "string" ? event.item_id : "";
    if (!name || !callId) return null;
    return { callId, name, args: parseFunctionArgs(event.arguments) };
  }

  if (event.type === "response.output_item.done" && event.item && event.item.type === "function_call") {
    const item = event.item;
    const name = typeof item.name === "string" ? item.name : "";
    const callId = typeof item.call_id === "string" ? item.call_id : typeof item.id === "string" ? item.id : "";
    if (!name || !callId) return null;
    return { callId, name, args: parseFunctionArgs(item.arguments) };
  }

  return null;
}

function voiceStatus(config: VoiceRealtimeConfig | null, connection: ConnectionState): string {
  if (connection === "connected") return "connected";
  if (connection === "connecting") return "connecting";
  if (connection === "error") return "error";
  if (!config) return "loading";
  return config.enabled ? "ready" : "disabled";
}

function conciseRunStatus(status: VoiceRunStatus | null): string {
  if (!status) return "No Hermes delegate yet";
  return summarizeVoiceDelegateStatus(status);
}

function nowLabel(): string {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function transcriptId(prefix: string): string {
  return `${prefix}:${Date.now()}:${Math.random().toString(36).slice(2)}`;
}

export default function VoiceDispatchPage() {
  const [config, setConfig] = useState<VoiceRealtimeConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [connectError, setConnectError] = useState<string | null>(null);
  const [expiresAt, setExpiresAt] = useState<number | null>(null);
  const [events, setEvents] = useState<VoiceEvent[]>([]);
  const [transcript, setTranscript] = useState<VoiceTranscriptItem[]>([
    {
      id: "welcome",
      kind: "assistant",
      title: "Voice dispatch is ready.",
      body: "Start a voice session, speak the work you want done, and I’ll launch an isolated Hermes delegate without exposing local tools directly to the voice model.",
      timestamp: "Now",
    },
  ]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<VoiceRunStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionUpdateSentRef = useRef(false);
  const configuredRef = useRef(false);
  const handledCallIdsRef = useRef<Set<string>>(new Set());
  const inputAudioRef = useRef<InputAudioPipeline | null>(null);
  const inputAudioStartingRef = useRef(false);
  const outputAudioRef = useRef<OutputAudioPipeline | null>(null);
  const outputNextTimeRef = useRef(0);
  const activeRunIdRef = useRef<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const voiceConnectionSeqRef = useRef(0);
  const responseActiveRef = useRef(false);
  const pendingVoicePromptsRef = useRef<string[]>([]);
  const notifiedRunStatusKeysRef = useRef<Set<string>>(new Set());
  const { setAfterTitle, setEnd } = usePageHeader();

  const status = voiceStatus(config, connection);

  const addEvent = useCallback((type: string) => {
    if (type === "input_audio_buffer.append" || type === "response.output_audio.delta") {
      return;
    }
    setEvents((prev) => [
      {
        id: `${Date.now()}:${Math.random().toString(36).slice(2)}`,
        timestamp: new Date().toLocaleTimeString(),
        type,
      },
      ...prev,
    ].slice(0, MAX_EVENT_LOG));
  }, []);

  const addTranscript = useCallback((item: Omit<VoiceTranscriptItem, "id" | "timestamp"> & { id?: string; timestamp?: string }) => {
    setTranscript((prev) => [
      ...prev,
      {
        id: item.id ?? transcriptId(item.kind),
        kind: item.kind,
        title: item.title,
        body: item.body,
        timestamp: item.timestamp ?? nowLabel(),
      },
    ]);
  }, []);

  const setActiveRun = useCallback((runId: string | null) => {
    activeRunIdRef.current = runId;
    setActiveRunId(runId);
  }, []);

  const loadConfig = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getVoiceRealtimeConfig()
      .then((nextConfig) => {
        setConfig(nextConfig);
      })
      .catch((err) => {
        setError(describeError(err));
        setConfig(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const stopInputAudio = useCallback(() => {
    const pipeline = inputAudioRef.current;
    inputAudioRef.current = null;
    if (!pipeline) return;
    pipeline.processor.disconnect();
    pipeline.source.disconnect();
    pipeline.stream.getTracks().forEach((track) => track.stop());
    void pipeline.context.close().catch(() => undefined);
  }, []);

  const stopOutputAudio = useCallback(() => {
    const pipeline = outputAudioRef.current;
    outputAudioRef.current = null;
    outputNextTimeRef.current = 0;
    if (!pipeline) return;
    void pipeline.context.close().catch(() => undefined);
  }, []);

  const stopAudio = useCallback(() => {
    stopInputAudio();
    stopOutputAudio();
  }, [stopInputAudio, stopOutputAudio]);

  const sendFunctionOutput = useCallback((ws: WebSocket, callId: string, result: unknown) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: callId,
        output: JSON.stringify(result),
      },
    }));
    responseActiveRef.current = true;
    ws.send(JSON.stringify({ type: "response.create" }));
    addEvent("dashboard.function_output.sent");
  }, [addEvent]);

  const sendRealtimeVoicePrompt = useCallback((prompt: string): boolean => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !configuredRef.current) {
      addEvent("delegate.notification.skipped");
      return false;
    }

    ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: prompt }],
      },
    }));
    responseActiveRef.current = true;
    ws.send(JSON.stringify({ type: "response.create" }));
    addEvent("delegate.notification.sent");
    return true;
  }, [addEvent]);

  const flushPendingVoicePrompts = useCallback(() => {
    if (responseActiveRef.current) return;
    const nextPrompt = pendingVoicePromptsRef.current.shift();
    if (!nextPrompt) return;
    window.setTimeout(() => {
      if (!responseActiveRef.current) {
        sendRealtimeVoicePrompt(nextPrompt);
      } else {
        pendingVoicePromptsRef.current.unshift(nextPrompt);
      }
    }, 0);
  }, [sendRealtimeVoicePrompt]);

  const queueRealtimeVoicePrompt = useCallback((prompt: string): boolean => {
    if (responseActiveRef.current) {
      pendingVoicePromptsRef.current.push(prompt);
      addEvent("delegate.notification.queued");
      return true;
    }
    return sendRealtimeVoicePrompt(prompt);
  }, [addEvent, sendRealtimeVoicePrompt]);

  const maybeNotifyVoiceDelegateStatus = useCallback((nextStatus: VoiceRunStatus) => {
    const notification = buildVoiceDelegateStatusNotification(nextStatus);
    if (!notification || notifiedRunStatusKeysRef.current.has(notification.key)) return;
    if (!queueRealtimeVoicePrompt(notification.voicePrompt)) return;
    notifiedRunStatusKeysRef.current.add(notification.key);
    addTranscript({
      kind: notification.kind,
      title: notification.title,
      body: notification.kind === "approval" ? approvalTranscriptBody(nextStatus) : notification.transcriptBody,
    });
  }, [addTranscript, queueRealtimeVoicePrompt]);

  const executeVoiceTool = useCallback(async (call: FunctionCall): Promise<unknown> => {
    addEvent(`tool.${call.name}.called`);
    if (call.name === "start_delegate" || call.name === "start_hermes_run") {
      const rawInput = typeof call.args.input === "string" ? call.args.input : typeof call.args.objective === "string" ? call.args.objective : "";
      const input = rawInput.trim();
      if (!input) return { ok: false, error: "Missing required input." };
      addTranscript({ kind: "user", title: "Delegate request", body: input });
      const started = await api.startVoiceDelegate({ input });
      const delegateId = started.delegate_id ?? started.run_id;
      notifiedRunStatusKeysRef.current.clear();
      setActiveRun(delegateId);
      const nextStatus = await api.getVoiceDelegate(delegateId);
      setRunStatus(nextStatus);
      const initialNotification = buildVoiceDelegateStatusNotification(nextStatus);
      if (initialNotification) notifiedRunStatusKeysRef.current.add(initialNotification.key);
      addTranscript({
        kind: "run",
        title: "Hermes delegate started",
        body: `${delegateId} is ${nextStatus.status}. The delegate is handling the work locally while the voice session stays free.`,
      });
      return { ok: true, delegate_id: delegateId, run_id: delegateId, status: nextStatus.status, events: nextStatus.events ?? [] };
    }

    if (call.name === "get_delegate_status" || call.name === "get_hermes_run") {
      const delegateId = typeof call.args.delegate_id === "string" && call.args.delegate_id.trim() ? call.args.delegate_id.trim() : activeRunIdRef.current;
      if (!delegateId) return { ok: false, error: "No active Hermes delegate." };
      const nextStatus = await api.getVoiceDelegate(delegateId);
      setRunStatus(nextStatus);
      const notification = buildVoiceDelegateStatusNotification(nextStatus);
      if (notification) {
        notifiedRunStatusKeysRef.current.add(notification.key);
        addTranscript({
          kind: notification.kind,
          title: notification.title,
          body: notification.kind === "approval" ? approvalTranscriptBody(nextStatus) : notification.transcriptBody,
        });
      }
      return { ok: true, delegate: nextStatus, run: nextStatus, summary: conciseRunStatus(nextStatus), events: nextStatus.events ?? [] };
    }


    if (call.name === "stop_delegate" || call.name === "stop_hermes_run") {
      const delegateId = typeof call.args.delegate_id === "string" && call.args.delegate_id.trim() ? call.args.delegate_id.trim() : activeRunIdRef.current;
      if (!delegateId) return { ok: false, error: "No active Hermes delegate." };
      const stopped = await api.stopVoiceDelegate(delegateId);
      setRunStatus(stopped);
      const stopNotification = buildVoiceDelegateStatusNotification(stopped);
      if (stopNotification) notifiedRunStatusKeysRef.current.add(stopNotification.key);
      addTranscript({ kind: "run", title: "Delegate stop requested", body: conciseRunStatus(stopped) });
      return { ok: true, delegate: stopped, run: stopped };
    }

    return { ok: false, error: `Unsupported voice dispatch tool: ${call.name}` };
  }, [addEvent, addTranscript, setActiveRun]);

  const playOutputAudio = useCallback((base64: string, sampleRate: number) => {
    if (!base64) return;
    const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return;
    const pipeline = outputAudioRef.current ?? { context: new AudioContextCtor({ sampleRate }) };
    outputAudioRef.current = pipeline;
    const samples = new Float32Array(pcm16Base64ToFloat32(base64));
    const buffer = pipeline.context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = pipeline.context.createBufferSource();
    source.buffer = buffer;
    source.connect(pipeline.context.destination);
    const startAt = Math.max(pipeline.context.currentTime, outputNextTimeRef.current);
    source.start(startAt);
    outputNextTimeRef.current = startAt + buffer.duration;
    void pipeline.context.resume().catch(() => undefined);
  }, []);

  const startInputAudio = useCallback(async (ws: WebSocket, inputRate: number, connectionSeq: number) => {
    if (inputAudioStartingRef.current || inputAudioRef.current) {
      addEvent("microphone.start.skipped");
      return;
    }

    const isCurrentSession = () => (
      wsRef.current === ws
      && voiceConnectionSeqRef.current === connectionSeq
      && ws.readyState === WebSocket.OPEN
      && configuredRef.current
    );

    inputAudioStartingRef.current = true;
    let stream: MediaStream | null = null;
    let context: AudioContext | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    let processor: ScriptProcessorNode | null = null;

    const cleanupPendingPipeline = () => {
      if (stream && inputAudioRef.current?.stream === stream) {
        inputAudioRef.current = null;
      }
      try {
        processor?.disconnect();
      } catch {
        // Ignore cleanup failures from already-disconnected nodes.
      }
      try {
        source?.disconnect();
      } catch {
        // Ignore cleanup failures from already-disconnected nodes.
      }
      stream?.getTracks().forEach((track) => track.stop());
      if (context) {
        void context.close().catch(() => undefined);
      }
    };

    try {
      stopInputAudio();
      const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextCtor || !navigator.mediaDevices?.getUserMedia) {
        throw new Error("Browser microphone capture is not available.");
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (!isCurrentSession()) {
        cleanupPendingPipeline();
        return;
      }
      context = new AudioContextCtor();
      source = context.createMediaStreamSource(stream);
      processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        if (!isCurrentSession()) return;
        const input = event.inputBuffer.getChannelData(0);
        const mono = new Float32Array(input);
        const resampled = resampleFloat32(mono, context?.sampleRate ?? inputRate, inputRate);
        const audio = float32ToPcm16Base64(resampled);
        ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio }));
        const output = event.outputBuffer.getChannelData(0);
        output.fill(0);
      };
      source.connect(processor);
      processor.connect(context.destination);
      inputAudioRef.current = { stream, context, source, processor };
      await context.resume();
      if (!isCurrentSession()) {
        if (inputAudioRef.current?.stream === stream) {
          inputAudioRef.current = null;
        }
        cleanupPendingPipeline();
        return;
      }
      addEvent("microphone.started");
    } catch (err) {
      cleanupPendingPipeline();
      throw err;
    } finally {
      if (voiceConnectionSeqRef.current === connectionSeq) {
        inputAudioStartingRef.current = false;
      }
    }
  }, [addEvent, stopInputAudio]);

  const disconnect = useCallback(() => {
    voiceConnectionSeqRef.current += 1;
    sessionUpdateSentRef.current = false;
    configuredRef.current = false;
    inputAudioStartingRef.current = false;
    responseActiveRef.current = false;
    pendingVoicePromptsRef.current = [];
    handledCallIdsRef.current.clear();
    stopAudio();
    if (wsRef.current) {
      wsRef.current.close(1000, "dashboard disconnect");
      wsRef.current = null;
    }
    setConnection("idle");
    addEvent("dashboard.disconnected");
  }, [addEvent, stopAudio]);

  const connect = useCallback(async () => {
    if (!config) return;
    if (!config.enabled) {
      setConnectError("Realtime voice dispatch is disabled in config.");
      return;
    }

    disconnect();
    const connectionSeq = voiceConnectionSeqRef.current + 1;
    voiceConnectionSeqRef.current = connectionSeq;
    setConnectError(null);
    setConnection("connecting");
    addTranscript({ kind: "system", title: "Starting voice session", body: "Minting a short-lived xAI realtime secret through Hermes." });
    addEvent("dashboard.client_secret.requested");

    try {
      const session = await api.createVoiceRealtimeClientSecret();
      const ephemeralSecret = session.client_secret.value;
      setExpiresAt(session.client_secret.expires_at);
      addEvent("dashboard.client_secret.minted");

      configuredRef.current = false;
      sessionUpdateSentRef.current = false;
      responseActiveRef.current = false;
      pendingVoicePromptsRef.current = [];
      const ws = new WebSocket(`${XAI_REALTIME_URL}?model=${encodeURIComponent(session.model)}`, [
        "realtime",
        `openai-insecure-api-key.${ephemeralSecret}`,
        "openai-beta.realtime-v1",
      ]);
      wsRef.current = ws;

      const send = (message: Record<string, unknown>) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify(message));
        addEvent(String(message.type ?? "dashboard.sent"));
      };

      const configureSession = () => {
        if (sessionUpdateSentRef.current) return;
        sessionUpdateSentRef.current = true;
        send({
          type: "session.update",
          session: {
            voice: session.voice,
            instructions:
              "You are Hermes Voice Dispatch. You are a conversational controller, not the worker. When the user asks for work to be done, start a secure isolated Hermes delegate using the supplied dispatch tools. Do not claim direct shell, file, browser, MCP, credential, or local-machine access. The delegate performs work through Hermes' normal tool and approval system while the main Hermes session stays free. Ask a brief clarification only when required. Give concise progress updates from delegate status/events. Surface approval requests clearly. Approval requires an explicit Hermes UI click in this page; you cannot approve actions by tool call, voice narration, or user speech alone. Tell the user to review the approval card and choose once, session, or deny. Never invent tool results, never say work completed until delegate status confirms it, never bypass approval, never request or repeat secrets, and stop the delegate if the user asks.",
            tools: config.tools,
            tool_choice: "auto",
            audio: {
              input: { format: { type: "audio/pcm", rate: config.audio.input_rate } },
              output: { format: { type: "audio/pcm", rate: config.audio.output_rate } },
            },
            turn_detection: config.turn_detection,
          },
        });
      };

      ws.onopen = () => {
        if (wsRef.current !== ws) return;
        setConnection("connected");
        addEvent("websocket.open");
      };

      ws.onmessage = (event) => {
        if (wsRef.current !== ws) return;
        let message: RealtimeEvent;
        try {
          message = JSON.parse(String(event.data)) as RealtimeEvent;
        } catch {
          addEvent("websocket.message.unparseable");
          return;
        }

        const type = message.type ?? "websocket.message";
        addEvent(type);

        if (type === "response.created") {
          responseActiveRef.current = true;
        }

        if (type === "response.done" || type === "response.cancelled" || type === "response.failed") {
          responseActiveRef.current = false;
          flushPendingVoicePrompts();
        }

        if ((type === "conversation.created" || type === "session.created") && !configuredRef.current) {
          configureSession();
        }

        if (type === "session.updated") {
          configuredRef.current = true;
          if (inputAudioStartingRef.current || inputAudioRef.current) {
            addEvent("session.updated.duplicate_ignored");
          } else {
            addTranscript({ kind: "assistant", title: "Voice session live", body: "I’m listening. Tell me what you want Hermes to do." });
            void startInputAudio(ws, config.audio.input_rate, connectionSeq).catch((err) => {
              setConnection("error");
              setConnectError(describeError(err));
              addEvent("microphone.failed");
            });
          }
        }

        if (type === "response.output_audio.delta" && typeof message.delta === "string") {
          playOutputAudio(message.delta, config.audio.output_rate);
        }

        const call = functionCallFromEvent(message);
        if (call && !handledCallIdsRef.current.has(call.callId)) {
          handledCallIdsRef.current.add(call.callId);
          void executeVoiceTool(call)
            .then((result) => sendFunctionOutput(ws, call.callId, result))
            .catch((err) => sendFunctionOutput(ws, call.callId, { ok: false, error: describeError(err) }));
        }

        if (type === "error") {
          responseActiveRef.current = false;
          setConnection("error");
          setConnectError(eventErrorMessage(message) ?? "xAI realtime session returned an error.");
        }
      };

      ws.onerror = () => {
        if (wsRef.current !== ws) return;
        setConnection("error");
        setConnectError("WebSocket error while connecting to xAI realtime.");
        addEvent("websocket.error");
      };

      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        voiceConnectionSeqRef.current += 1;
        wsRef.current = null;
        sessionUpdateSentRef.current = false;
        configuredRef.current = false;
        inputAudioStartingRef.current = false;
        responseActiveRef.current = false;
        pendingVoicePromptsRef.current = [];
        stopAudio();
        setConnection((current) => (current === "error" ? current : "idle"));
        addEvent("websocket.closed");
      };
    } catch (err) {
      setConnection("error");
      setConnectError(describeError(err));
      addEvent("dashboard.client_secret.failed");
    }
  }, [addEvent, addTranscript, config, disconnect, executeVoiceTool, flushPendingVoicePrompts, playOutputAudio, sendFunctionOutput, startInputAudio, stopAudio]);

  useEffect(() => {
    const timer = window.setTimeout(loadConfig, 0);
    return () => window.clearTimeout(timer);
  }, [loadConfig]);

  useEffect(() => disconnect, [disconnect]);

  useEffect(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (!activeRunId) return undefined;

    const refreshDelegateStatus = () => {
      void api.getVoiceDelegate(activeRunId).then((nextStatus) => {
        setRunStatus(nextStatus);
        maybeNotifyVoiceDelegateStatus(nextStatus);
        if (TERMINAL_VOICE_DELEGATE_STATUSES.has(nextStatus.status) && pollTimerRef.current !== null) {
          window.clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
      }).catch((err) => {
        setConnectError(describeError(err));
      });
    };

    refreshDelegateStatus();
    pollTimerRef.current = window.setInterval(refreshDelegateStatus, 2_000);

    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [activeRunId, maybeNotifyVoiceDelegateStatus]);

  useLayoutEffect(() => {
    setAfterTitle(
      <Badge
        tone={status === "connected" || status === "ready" ? "success" : status === "error" ? "destructive" : "outline"}
        className="rounded-full border-white/10 bg-[#121214] px-3 py-1 font-mono-ui text-xs uppercase tracking-[0.16em] text-white"
      >
        {status}
      </Badge>,
    );
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          type="button"
          size="sm"
          outlined
          onClick={loadConfig}
          disabled={loading}
          prefix={loading ? <Spinner /> : <RefreshCw />}
          className="rounded-full border-white/10 bg-white/[0.06] px-4 text-white backdrop-blur-xl hover:bg-white/[0.1] disabled:text-white/35"
        >
          Refresh
        </Button>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [loadConfig, loading, setAfterTitle, setEnd, status]);

  const toolNames = useMemo(() => config?.tools.map((tool) => tool.name) ?? [], [config]);
  const canConnect = Boolean(config?.enabled) && connection !== "connecting" && connection !== "connected";

  const approveFromUi = useCallback(async (choice: "once" | "session" | "deny") => {
    if (!activeRunId) return;
    const approved = await api.approveVoiceDelegate(activeRunId, { choice });
    const nextStatus = await api.getVoiceDelegate(activeRunId);
    setRunStatus(nextStatus);
    addTranscript({ kind: "approval", title: "Approval response", body: `${approved.choice} sent to the delegate. ${nextStatus.status}.` });
  }, [activeRunId, addTranscript]);

  const stopActiveRun = useCallback(async () => {
    if (!activeRunId) return;
    const stopped = await api.stopVoiceDelegate(activeRunId);
    setRunStatus(stopped);
    addTranscript({ kind: "run", title: "Delegate stop requested", body: conciseRunStatus(stopped) });
  }, [activeRunId, addTranscript]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <PluginSlot name="voice:top" />

      <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:gap-3">
        <section className="flex min-h-[32rem] min-w-0 flex-1 flex-col overflow-hidden rounded-lg border border-border bg-background-base/60">
          <header className="flex shrink-0 flex-col gap-3 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded border border-current/25 bg-background-base text-midground">
                <Mic className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <h2 className="font-mondwest text-display text-[1.375rem] font-bold leading-none tracking-[0.06em] text-midground">
                  Voice Dispatch
                </h2>
                <p className="mt-1 truncate text-xs text-text-secondary">
                  Voice chat interface for launching isolated Hermes delegates safely.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={status === "connected" || status === "ready" ? "success" : status === "error" ? "destructive" : "outline"} className="shrink-0 text-xs">
                {status}
              </Badge>
              <Badge tone="secondary" className="shrink-0 font-mono-ui text-xs">
                {config?.model ?? "grok-voice-latest"}
              </Badge>
            </div>
          </header>

          <main className="min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-5">
            <div className="mx-auto flex max-w-3xl flex-col gap-3">
              {error && (
                <div className="border border-destructive/30 bg-destructive/[0.06] p-3 text-sm text-destructive">
                  <div className="mb-1 flex items-center gap-2 font-medium"><AlertTriangle className="h-4 w-4" />Config error</div>
                  {error}
                </div>
              )}
              {connectError && (
                <div className="border border-warning/40 bg-warning/10 p-3 text-sm text-warning">
                  <div className="mb-1 flex items-center gap-2 font-medium"><AlertTriangle className="h-4 w-4" />Voice session issue</div>
                  {connectError}
                </div>
              )}

              {transcript.map((item) => {
                const isUser = item.kind === "user";
                const isRun = item.kind === "run" || item.kind === "approval";
                return (
                  <div key={item.id} className={isUser ? "flex justify-end" : "flex justify-start"}>
                    <div
                      className={
                        isUser
                          ? "max-w-[78%] border border-current/25 bg-midground px-4 py-3 text-background-base"
                          : isRun
                            ? "w-full border border-border bg-background-base/80 p-4"
                            : "max-w-[82%] border border-border bg-background-base/50 px-4 py-3"
                      }
                    >
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-2">
                          {!isUser && (
                            <span className={isRun ? "grid h-7 w-7 shrink-0 place-items-center rounded border border-current/25 bg-midground text-background-base" : "grid h-7 w-7 shrink-0 place-items-center rounded border border-border bg-background-base text-midground"}>
                              {item.kind === "approval" ? <ShieldCheck className="h-3.5 w-3.5" /> : item.kind === "run" ? <Radio className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
                            </span>
                          )}
                          <span className={isUser ? "truncate text-xs font-mondwest tracking-[0.08em] text-background-base/80" : "truncate text-xs font-mondwest tracking-[0.08em] text-midground"}>
                            {item.title ?? (isUser ? "You" : "Hermes")}
                          </span>
                        </div>
                        <span className={isUser ? "shrink-0 text-[0.6875rem] text-background-base/55" : "shrink-0 text-[0.6875rem] text-text-tertiary"}>{item.timestamp}</span>
                      </div>
                      <p className={isUser ? "whitespace-pre-wrap text-sm leading-relaxed text-background-base" : "whitespace-pre-wrap text-sm leading-relaxed text-text-secondary"}>{item.body}</p>
                      {item.kind === "approval" && activeRunId && runStatus?.status === "waiting_for_approval" && (
                        <div className="mt-4 border border-warning/40 bg-warning/10 p-3 text-xs text-text-secondary">
                          <div className="mb-2 font-mondwest tracking-[0.08em] text-warning">Explicit human approval required</div>
                          <p className="mb-3 leading-relaxed">Grok can explain this request, but only your click here can approve or deny it.</p>
                          <div className="space-y-2">
                            {approvalDetailRows(runStatus).map((row) => (
                              <div key={row.label} className="grid gap-1 border border-border/60 bg-background-base/60 p-2 sm:grid-cols-[6rem_1fr]">
                                <span className="font-mondwest tracking-[0.08em] text-text-tertiary">{row.label}</span>
                                <span className={row.label === "Command" ? "break-all font-mono-ui text-foreground" : "text-foreground"}>{row.value}</span>
                              </div>
                            ))}
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                          {voiceApprovalChoices(runStatus).map((choice) => (
                            <Button
                              key={choice}
                              type="button"
                              size="sm"
                              outlined={choice === "deny"}
                              onClick={() => void approveFromUi(choice)}
                              className="font-mondwest text-xs tracking-[0.08em]"
                            >
                              {choice}
                            </Button>
                          ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </main>

          <footer className="shrink-0 border-t border-border bg-background-base/80 px-3 py-3 sm:px-4">
            <div className="mx-auto max-w-3xl border border-border bg-background-base p-3">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <div className="font-mondwest text-sm tracking-[0.08em] text-midground">
                    {connection === "connected" ? "Listening for dispatch" : connection === "connecting" ? "Opening realtime session" : config?.enabled ? "Ready for voice" : "Voice disabled"}
                  </div>
                  <div className="mt-1 truncate text-xs text-text-secondary">
                    {connection === "connected"
                      ? "Speak naturally. Grok can start delegates; approvals still require a UI click."
                      : config?.enabled
                        ? "Start a session to launch Hermes delegates through Grok Voice."
                        : "Enable voice.realtime.enabled and set XAI_API_KEY to start."}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {connection === "connected" ? (
                    <Button type="button" outlined onClick={disconnect} prefix={<WifiOff />}>
                      End session
                    </Button>
                  ) : (
                    <Button type="button" onClick={connect} disabled={!canConnect || loading} prefix={connection === "connecting" ? <Spinner /> : <Mic />}>
                      {connection === "connecting" ? "Connecting" : "Start voice"}
                    </Button>
                  )}
                </div>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs text-text-secondary">
                <div className="border border-border bg-background-base/50 px-3 py-2 font-mono-ui">{config?.provider ?? "xai"}</div>
                <div className="border border-border bg-background-base/50 px-3 py-2 font-mono-ui">{config?.voice ?? "eve"}</div>
                <div className="border border-border bg-background-base/50 px-3 py-2 font-mono-ui">{toolNames.length} tools</div>
              </div>
            </div>
          </footer>
        </section>

        <aside className="flex min-h-0 flex-col gap-2 overflow-y-auto pr-1 lg:w-80 lg:shrink-0">
          <div className="border border-border bg-background-base/60 p-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h3 className="font-mondwest text-sm tracking-[0.08em] text-midground">Dispatch context</h3>
              <Button type="button" size="sm" outlined onClick={loadConfig} disabled={loading} prefix={loading ? <Spinner /> : <RefreshCw />}>
                Refresh
              </Button>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between border border-border bg-background-base/50 px-3 py-2">
                <span className="text-text-secondary">Realtime</span>
                <span className="flex items-center gap-1 font-medium text-foreground">{config?.enabled ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <AlertTriangle className="h-3.5 w-3.5 text-warning" />}{config?.enabled ? "Enabled" : "Disabled"}</span>
              </div>
              <div className="flex items-center justify-between border border-border bg-background-base/50 px-3 py-2">
                <span className="text-text-secondary">Connection</span>
                <span className="font-medium text-foreground">{connection}</span>
              </div>
              <div className="flex items-center justify-between border border-border bg-background-base/50 px-3 py-2">
                <span className="text-text-secondary">Secret expiry</span>
                <span className="max-w-[9.5rem] truncate font-medium text-foreground">{formatEpochSeconds(expiresAt)}</span>
              </div>
            </div>
          </div>

          <div className="border border-border bg-background-base/60 p-4">
            <div className="mb-3 flex items-center gap-2">
              <Radio className="h-4 w-4 text-midground" />
              <h3 className="font-mondwest text-sm tracking-[0.08em] text-midground">Active delegate</h3>
            </div>
            <div className="space-y-2 text-sm text-text-secondary">
              <div>Delegate ID: <span className="font-mono-ui text-xs text-foreground">{activeRunId ?? "—"}</span></div>
              <div>Status: <span className="font-medium text-foreground">{runStatus?.status ?? "—"}</span></div>
              {runStatus?.last_event && <div>Last event: <span className="font-medium text-foreground">{runStatus.last_event}</span></div>}
              {runStatus?.status === "waiting_for_approval" && (
                <div className="border border-warning/40 bg-warning/10 p-3">
                  <div className="mb-2 font-mondwest text-xs tracking-[0.08em] text-warning">Approval pending</div>
                  <div className="space-y-2">
                    {approvalDetailRows(runStatus).map((row) => (
                      <div key={row.label}>
                        <div className="font-mondwest text-[0.6875rem] tracking-[0.08em] text-text-tertiary">{row.label}</div>
                        <div className={row.label === "Command" ? "break-all font-mono-ui text-xs text-foreground" : "text-xs text-foreground"}>{row.value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {voiceApprovalChoices(runStatus).map((choice) => (
                      <Button
                        key={choice}
                        type="button"
                        size="sm"
                        outlined={choice === "deny"}
                        onClick={() => void approveFromUi(choice)}
                        className="font-mondwest text-xs tracking-[0.08em]"
                      >
                        {choice}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
              {runStatus?.events && runStatus.events.length > 0 && (
                <div className="border border-border bg-background-base/50 p-3">
                  <div className="mb-2 font-mondwest text-xs tracking-[0.08em] text-midground">Recent delegate events</div>
                  <div className="space-y-1">
                    {runStatus.events.slice(-5).map((event, index) => (
                      <div key={index} className="flex items-center justify-between gap-2 text-xs">
                        <span className="font-mono-ui text-text-secondary">{typeof event.event === "string" ? event.event : "event"}</span>
                        {typeof event.tool === "string" && <span className="truncate text-text-tertiary">{event.tool}</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {runStatus?.output && <div className="border border-border bg-background-base/50 p-3 text-foreground">{runStatus.output}</div>}
              {runStatus?.error && <div className="border border-destructive/30 bg-destructive/[0.06] p-3 text-destructive">{runStatus.error}</div>}
            </div>
            {activeRunId && runStatus?.status && !TERMINAL_VOICE_DELEGATE_STATUSES.has(runStatus.status) && (
              <Button type="button" outlined onClick={() => void stopActiveRun()} className="mt-4 w-full">
                Stop delegate
              </Button>
            )}
          </div>

          <div className="border border-border bg-background-base/60 p-4">
            <div className="mb-3 flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-midground" />
              <h3 className="font-mondwest text-sm tracking-[0.08em] text-midground">Allowed delegate tools</h3>
            </div>
            <div className="space-y-2">
              {config?.tools.map((tool) => (
                <div key={tool.name} className="border border-border bg-background-base/50 px-3 py-2">
                  <div className="font-mono-ui text-[0.6875rem] uppercase tracking-[0.08em] text-foreground">{tool.name}</div>
                  <p className="mt-1 text-xs leading-relaxed text-text-secondary">{tool.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="border border-border bg-background-base/60 p-4">
            <div className="mb-3 flex items-center gap-2">
              <Volume2 className="h-4 w-4 text-midground" />
              <h3 className="font-mondwest text-sm tracking-[0.08em] text-midground">Diagnostics</h3>
            </div>
            <div className="max-h-48 space-y-2 overflow-auto">
              {events.length ? events.map((event) => (
                <div key={event.id} className="flex items-center justify-between gap-3 border border-border bg-background-base/50 px-3 py-2 text-xs">
                  <span className="font-mono-ui text-text-secondary">{event.type}</span>
                  <span className="text-text-tertiary">{event.timestamp}</span>
                </div>
              )) : <div className="text-sm text-text-secondary">No realtime events yet.</div>}
            </div>
          </div>
        </aside>
      </div>

      <PluginSlot name="voice:bottom" />
    </div>
  );
}
