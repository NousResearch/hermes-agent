import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { api, type VoiceToolRequest } from "@/lib/api";

type CallStatus = "idle" | "requesting" | "connecting" | "live" | "ending" | "error";

type LogKind = "system" | "user" | "rolly" | "tool" | "error";

interface LogEntry {
  id: string;
  kind: LogKind;
  text: string;
  timestamp: string;
  elapsedMs: number | null;
}

function logId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function eventText(event: unknown): string | null {
  if (!event || typeof event !== "object") return null;
  const obj = event as Record<string, unknown>;
  const direct = obj.transcript ?? obj.text ?? obj.delta;
  return typeof direct === "string" && direct.trim() ? direct.trim() : null;
}

function formatClock(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatElapsed(ms: number | null): string {
  if (ms === null) return "+0.0s";
  return `+${(ms / 1000).toFixed(1)}s`;
}

export default function VoiceCallPage() {
  const [status, setStatus] = useState<CallStatus>("idle");
  const [muted, setMuted] = useState(false);
  const [micInfo, setMicInfo] = useState("Mic: not connected");
  const [micLevel, setMicLevel] = useState(0);
  const [inputDevices, setInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedInputId, setSelectedInputId] = useState("");
  const [speaker, setSpeaker] = useState(() => window.localStorage.getItem("rolly.voice.user") || "deniz");
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: logId(),
      kind: "system",
      text: "Prototype: browser WebRTC to realtime voice, with backend tool bridge for research.",
      timestamp: new Date().toISOString(),
      elapsedMs: null,
    },
  ]);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const dataRef = useRef<RTCDataChannel | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const micMonitorRafRef = useRef<number | null>(null);
  const callIdRef = useRef(`voice-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  const callStartedAtRef = useRef<number | null>(null);
  const eventSeqRef = useRef(0);
  const pendingTranscriptSavesRef = useRef<Promise<unknown>[]>([]);
  const callSeqRef = useRef(0);
  const [saveStatus, setSaveStatus] = useState("Not saving yet");
  const [lastSavePath, setLastSavePath] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<string | null>(null);

  const addLog = useCallback((kind: LogKind, text: string) => {
    const now = Date.now();
    const startedAt = callStartedAtRef.current;
    setLogs((prev) => [
      ...prev.slice(-120),
      { id: logId(), kind, text, timestamp: new Date(now).toISOString(), elapsedMs: startedAt === null ? null : now - startedAt },
    ]);
  }, []);

  const persistTranscript = useCallback(
    (role: string, text: string, eventType = "transcript", metadata: Record<string, unknown> = {}) => {
      const now = Date.now();
      const startedAt = callStartedAtRef.current;
      const sequence = ++eventSeqRef.current;
      const save = api.saveVoiceTranscript(
        {
          call_id: callIdRef.current,
          role,
          text,
          event_type: eventType,
          user: speaker,
          timestamp: new Date(now).toISOString(),
          sequence,
          elapsed_ms: startedAt === null ? undefined : now - startedAt,
          metadata,
        },
        speaker,
      )
        .then((resp) => {
          setLastSavePath(resp.path);
          setSaveStatus(`Saved event #${sequence}`);
          return resp;
        })
        .catch((exc) => {
          const message = exc instanceof Error ? exc.message : String(exc);
          setSaveStatus(`Save failed: ${message}`);
          addLog("error", `Transcript save failed: ${message}`);
        })
        .finally(() => {
          pendingTranscriptSavesRef.current = pendingTranscriptSavesRef.current.filter((item) => item !== save);
        });
      pendingTranscriptSavesRef.current.push(save);
      return save;
    },
    [addLog, speaker],
  );

  const refreshInputDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    setInputDevices(inputs);
    if (!selectedInputId && inputs.some((device) => device.deviceId === "default")) {
      setSelectedInputId("default");
    }
  }, [selectedInputId]);

  const stopMicMonitor = useCallback(() => {
    if (micMonitorRafRef.current !== null) {
      window.cancelAnimationFrame(micMonitorRafRef.current);
    }
    micMonitorRafRef.current = null;
    void audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    setMicLevel(0);
    setMicInfo("Mic: not connected");
  }, []);

  const startMicMonitor = useCallback((stream: MediaStream) => {
    stopMicMonitor();
    const track = stream.getAudioTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    setMicInfo(`Mic: ${track?.label || "unknown"} (${settings.sampleRate ?? "?"} Hz)`);

    const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) {
      addLog("error", "Browser does not expose AudioContext; cannot meter microphone input.");
      return;
    }

    const context = new AudioContextCtor();
    audioContextRef.current = context;
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (const sample of samples) {
        const centered = sample - 128;
        sum += centered * centered;
      }
      setMicLevel(Math.min(100, Math.round((Math.sqrt(sum / samples.length) / 128) * 160)));
      micMonitorRafRef.current = window.requestAnimationFrame(tick);
    };
    tick();
  }, [addLog, stopMicMonitor]);

  const enableMicList = useCallback(async () => {
    setError(null);
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Microphone access requires HTTPS. Open https://denizs-mac-mini.taildfdcc0.ts.net:9119/voice instead of the raw http:// Tailscale IP.",
        );
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      await refreshInputDevices();
      addLog("system", "Microphone permission granted; input list refreshed.");
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setError(message);
      addLog("error", message);
    }
  }, [addLog, refreshInputDevices]);

  const stopCall = useCallback(() => {
    const endStartedAt = Date.now();
    const durationMs = callStartedAtRef.current === null ? 0 : endStartedAt - callStartedAtRef.current;
    callSeqRef.current += 1;
    setStatus((current) => (current === "idle" ? current : "ending"));
    persistTranscript("system", "Call ended by user; microphone released.", "call_end", {
      duration_ms: durationMs,
      pending_saves_at_end: pendingTranscriptSavesRef.current.length,
      log_entries: logs.length,
    });
    if (dataRef.current) {
      dataRef.current.onerror = null;
      dataRef.current.onopen = null;
    }
    dataRef.current?.close();
    peerRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    stopMicMonitor();
    if (audioRef.current?.srcObject instanceof MediaStream) {
      audioRef.current.srcObject.getTracks().forEach((track) => track.stop());
      audioRef.current.srcObject = null;
    }
    dataRef.current = null;
    peerRef.current = null;
    streamRef.current = null;
    setMuted(false);
    setActiveTool(null);
    Promise.allSettled([...pendingTranscriptSavesRef.current]).then(() => setSaveStatus("Call saved"));
    setStatus("idle");
    addLog("system", `Call ended; saved end marker (${(durationMs / 1000).toFixed(1)}s).`);
  }, [addLog, logs.length, persistTranscript, stopMicMonitor]);

  const sendRealtimeEvent = useCallback((payload: Record<string, unknown>) => {
    const channel = dataRef.current;
    if (!channel || channel.readyState !== "open") return;
    channel.send(JSON.stringify(payload));
  }, []);

  const handleToolCall = useCallback(
    async (event: Record<string, unknown>) => {
      const name = typeof event.name === "string" ? event.name : "";
      const callId = typeof event.call_id === "string" ? event.call_id : "";
      const rawArgs = typeof event.arguments === "string" ? event.arguments : "{}";
      if (!name || !callId) {
        addLog("error", `Malformed Realtime tool call: ${JSON.stringify(event).slice(0, 500)}`);
        return;
      }

      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(rawArgs) as Record<string, unknown>;
      } catch {
        args = { raw: rawArgs };
      }

      const startedAt = Date.now();
      setActiveTool(`${name} running since ${formatClock(new Date(startedAt).toISOString())}`);
      addLog("tool", `Running ${name}… ${JSON.stringify(args).slice(0, 300)}`);
      persistTranscript("tool", `Running ${name}: ${JSON.stringify(args)}`, "tool_call", {
        realtime_call_id: callId,
        tool_name: name,
        started_at: new Date(startedAt).toISOString(),
      });
      try {
        const result = await api.runVoiceTool({ name, arguments: args } satisfies VoiceToolRequest, speaker);
        const durationMs = Date.now() - startedAt;
        const output = result.ok ? result.result : `Tool failed: ${result.error ?? "unknown error"}`;
        setActiveTool(null);
        addLog(result.ok ? "tool" : "error", `${name} finished in ${(durationMs / 1000).toFixed(1)}s\n${output.slice(0, 700)}`);
        persistTranscript("tool", output, result.ok ? "tool_result" : "tool_error", {
          realtime_call_id: callId,
          tool_name: name,
          duration_ms: durationMs,
        });
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: callId,
            output,
          },
        });
        sendRealtimeEvent({ type: "response.create" });
      } catch (exc) {
        const durationMs = Date.now() - startedAt;
        const message = exc instanceof Error ? exc.message : String(exc);
        setActiveTool(null);
        addLog("error", `${name} failed in ${(durationMs / 1000).toFixed(1)}s: ${message}`);
        persistTranscript("tool", message, "tool_error", {
          realtime_call_id: callId,
          tool_name: name,
          duration_ms: durationMs,
        });
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: callId,
            output: `Tool failed: ${message}`,
          },
        });
        sendRealtimeEvent({ type: "response.create" });
      }
    },
    [addLog, persistTranscript, sendRealtimeEvent, speaker],
  );

  const handleRealtimeEvent = useCallback(
    (message: MessageEvent<string>) => {
      let event: Record<string, unknown>;
      try {
        event = JSON.parse(message.data) as Record<string, unknown>;
      } catch {
        return;
      }
      const type = typeof event.type === "string" ? event.type : "";

      if (type === "response.function_call_arguments.done") {
        void handleToolCall(event);
        return;
      }
      if (type === "response.output_item.done") {
        const item = event.item;
        if (item && typeof item === "object" && (item as Record<string, unknown>).type === "function_call") {
          void handleToolCall(item as Record<string, unknown>);
        }
        return;
      }
      if (type === "conversation.item.input_audio_transcription.completed") {
        const text = eventText(event);
        if (text) {
          addLog("user", text);
          persistTranscript("user", text);
        }
        return;
      }
      if (type === "input_audio_buffer.speech_started") {
        addLog("system", "Realtime API heard speech start.");
        persistTranscript("system", "Realtime API heard speech start.", "speech_started");
        return;
      }
      if (type === "input_audio_buffer.speech_stopped") {
        addLog("system", "Realtime API heard speech stop.");
        persistTranscript("system", "Realtime API heard speech stop.", "speech_stopped");
        return;
      }
      if (type === "input_audio_buffer.committed") {
        addLog("system", "Realtime API committed mic audio.");
        persistTranscript("system", "Realtime API committed mic audio.", "audio_committed");
        return;
      }
      if (type === "response.output_audio_transcript.done" || type === "response.audio_transcript.done" || type === "response.output_text.done") {
        const text = eventText(event);
        if (text) {
          addLog("rolly", text);
          persistTranscript("rolly", text);
        }
        return;
      }
      if (type === "error") {
        const messageText = JSON.stringify(event.error ?? event).slice(0, 700);
        addLog("error", messageText);
        persistTranscript("error", messageText, "realtime_error");
      }
    },
    [addLog, handleToolCall, persistTranscript],
  );

  const startCall = useCallback(async () => {
    const callSeq = callSeqRef.current + 1;
    callSeqRef.current = callSeq;
    callIdRef.current = `voice-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    callStartedAtRef.current = Date.now();
    eventSeqRef.current = 0;
    pendingTranscriptSavesRef.current = [];
    setLastSavePath(null);
    setSaveStatus("Saving call events…");
    setActiveTool(null);
    window.localStorage.setItem("rolly.voice.user", speaker);
    persistTranscript("system", "Call started.", "call_start", {
      user_agent: navigator.userAgent,
      selected_input_id: selectedInputId || "browser-default",
    });
    const isCurrentCall = () => callSeqRef.current === callSeq;
    setError(null);
    setStatus("requesting");
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Microphone access requires HTTPS. Open https://denizs-mac-mini.taildfdcc0.ts.net:9119/voice instead of the raw http:// Tailscale IP.",
        );
      }
      await refreshInputDevices();
      const audio: boolean | MediaTrackConstraints = selectedInputId
        ? { deviceId: { exact: selectedInputId } }
        : true;
      const stream = await navigator.mediaDevices.getUserMedia({ audio });
      if (!isCurrentCall()) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      await refreshInputDevices();
      streamRef.current = stream;
      startMicMonitor(stream);
      const track = stream.getAudioTracks()[0];
      addLog("system", `Browser mic opened: ${track?.label || "unknown device"}. Watch the mic level; it should move when you talk.`);
      persistTranscript("system", `Browser mic opened: ${track?.label || "unknown device"}.`, "mic_opened");
      setStatus("connecting");

      const peer = new RTCPeerConnection();
      peerRef.current = peer;
      peer.onconnectionstatechange = () => {
        if (["closed", "disconnected", "failed"].includes(peer.connectionState)) {
          addLog("system", `Connection ${peer.connectionState}.`);
          persistTranscript("system", `Connection ${peer.connectionState}.`, "connection_state", { state: peer.connectionState });
        }
      };

      peer.ontrack = (event) => {
        if (!audioRef.current) return;
        audioRef.current.srcObject = event.streams[0];
      };
      stream.getAudioTracks().forEach((track) => peer.addTrack(track, stream));

      const dataChannel = peer.createDataChannel("oai-events");
      dataRef.current = dataChannel;
      dataChannel.onopen = () => {
        setStatus("live");
        addLog("system", "Live. Talk normally; Rolly can answer by voice and call tools.");
        persistTranscript("system", "Realtime data channel live.", "call_live");
      };
      dataChannel.onmessage = handleRealtimeEvent;
      dataChannel.onerror = () => addLog("error", "Realtime data channel error.");

      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);

      const answerSdp = await api.createVoiceCall(offer.sdp || "", speaker);
      if (!isCurrentCall()) return;
      await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setError(message);
      setStatus("error");
      addLog("error", message);
      stopCall();
    }
  }, [addLog, handleRealtimeEvent, persistTranscript, refreshInputDevices, selectedInputId, speaker, startMicMonitor, stopCall]);

  const toggleMute = useCallback(() => {
    const next = !muted;
    streamRef.current?.getAudioTracks().forEach((track) => {
      track.enabled = !next;
    });
    setMuted(next);
  }, [muted]);

  useEffect(() => {
    return () => {
      dataRef.current?.close();
      peerRef.current?.close();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      stopMicMonitor();
    };
  }, [stopMicMonitor]);
  useEffect(() => {
    void refreshInputDevices().catch(() => undefined);
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshInputDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", refreshInputDevices);
  }, [refreshInputDevices]);

  const live = status === "live";
  const busy = status === "requesting" || status === "connecting" || status === "ending";

  return (
    <main className="flex h-full min-h-0 flex-col gap-4 overflow-auto p-4 lg:p-6">
      <audio ref={audioRef} autoPlay />
      <section className="border border-current/20 bg-background-base/70 p-5 text-midground shadow-xl">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <Typography className="font-mondwest text-display text-2xl uppercase tracking-[0.12em]">
              Rolly Voice
            </Typography>
            <p className="mt-2 max-w-2xl text-sm text-text-secondary">
              One-on-one call prototype: browser mic/audio, realtime speech, and a backend bridge for bounded research/tool calls.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {!live && !busy ? <Button onClick={enableMicList}>Enable mic list</Button> : null}
            {!live && !busy ? <Button onClick={startCall}>Start call</Button> : null}
            {live ? <Button onClick={toggleMute}>{muted ? "Unmute" : "Mute"}</Button> : null}
            {live || busy ? <Button onClick={stopCall}>End call</Button> : null}
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2 text-xs uppercase tracking-[0.12em] text-text-secondary">
          <span className="border border-current/20 px-2 py-1">Status: {status}</span>
          <span className="border border-current/20 px-2 py-1">Call: {callIdRef.current}</span>
          <span className="border border-current/20 px-2 py-1">Save: {saveStatus}</span>
          {activeTool ? <span className="border border-current/20 px-2 py-1">Tool: {activeTool}</span> : null}
          <span className="border border-current/20 px-2 py-1">Provider: OpenAI Realtime WebRTC</span>
          <span className="border border-current/20 px-2 py-1">Tools: full Rolly</span>
          {lastSavePath ? <span className="border border-current/20 px-2 py-1 normal-case">Transcript: {lastSavePath}</span> : null}
        </div>
        <div className="mt-3 text-xs uppercase tracking-[0.12em] text-text-secondary">
          <label className="mb-3 block">
            SPEAKER
            <select
              className="mt-1 w-full border border-current/20 bg-black/40 p-2 text-midground"
              value={speaker}
              onChange={(event) => setSpeaker(event.target.value)}
              disabled={live || busy}
            >
              <option value="deniz">Deniz</option>
              <option value="arman">Arman</option>
              <option value="buket">Buket</option>
              <option value="metin">Metin</option>
              <option value="guest">Guest</option>
            </select>
          </label>
          <label className="block">
            MIC INPUT
            <select
              className="mt-1 w-full border border-current/20 bg-black/40 p-2 text-midground"
              value={selectedInputId}
              onChange={(event) => setSelectedInputId(event.target.value)}
              disabled={live || busy}
            >
              <option value="">Browser default</option>
              {inputDevices.map((device, index) => (
                <option key={device.deviceId || index} value={device.deviceId}>
                  {device.label || `Microphone ${index + 1}`}
                </option>
              ))}
            </select>
          </label>
          <div>{micInfo}</div>
          <div className="mt-1 h-2 overflow-hidden border border-current/20 bg-black/40">
            <div className="h-full bg-current transition-[width]" style={{ width: `${micLevel}%` }} />
          </div>
          <div className="mt-1">Mic level: {micLevel}%</div>
        </div>
        {error ? <p className="mt-3 text-sm text-red-300">{error}</p> : null}
      </section>

      <section className="grid min-h-[24rem] gap-4 lg:grid-cols-[1fr_22rem]">
        <div className="min-h-0 border border-current/20 bg-black/30 p-4">
          <Typography className="font-mondwest text-display text-lg uppercase tracking-[0.12em]">
            Transcript + events
          </Typography>
          <div className="mt-3 flex max-h-[60vh] flex-col gap-2 overflow-auto pr-1 text-sm">
            {logs.map((entry) => (
              <div key={entry.id} className="border border-current/10 bg-background-base/50 p-3">
                <div className="mb-1 text-[0.65rem] uppercase tracking-[0.14em] text-text-secondary">
                  {entry.kind} · {formatClock(entry.timestamp)} · {formatElapsed(entry.elapsedMs)}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">{entry.text}</div>
              </div>
            ))}
          </div>
        </div>
        <aside className="border border-current/20 bg-background-base/50 p-4 text-sm text-text-secondary">
          <Typography className="font-mondwest text-display text-lg uppercase tracking-[0.12em] text-midground">
            Try saying
          </Typography>
          <ul className="mt-3 list-disc space-y-2 pl-4">
            <li>“Rolly, what were we trying to finish tonight?”</li>
            <li>“Check current MIX review blockers and keep it brief.”</li>
            <li>“Use your tools and tell me what to do next.”</li>
          </ul>
          <p className="mt-4">
            Realtime voice can call full Rolly when it needs project context, files, web, or actions.
          </p>
        </aside>
      </section>
    </main>
  );
}
