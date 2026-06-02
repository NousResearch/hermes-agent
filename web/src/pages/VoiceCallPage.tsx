import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { api, type VoiceTaskResponse, type VoiceToolRequest } from "@/lib/api";
import { getRollyUser, getRollyUserSlug } from "@/lib/rollyIdentity";
import { isRollyWakePhrase } from "@/lib/voiceMeet";

type CallStatus = "idle" | "requesting" | "connecting" | "live" | "ending" | "error";

type LogKind = "system" | "user" | "rolly" | "tool" | "error";

const VOICE_ACTION_BUTTON_CLASS = "leading-tight text-sm tracking-[0.12em] sm:text-base sm:tracking-[0.2em]";

type WakeLockSentinelLike = EventTarget & {
  release: () => Promise<void>;
  released?: boolean;
};

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
  const [speaker, setSpeaker] = useState(() => getRollyUserSlug());
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
  const cueAudioContextRef = useRef<AudioContext | null>(null);
  const backgroundAudioContextRef = useRef<AudioContext | null>(null);
  const backgroundOscillatorRef = useRef<OscillatorNode | null>(null);
  const wakeLockRef = useRef<WakeLockSentinelLike | null>(null);
  const workingCueIntervalRef = useRef<number | null>(null);
  const workingCueTimeoutRef = useRef<number | null>(null);
  const micMonitorRafRef = useRef<number | null>(null);
  const callIdRef = useRef(`voice-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  const callStartedAtRef = useRef<number | null>(null);
  const eventSeqRef = useRef(0);
  const pendingTranscriptSavesRef = useRef<Promise<unknown>[]>([]);
  const transcriptSaveChainRef = useRef<Promise<unknown>>(Promise.resolve());
  const callSeqRef = useRef(0);
  const [saveStatus, setSaveStatus] = useState("Not saving yet");
  const [lastSavePath, setLastSavePath] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [activeWorkCount, setActiveWorkCount] = useState(0);
  const [pendingHandoffs, setPendingHandoffs] = useState(0);
  const [backgroundSupport, setBackgroundSupport] = useState("Background call support: idle");
  const [mode, setMode] = useState<"solo" | "meet">("solo");
  const [callIdDisplay, setCallIdDisplay] = useState(callIdRef.current);
  const [inviteUrl, setInviteUrl] = useState<string | null>(null);
  const [voiceTasks, setVoiceTasks] = useState<VoiceTaskResponse[]>([]);
  const [rollyListenState, setRollyListenState] = useState("Always on");
  const handledToolCallsRef = useRef<Set<string>>(new Set());
  const activeCallModeRef = useRef<"solo" | "meet">("solo");
  const meetInvokedRef = useRef(false);
  const userSpeakingRef = useRef(false);
  const responseActiveRef = useRef(false);

  useEffect(() => {
    if (status !== "idle") return;
    meetInvokedRef.current = false;
    setRollyListenState(mode === "meet" ? "Silent until “Hey Rolly”" : "Always on");
  }, [mode, status]);
  const pendingResponseCreateRef = useRef(false);
  const activeWorkRef = useRef<Map<string, string>>(new Map());
  const pendingHandoffsRef = useRef<Array<{ toolCallId: string; taskId: string; output: string }>>([]);

  const getCueAudioContext = useCallback(() => {
    const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return null;
    if (!cueAudioContextRef.current || cueAudioContextRef.current.state === "closed") {
      cueAudioContextRef.current = new AudioContextCtor();
    }
    void cueAudioContextRef.current.resume().catch(() => undefined);
    return cueAudioContextRef.current;
  }, []);

  const playTone = useCallback(
    (frequency: number, delayMs = 0, durationMs = 90, volume = 0.035) => {
      const context = getCueAudioContext();
      if (!context) return;
      const start = context.currentTime + delayMs / 1000;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(frequency, start);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(volume, start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + durationMs / 1000);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + durationMs / 1000 + 0.03);
    },
    [getCueAudioContext],
  );

  const playVoiceCue = useCallback(
    (kind: "live" | "working" | "done" | "error") => {
      if (kind === "live") {
        playTone(523, 0, 80);
        playTone(784, 95, 110);
      } else if (kind === "working") {
        playTone(659, 0, 65, 0.025);
        playTone(880, 95, 65, 0.022);
      } else if (kind === "done") {
        playTone(880, 0, 70, 0.026);
        playTone(659, 90, 90, 0.022);
      } else {
        playTone(220, 0, 130, 0.035);
      }
    },
    [playTone],
  );

  const releaseWakeLock = useCallback(() => {
    const lock = wakeLockRef.current;
    wakeLockRef.current = null;
    if (lock && !lock.released) void lock.release().catch(() => undefined);
  }, []);

  const requestWakeLock = useCallback(async () => {
    const wakeLock = (navigator as Navigator & { wakeLock?: { request: (type: "screen") => Promise<WakeLockSentinelLike> } }).wakeLock;
    if (!wakeLock || document.visibilityState !== "visible") return false;
    try {
      wakeLockRef.current = await wakeLock.request("screen");
      wakeLockRef.current.addEventListener("release", () => {
        wakeLockRef.current = null;
      });
      return true;
    } catch {
      return false;
    }
  }, []);

  const startBackgroundCallSupport = useCallback(async () => {
    const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (AudioContextCtor && (!backgroundAudioContextRef.current || backgroundAudioContextRef.current.state === "closed")) {
      const context = new AudioContextCtor();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = 20;
      gain.gain.value = 0.00001;
      oscillator.connect(gain).connect(context.destination);
      oscillator.start();
      backgroundAudioContextRef.current = context;
      backgroundOscillatorRef.current = oscillator;
      await context.resume().catch(() => undefined);
    }

    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: "Rolly Voice Call",
        artist: "Rolly",
        album: mode === "meet" ? "Meet mode" : "1:1 mode",
      });
      navigator.mediaSession.playbackState = "playing";
    }

    const wakeLocked = await requestWakeLock();
    setBackgroundSupport(wakeLocked ? "Background call support: app registered + screen wake lock active" : "Background call support: app registered; install to Home Screen for best lock-screen behavior");
  }, [mode, requestWakeLock]);

  const stopBackgroundCallSupport = useCallback(() => {
    releaseWakeLock();
    backgroundOscillatorRef.current?.stop();
    backgroundOscillatorRef.current?.disconnect();
    backgroundOscillatorRef.current = null;
    void backgroundAudioContextRef.current?.close().catch(() => undefined);
    backgroundAudioContextRef.current = null;
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "none";
    setBackgroundSupport("Background call support: idle");
  }, [releaseWakeLock]);

  const startWorkingCue = useCallback(() => {
    if (workingCueIntervalRef.current !== null) return;
    playVoiceCue("working");
    workingCueIntervalRef.current = window.setInterval(() => playVoiceCue("working"), 6500);
  }, [playVoiceCue]);

  const stopWorkingCue = useCallback(
    (finish: "done" | "error" | false = false) => {
      if (workingCueTimeoutRef.current !== null) {
        window.clearTimeout(workingCueTimeoutRef.current);
        workingCueTimeoutRef.current = null;
      }
      if (workingCueIntervalRef.current !== null) {
        window.clearInterval(workingCueIntervalRef.current);
        workingCueIntervalRef.current = null;
      }
      if (finish) playVoiceCue(finish);
    },
    [playVoiceCue],
  );

  const startBoundedWorkingCue = useCallback(() => {
    startWorkingCue();
    if (workingCueTimeoutRef.current !== null) window.clearTimeout(workingCueTimeoutRef.current);
    workingCueTimeoutRef.current = window.setTimeout(() => {
      if (activeWorkRef.current.size === 0) stopWorkingCue(false);
    }, 15000);
  }, [startWorkingCue, stopWorkingCue]);

  const refreshActiveWork = useCallback(() => {
    const count = activeWorkRef.current.size;
    setActiveWorkCount(count);
    if (count > 0) startWorkingCue();
    else stopWorkingCue(false);
  }, [startWorkingCue, stopWorkingCue]);

  const rememberVoiceTask = useCallback((task: VoiceTaskResponse) => {
    setVoiceTasks((prev) => {
      const without = prev.filter((item) => item.task_id !== task.task_id);
      return [task, ...without].slice(0, 12);
    });
  }, []);

  const markWorkStarted = useCallback(
    (id: string, label: string) => {
      activeWorkRef.current.set(id, label);
      setActiveTool(label);
      refreshActiveWork();
    },
    [refreshActiveWork],
  );

  const markWorkFinished = useCallback(
    (id: string, finish: "done" | "error" | false = false) => {
      activeWorkRef.current.delete(id);
      const remaining = Array.from(activeWorkRef.current.values());
      setActiveTool(remaining[remaining.length - 1] ?? null);
      setActiveWorkCount(activeWorkRef.current.size);
      if (activeWorkRef.current.size === 0) stopWorkingCue(finish);
    },
    [stopWorkingCue],
  );

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
      const payload = {
        call_id: callIdRef.current,
        role,
        text,
        event_type: eventType,
        user: speaker,
        timestamp: new Date(now).toISOString(),
        sequence,
        elapsed_ms: startedAt === null ? undefined : now - startedAt,
        metadata,
      };
      const save = transcriptSaveChainRef.current
        .catch(() => undefined)
        .then(() => api.saveVoiceTranscript(payload, speaker))
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
      transcriptSaveChainRef.current = save.catch(() => undefined);
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

  const switchMicrophone = useCallback(
    async (deviceId: string) => {
      if (status !== "live" || !peerRef.current) return;
      setError(null);
      let nextStream: MediaStream | null = null;
      try {
        const audio: MediaTrackConstraints = deviceId ? { deviceId: { exact: deviceId } } : {};
        nextStream = await navigator.mediaDevices.getUserMedia({ audio });
        const nextTrack = nextStream.getAudioTracks()[0];
        if (!nextTrack) throw new Error("Selected microphone produced no audio track.");
        const sender = peerRef.current.getSenders().find((item) => item.track?.kind === "audio");
        if (!sender) throw new Error("Active call has no microphone sender to replace.");
        await sender.replaceTrack(nextTrack);
        streamRef.current?.getTracks().forEach((track) => track.stop());
        const acceptedStream = nextStream;
        if (!acceptedStream) throw new Error("Selected microphone stream was unavailable.");
        streamRef.current = acceptedStream;
        nextStream = null;
        nextTrack.enabled = !muted;
        startMicMonitor(acceptedStream);
        await refreshInputDevices();
        addLog("system", `Switched microphone to ${nextTrack.label || "selected input"}.`);
        persistTranscript("system", `Switched microphone to ${nextTrack.label || "selected input"}.`, "mic_switched", {
          selected_input_id: deviceId || "browser-default",
        });
      } catch (exc) {
        nextStream?.getTracks().forEach((track) => track.stop());
        const message = exc instanceof Error ? exc.message : String(exc);
        setError(`Microphone switch failed: ${message}`);
        addLog("error", `Microphone switch failed: ${message}`);
        persistTranscript("error", `Microphone switch failed: ${message}`, "mic_switch_failed", {
          selected_input_id: deviceId || "browser-default",
        });
      }
    },
    [addLog, muted, persistTranscript, refreshInputDevices, startMicMonitor, status],
  );

  const stopCall = useCallback((reason = "user") => {
    const endStartedAt = Date.now();
    const durationMs = callStartedAtRef.current === null ? 0 : endStartedAt - callStartedAtRef.current;
    callSeqRef.current += 1;
    const statusAtEnd = status;
    setStatus((current) => (current === "idle" ? current : "ending"));
    stopWorkingCue(false);
    stopBackgroundCallSupport();
    const endText = reason === "setup_error"
      ? "Call ended after setup error; microphone released."
      : "Call ended by user; microphone released.";
    persistTranscript("system", endText, "call_end", {
      duration_ms: durationMs,
      pending_saves_at_end: pendingTranscriptSavesRef.current.length,
      log_entries: logs.length,
      status_at_end: statusAtEnd,
      reason,
    });
    if (dataRef.current) {
      dataRef.current.onerror = null;
      dataRef.current.onmessage = null;
      dataRef.current.onopen = null;
    }
    if (peerRef.current) {
      peerRef.current.onconnectionstatechange = null;
      peerRef.current.ontrack = null;
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
    activeWorkRef.current.clear();
    pendingHandoffsRef.current = [];
    userSpeakingRef.current = false;
    responseActiveRef.current = false;
    pendingResponseCreateRef.current = false;
    setActiveWorkCount(0);
    setPendingHandoffs(0);
    setMuted(false);
    setActiveTool(null);
    Promise.allSettled([...pendingTranscriptSavesRef.current]).then(() => setSaveStatus("Call saved"));
    setStatus("idle");
    addLog("system", `Call ended; saved end marker (${(durationMs / 1000).toFixed(1)}s).`);
  }, [addLog, logs.length, persistTranscript, status, stopBackgroundCallSupport, stopMicMonitor, stopWorkingCue]);

  const sendRealtimeEvent = useCallback((payload: Record<string, unknown>) => {
    const channel = dataRef.current;
    if (!channel || channel.readyState !== "open") return;
    channel.send(JSON.stringify(payload));
  }, []);

  const requestResponseCreate = useCallback(
    (reason: string) => {
      if (responseActiveRef.current) {
        pendingResponseCreateRef.current = true;
        addLog("system", `Queued voice response until current response finishes (${reason}).`);
        return;
      }
      responseActiveRef.current = true;
      sendRealtimeEvent({ type: "response.create" });
    },
    [addLog, sendRealtimeEvent],
  );

  const finishResponse = useCallback(
    (status: "done" | "cancelled" | "error") => {
      responseActiveRef.current = false;
      if (!pendingResponseCreateRef.current) return;
      pendingResponseCreateRef.current = false;
      window.setTimeout(() => {
        requestResponseCreate(`pending after ${status}`);
      }, 100);
    },
    [requestResponseCreate],
  );

  const flushPendingHandoffs = useCallback(() => {
    if (userSpeakingRef.current || pendingHandoffsRef.current.length === 0) return;
    const handoffs = pendingHandoffsRef.current.splice(0);
    setPendingHandoffs(0);
    for (const handoff of handoffs) {
      if (handoff.toolCallId.startsWith("handoff:")) {
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: `Background Rolly task result is ready. Summarize it to the user in no more than two short spoken sentences. Full result: ${handoff.output}` }],
          },
        });
      } else {
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: handoff.toolCallId,
            output: handoff.output,
          },
        });
      }
      persistTranscript("tool", handoff.output, "delegation_handoff", { task_id: handoff.taskId });
      requestResponseCreate("tool output");
    }
  }, [persistTranscript, requestResponseCreate, sendRealtimeEvent]);

  const queueOrSendToolOutput = useCallback(
    (toolCallId: string, output: string, taskId = "foreground") => {
      if (userSpeakingRef.current) {
        pendingHandoffsRef.current.push({ toolCallId, taskId, output });
        setPendingHandoffs(pendingHandoffsRef.current.length);
        persistTranscript("system", "Queued Rolly result until user stops speaking.", "handoff_queued", { task_id: taskId });
        return;
      }
      if (toolCallId.startsWith("handoff:")) {
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: `Background Rolly task result is ready. Summarize it to the user in no more than two short spoken sentences. Full result: ${output}` }],
          },
        });
      } else {
        sendRealtimeEvent({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: toolCallId,
            output,
          },
        });
      }
      requestResponseCreate("tool output");
    },
    [persistTranscript, requestResponseCreate, sendRealtimeEvent],
  );

  const pollVoiceTask = useCallback(
    async (taskId: string, toolCallId: string) => {
      let lastProgress = "";
      try {
        for (;;) {
          await new Promise((resolve) => window.setTimeout(resolve, 2000));
          const task: VoiceTaskResponse = await api.getVoiceTask(taskId, speaker);
          rememberVoiceTask(task);
          const latest = task.progress?.[task.progress.length - 1]?.message ?? task.status;
          if (latest && latest !== lastProgress) {
            lastProgress = latest;
            addLog("tool", `${taskId}: ${latest}`);
            persistTranscript("tool", latest, "delegation_progress", { task_id: taskId, status: task.status });
          }
          if (task.status === "complete") {
            const output = task.result || "Background Rolly task completed.";
            addLog("tool", `${taskId} complete\n${output.slice(0, 700)}`);
            persistTranscript("tool", output, "delegation_result", { task_id: taskId, session_id: task.session_id });
            queueOrSendToolOutput(`handoff:${taskId}`, output, taskId);
            return;
          }
          if (task.status === "failed" || task.status === "cancelled") {
            const output = `Background Rolly task ${task.status}: ${task.error || "no error detail"}`;
            addLog("error", output);
            persistTranscript("tool", output, "delegation_error", { task_id: taskId, session_id: task.session_id });
            queueOrSendToolOutput(`handoff:${taskId}`, output, taskId);
            return;
          }
        }
      } catch (exc) {
        const message = exc instanceof Error ? exc.message : String(exc);
        addLog("error", `${taskId} polling failed: ${message}`);
        persistTranscript("tool", message, "delegation_error", { task_id: taskId });
        queueOrSendToolOutput(toolCallId, `Background task status check failed: ${message}`, taskId);
      }
    },
    [addLog, persistTranscript, queueOrSendToolOutput, rememberVoiceTask, speaker],
  );

  const handleToolCall = useCallback(
    async (event: Record<string, unknown>) => {
      const name = typeof event.name === "string" ? event.name : "";
      const callId = typeof event.call_id === "string" ? event.call_id : "";
      const rawArgs = typeof event.arguments === "string" ? event.arguments : "";
      if (!name || !callId) {
        addLog("error", `Malformed Realtime tool call: ${JSON.stringify(event).slice(0, 500)}`);
        return;
      }
      if (!rawArgs.trim()) {
        addLog("system", `Waiting for Realtime function arguments for ${name}:${callId}.`);
        return;
      }
      const toolKey = `${name}:${callId}`;
      if (handledToolCallsRef.current.has(toolKey)) {
        addLog("system", `Skipped duplicate Realtime tool call ${toolKey}.`);
        return;
      }

      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(rawArgs) as Record<string, unknown>;
      } catch {
        addLog("error", `Invalid Realtime tool arguments for ${toolKey}: ${rawArgs.slice(0, 300)}`);
        return;
      }
      handledToolCallsRef.current.add(toolKey);

      const startedAt = Date.now();
      markWorkStarted(`tool:${toolKey}`, `${name} running since ${formatClock(new Date(startedAt).toISOString())}`);
      addLog("tool", `Running ${name}… ${JSON.stringify(args).slice(0, 300)}`);
      persistTranscript("tool", `Running ${name}: ${JSON.stringify(args)}`, "tool_call", {
        realtime_call_id: callId,
        tool_name: name,
        started_at: new Date(startedAt).toISOString(),
      });
      try {
        const idempotencyKey = `${callIdRef.current}:${callId}`;
        const result = await api.runVoiceTool({ name, arguments: args, call_id: callIdRef.current, realtime_call_id: callId, idempotency_key: idempotencyKey } satisfies VoiceToolRequest, speaker);
        const durationMs = Date.now() - startedAt;
        const output = result.ok ? result.result : `Tool failed: ${result.error ?? "unknown error"}`;
        markWorkFinished(`tool:${toolKey}`, false);
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
        requestResponseCreate("tool output");
        const taskId = typeof result.data?.task_id === "string" ? result.data.task_id : "";
        if (name === "rolly_background" && taskId) {
          rememberVoiceTask(result.data as unknown as VoiceTaskResponse);
          void pollVoiceTask(taskId, callId);
        }
      } catch (exc) {
        const durationMs = Date.now() - startedAt;
        const message = exc instanceof Error ? exc.message : String(exc);
        markWorkFinished(`tool:${toolKey}`, "error");
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
        requestResponseCreate("tool output");
      }
    },
    [addLog, markWorkFinished, markWorkStarted, persistTranscript, pollVoiceTask, rememberVoiceTask, requestResponseCreate, sendRealtimeEvent, speaker],
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
          const currentMode = activeCallModeRef.current;
          if (currentMode === "meet") {
            const invoked = isRollyWakePhrase(text);
            if (invoked) {
              meetInvokedRef.current = true;
              setRollyListenState("Invoked by “Hey Rolly”");
              startBoundedWorkingCue();
              requestResponseCreate("meet wake phrase");
            } else if (!meetInvokedRef.current) {
              setRollyListenState("Silent; no “Hey Rolly” heard");
            }
          }
          const meetMetadata =
            currentMode === "meet"
              ? { mode: currentMode, invoked_rolly: meetInvokedRef.current, dashboard_user: speaker, speaker_attribution: "unknown_room_speaker" }
              : { mode: currentMode, invoked_rolly: meetInvokedRef.current };
          addLog("user", currentMode === "meet" ? `[room mic / dashboard: ${speaker || "unknown"}] ${text}` : text);
          persistTranscript("user", text, "transcript", meetMetadata);
        }
        return;
      }
      if (type === "input_audio_buffer.speech_started") {
        userSpeakingRef.current = true;
        addLog("system", "Realtime API heard speech start.");
        persistTranscript("system", "Realtime API heard speech start.", "speech_started");
        return;
      }
      if (type === "input_audio_buffer.speech_stopped") {
        userSpeakingRef.current = false;
        addLog("system", "Realtime API heard speech stop.");
        persistTranscript("system", "Realtime API heard speech stop.", "speech_stopped");
        window.setTimeout(flushPendingHandoffs, 250);
        return;
      }
      if (type === "input_audio_buffer.committed") {
        addLog("system", "Realtime API committed mic audio.");
        persistTranscript("system", "Realtime API committed mic audio.", "audio_committed");
        return;
      }
      if (type === "response.created") {
        responseActiveRef.current = true;
        return;
      }
      if (type === "response.done" || type === "response.cancelled") {
        finishResponse(type === "response.cancelled" ? "cancelled" : "done");
        stopWorkingCue(type === "response.cancelled" ? false : "done");
        if (activeCallModeRef.current === "meet") {
          meetInvokedRef.current = false;
          setRollyListenState("Silent until “Hey Rolly”");
        }
        return;
      }
      if (type === "response.output_audio_transcript.done" || type === "response.audio_transcript.done" || type === "response.output_text.done") {
        stopWorkingCue(false);
        const text = eventText(event);
        if (text) {
          addLog("rolly", text);
          persistTranscript("rolly", text);
        }
        return;
      }
      if (type === "error") {
        const errorObj = event.error && typeof event.error === "object" ? (event.error as Record<string, unknown>) : {};
        if (errorObj.code === "conversation_already_has_active_response") {
          pendingResponseCreateRef.current = true;
          addLog("system", "Realtime response was already active; queued another response after it finishes.");
          persistTranscript("system", "Queued voice response after active-response conflict.", "realtime_response_queued");
          return;
        }
        finishResponse("error");
        stopWorkingCue("error");
        const messageText = JSON.stringify(event.error ?? event).slice(0, 700);
        addLog("error", messageText);
        persistTranscript("error", messageText, "realtime_error");
      }
    },
    [addLog, finishResponse, flushPendingHandoffs, handleToolCall, persistTranscript, requestResponseCreate, sendRealtimeEvent, speaker, startBoundedWorkingCue, startWorkingCue, stopWorkingCue],
  );

  const startCall = useCallback(async (overrideMode?: "solo" | "meet") => {
    const callMode = overrideMode ?? mode;
    activeCallModeRef.current = callMode;
    if (!speaker) {
      setError("Pick a dashboard user first, then start the call.");
      addLog("error", "Pick a dashboard user first, then start the call.");
      return;
    }
    const callSeq = callSeqRef.current + 1;
    callSeqRef.current = callSeq;
    if (!new URLSearchParams(window.location.search).get("call_id")) {
      callIdRef.current = `voice-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    setCallIdDisplay(callIdRef.current);
    callStartedAtRef.current = Date.now();
    eventSeqRef.current = 0;
    pendingTranscriptSavesRef.current = [];
    setLastSavePath(null);
    setSaveStatus("Saving call events…");
    activeWorkRef.current.clear();
    pendingHandoffsRef.current = [];
    userSpeakingRef.current = false;
    responseActiveRef.current = false;
    pendingResponseCreateRef.current = false;
    setActiveWorkCount(0);
    setPendingHandoffs(0);
    setActiveTool(null);
    setVoiceTasks([]);
    handledToolCallsRef.current = new Set();
    meetInvokedRef.current = false;
    setRollyListenState(callMode === "meet" ? "Silent until “Hey Rolly”" : "Always on");
    persistTranscript("system", "Call started.", "call_start", {
      user_agent: navigator.userAgent,
      selected_input_id: selectedInputId || "browser-default",
      mode: callMode,
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
      await startBackgroundCallSupport();
      if (!isCurrentCall()) {
        stopBackgroundCallSupport();
        stopMicMonitor();
        if (streamRef.current === stream) streamRef.current = null;
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
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
        playVoiceCue("live");
        const liveMessage = callMode === "meet"
          ? "Meet mode live. Rolly is silent until someone says “Hey Rolly.”"
          : "Live. Talk normally; Rolly can answer by voice and call tools.";
        addLog("system", liveMessage);
        persistTranscript("system", "Realtime data channel live.", "call_live", { mode: callMode });
      };
      dataChannel.onmessage = handleRealtimeEvent;
      dataChannel.onerror = () => addLog("error", "Realtime data channel error.");

      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      persistTranscript("system", "WebRTC offer created; requesting Realtime answer.", "webrtc_offer_created", { mode: callMode });

      const answerSdp = await api.createVoiceCall(offer.sdp || "", speaker, callMode);
      if (!isCurrentCall()) return;
      persistTranscript("system", "Realtime SDP answer received.", "realtime_answer_received", { mode: callMode });
      await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });
      persistTranscript("system", "Realtime remote description applied.", "webrtc_remote_description_set", { mode: callMode });
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setError(message);
      setStatus("error");
      addLog("error", message);
      stopCall("setup_error");
    }
  }, [addLog, handleRealtimeEvent, mode, persistTranscript, refreshInputDevices, selectedInputId, speaker, startBackgroundCallSupport, startMicMonitor, stopBackgroundCallSupport, stopCall, stopMicMonitor, playVoiceCue]);

  const startMeetingInvite = useCallback(async () => {
    setError(null);
    try {
      const resp = await api.createVoiceMeetInvite({ call_id: callIdRef.current, user: speaker });
      setMode("meet");
      setInviteUrl(resp.invite_url);
      addLog("system", `Meet invite ready: ${resp.invite_url}`);
      persistTranscript("system", `Meet invite created: ${resp.invite_url}`, "meet_invite_created", { mode: "meet" });
      await startCall("meet");
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setError(`Meet invite unavailable: ${message}`);
      addLog("error", `Meet invite unavailable: ${message}`);
    }
  }, [addLog, persistTranscript, speaker, startCall]);

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
      stopWorkingCue(false);
      stopBackgroundCallSupport();
      void cueAudioContextRef.current?.close().catch(() => undefined);
      cueAudioContextRef.current = null;
      stopMicMonitor();
    };
  }, [stopBackgroundCallSupport, stopMicMonitor, stopWorkingCue]);
  useEffect(() => {
    void refreshInputDevices().catch(() => undefined);
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshInputDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", refreshInputDevices);
  }, [refreshInputDevices]);
  useEffect(() => {
    const updateUser = () => setSpeaker(getRollyUserSlug());
    window.addEventListener("rolly-user-change", updateUser);
    updateUser();
    return () => window.removeEventListener("rolly-user-change", updateUser);
  }, []);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requestedMode = params.get("mode");
    const requestedCallId = params.get("call_id");
    if (requestedMode === "meet") setMode("meet");
    if (requestedCallId) {
      callIdRef.current = requestedCallId;
      setCallIdDisplay(requestedCallId);
      setInviteUrl(window.location.href);
    }
  }, []);
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === "visible" && status === "live") {
        void requestWakeLock();
        void backgroundAudioContextRef.current?.resume().catch(() => undefined);
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [requestWakeLock, status]);

  const live = status === "live";
  const busy = status === "requesting" || status === "connecting" || status === "ending";
  const speakerLabel = getRollyUser(speaker)?.label ?? "No dashboard user selected";
  const transcriptLogs = logs.filter((entry) => entry.kind === "user" || entry.kind === "rolly");
  const eventLogs = logs.filter((entry) => entry.kind !== "user" && entry.kind !== "rolly");

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
              Realtime Rolly Voice with fast local context, durable transcripts, and optional Meet mode where Rolly only speaks after “Hey Rolly.”
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {!live && !busy ? <Button className={VOICE_ACTION_BUTTON_CLASS} onClick={startMeetingInvite} disabled={!speaker}>Start meeting / invite</Button> : null}
            {!live && !busy ? <Button className={VOICE_ACTION_BUTTON_CLASS} onClick={() => void startCall()} disabled={!speaker}>Start call</Button> : null}
            {live ? <Button className={VOICE_ACTION_BUTTON_CLASS} onClick={toggleMute}>{muted ? "Unmute" : "Mute"}</Button> : null}
            {live || busy ? <Button className={VOICE_ACTION_BUTTON_CLASS} onClick={() => stopCall()}>End call</Button> : null}
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2 text-xs uppercase tracking-[0.12em] text-text-secondary">
          <span className="border border-current/20 px-2 py-1">Status: {status}</span>
          <span className="border border-current/20 px-2 py-1">Call: {callIdDisplay}</span>
          <span className="border border-current/20 px-2 py-1">Save: {saveStatus}</span>
          <span className="border border-current/20 px-2 py-1">Mode: {mode === "meet" ? "Meet" : "1:1"}</span>
          <span className="border border-current/20 px-2 py-1 normal-case">Rolly: {rollyListenState}</span>
          <span className="border border-current/20 px-2 py-1 normal-case">{backgroundSupport}</span>
          {activeTool ? <span className="border border-current/20 px-2 py-1">Tool: {activeTool}</span> : null}
          {activeWorkCount > 0 ? <span className="border border-current/20 px-2 py-1">Working: {activeWorkCount}</span> : null}
          {pendingHandoffs > 0 ? <span className="border border-current/20 px-2 py-1">Queued handoffs: {pendingHandoffs}</span> : null}
          <span className="border border-current/20 px-2 py-1">Provider: OpenAI Realtime WebRTC</span>
          <span className="border border-current/20 px-2 py-1">Tools: fast context + full Rolly</span>
          {lastSavePath ? <span className="border border-current/20 px-2 py-1 normal-case">Transcript: {lastSavePath}</span> : null}
        </div>
        {inviteUrl ? (
          <div className="mt-3 border border-current/20 bg-black/30 p-2 text-xs text-text-secondary">
            Invite: <span className="break-all normal-case text-midground">{inviteUrl}</span>
          </div>
        ) : null}
        <div className="mt-3 text-xs uppercase tracking-[0.12em] text-text-secondary">
          <div className="mb-3">USER: {speakerLabel}</div>
          <div className="mb-3 flex gap-2">
            <Button className={VOICE_ACTION_BUTTON_CLASS} disabled={live || busy} onClick={() => setMode("solo")}>
              1:1 Rolly
            </Button>
            <Button className={VOICE_ACTION_BUTTON_CLASS} disabled={live || busy} onClick={() => setMode("meet")}>
              Meet: Hey Rolly
            </Button>
          </div>
          <label className="block">
            MIC INPUT
            <select
              className="mt-1 w-full border border-current/20 bg-black/40 p-2 text-midground"
              value={selectedInputId}
              onChange={(event) => {
                const next = event.target.value;
                setSelectedInputId(next);
                if (live) void switchMicrophone(next);
              }}
              disabled={busy}
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

      <section className="grid min-h-[24rem] gap-4 xl:grid-cols-[1fr_1fr_22rem]">
        <div className="min-h-0 border border-current/20 bg-black/30 p-4">
          <Typography className="font-mondwest text-display text-lg uppercase tracking-[0.12em]">
            Live transcript
          </Typography>
          <div className="mt-3 flex max-h-[60vh] flex-col gap-2 overflow-auto pr-1 text-sm">
            {(transcriptLogs.length ? transcriptLogs : [{ id: "empty-transcript", kind: "system" as LogKind, text: "No spoken transcript yet.", timestamp: new Date().toISOString(), elapsedMs: null }]).map((entry) => (
              <div key={entry.id} className="border border-current/10 bg-background-base/50 p-3">
                <div className="mb-1 text-[0.65rem] uppercase tracking-[0.14em] text-text-secondary">
                  {entry.kind} · {formatClock(entry.timestamp)} · {formatElapsed(entry.elapsedMs)}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">{entry.text}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="min-h-0 border border-current/20 bg-black/30 p-4">
          <Typography className="font-mondwest text-display text-lg uppercase tracking-[0.12em]">
            Events + work
          </Typography>
          <div className="mt-3 flex max-h-[60vh] flex-col gap-2 overflow-auto pr-1 text-sm">
            {eventLogs.map((entry) => (
              <div key={entry.id} className="border border-current/10 bg-background-base/50 p-3">
                <div className="mb-1 text-[0.65rem] uppercase tracking-[0.14em] text-text-secondary">
                  {entry.kind} · {formatClock(entry.timestamp)} · {formatElapsed(entry.elapsedMs)}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">{entry.text}</div>
              </div>
            ))}
            {voiceTasks.map((task) => (
              <div key={task.task_id} className="border border-current/10 bg-background-base/50 p-3">
                <div className="mb-1 text-[0.65rem] uppercase tracking-[0.14em] text-text-secondary">
                  {task.task_id} · {task.status} · {formatClock(task.updated_at)}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">{task.progress?.[task.progress.length - 1]?.message || task.request}</div>
              </div>
            ))}
          </div>
        </div>
        <aside className="border border-current/20 bg-background-base/50 p-4 text-sm text-text-secondary">
          <Typography className="font-mondwest text-display text-lg uppercase tracking-[0.12em] text-midground">
            Pinned
          </Typography>
          <div className="mt-3 space-y-3">
            <div>Transcript: {lastSavePath || "not saved yet"}</div>
            <div>Queued tasks: {voiceTasks.length || "none"}</div>
            <div>Speaking pace: slightly faster style instruction; explicit provider rate unsupported.</div>
          </div>
          <Typography className="mt-5 font-mondwest text-display text-lg uppercase tracking-[0.12em] text-midground">
            Try saying
          </Typography>
          <ul className="mt-3 list-disc space-y-2 pl-4">
            <li>“Rolly, what were we trying to finish tonight?”</li>
            <li>“Check current MIX review blockers and keep it brief.”</li>
            <li>“Use your tools and tell me what to do next.”</li>
          </ul>
        </aside>
      </section>
    </main>
  );
}
