import { Button } from "@nous-research/ui/ui/components/button";
import {
  Globe,
  Maximize2,
  Mic,
  MicOff,
  Minimize2,
  Send,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EventsFeedClient } from "@/lib/eventsFeedClient";
import {
  EVENTS_MAX_RECONNECT_ATTEMPTS,
  eventsReconnectDelayMs,
  isEventsAuthRejection,
  shouldRetryEventsClose,
} from "@/lib/events-reconnect";
import {
  cleanSpokenText,
  extractNextSpokenSentence,
} from "@/lib/speechUtils";
import {
  detectDominantScript,
  getVoicePauseTimeoutMs,
  normalizeVoicePrompt,
  type VoiceLanguageMode,
  type VoiceRecognitionEvent,
} from "@/lib/chat-voice";
import { speakWithNabra, stopNabraAudio } from "@/utils/jarvisSpeechUtils";
import { JarvisUltronVoiceOrb } from "./JarvisUltronVoiceOrb";

const VOICE_LANG_STORAGE_KEY = "hermes_chat_voice_lang";
const ORB_MODE_STORAGE_KEY = "hermes_chat_orb_mode";

interface ChatVoiceControlsProps {
  channel: string;
  connected: boolean;
  foreground: string;
  onSubmit: (text: string) => boolean;
}

interface RecognitionErrorLike {
  error?: string;
}

interface RecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onresult: ((event: VoiceRecognitionEvent) => void) | null;
  onerror: ((event: RecognitionErrorLike) => void) | null;
  onend: (() => void) | null;
  start(): void;
  abort(): void;
}

type RecognitionConstructor = new () => RecognitionLike;

function speechRecognitionConstructor(): RecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const candidate = (
    window as typeof window & {
      SpeechRecognition?: RecognitionConstructor;
      webkitSpeechRecognition?: RecognitionConstructor;
    }
  ).SpeechRecognition ?? (
    window as typeof window & { webkitSpeechRecognition?: RecognitionConstructor }
  ).webkitSpeechRecognition;
  return candidate ?? null;
}

export function ChatVoiceControls({
  channel,
  connected,
  foreground,
  onSubmit,
}: ChatVoiceControlsProps) {
  const feed = useMemo(() => new EventsFeedClient(), []);
  const [liveEnabled, setLiveEnabled] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const [listening, setListening] = useState(false);
  const [status, setStatus] = useState("Voice ready");
  const [draft, setDraft] = useState("");
  const [pauseCountdown, setPauseCountdown] = useState<number | null>(null);

  const [stageMode, setStageMode] = useState<"compact" | "expanded">(() => {
    if (typeof window !== "undefined") {
      try {
        const saved = localStorage.getItem(ORB_MODE_STORAGE_KEY);
        if (saved === "compact" || saved === "expanded") {
          return saved;
        }
      } catch {
        // ignore
      }
    }
    return "compact";
  });

  const [languageMode, setLanguageMode] = useState<VoiceLanguageMode>(() => {
    if (typeof window !== "undefined") {
      try {
        const saved = localStorage.getItem(VOICE_LANG_STORAGE_KEY);
        if (saved === "en" || saved === "ar" || saved === "auto") {
          return saved;
        }
      } catch {
        // ignore
      }
    }
    return "en";
  });

  const languageModeRef = useRef<VoiceLanguageMode>(languageMode);
  const activeAutoLangRef = useRef<"en-US" | "ar-EG">("en-US");

  const [micAnalyser, setMicAnalyser] = useState<AnalyserNode | null>(null);
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const liveEnabledRef = useRef(false);
  const speechEnabledRef = useRef(false);
  const recognitionRef = useRef<RecognitionLike | null>(null);
  const recognitionRunningRef = useRef(false);
  const assistantBusyRef = useRef(false);
  const waitingForReplyRef = useRef(false);

  const speechAccumulatorRef = useRef("");
  const interimDraftRef = useRef("");
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const rawReplyRef = useRef("");
  const spokenIndexRef = useRef(0);
  const speechQueueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const speechGenerationRef = useRef(0);
  const mountedRef = useRef(true);
  const startListeningRef = useRef<() => void>(() => undefined);

  useEffect(() => {
    languageModeRef.current = languageMode;
  }, [languageMode]);

  const startAudioAnalyser = useCallback(async () => {
    if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (!liveEnabledRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      mediaStreamRef.current = stream;
      const AudioCtxCtor = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtxCtor) {
        const ctx = new AudioCtxCtor();
        audioContextRef.current = ctx;
        if (ctx.state === "suspended") {
          void ctx.resume();
        }
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.75;
        source.connect(analyser);
        setMicAnalyser(analyser);
      }
    } catch (e) {
      console.warn("[ChatVoiceControls] Mic audio analyser setup note:", e);
    }
  }, []);

  const stopAudioAnalyser = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    setMicAnalyser(null);
  }, []);

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setPauseCountdown(null);
  }, []);

  const stopRecognition = useCallback(() => {
    clearSilenceTimer();
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    recognitionRunningRef.current = false;
    setListening(false);
    if (!recognition) return;
    recognition.onstart = null;
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    try {
      recognition.abort();
    } catch {
      // The browser may already have ended this recognition generation.
    }
  }, [clearSilenceTimer]);

  const maybeResumeListening = useCallback(() => {
    if (
      liveEnabledRef.current &&
      connected &&
      !assistantBusyRef.current &&
      !waitingForReplyRef.current &&
      !speakingRef.current
    ) {
      window.setTimeout(() => startListeningRef.current(), 180);
    }
  }, [connected]);

  const submitCurrentDraft = useCallback((): boolean => {
    clearSilenceTimer();
    const fullText = (
      speechAccumulatorRef.current +
      (interimDraftRef.current ? ` ${interimDraftRef.current}` : "")
    ).trim();
    const prompt = normalizeVoicePrompt(fullText);
    if (!prompt) return false;

    speechAccumulatorRef.current = "";
    interimDraftRef.current = "";
    setDraft("");

    waitingForReplyRef.current = true;
    assistantBusyRef.current = true;
    stopRecognition();

    if (onSubmit(prompt)) {
      setStatus("Sent to Hermes");
      return true;
    } else {
      waitingForReplyRef.current = false;
      assistantBusyRef.current = false;
      setStatus("Chat is reconnecting");
      maybeResumeListening();
      return false;
    }
  }, [clearSilenceTimer, maybeResumeListening, onSubmit, stopRecognition]);

  const pumpSpeechQueue = useCallback(async () => {
    if (speakingRef.current || !speechEnabledRef.current) {
      return;
    }
    speakingRef.current = true;
    setIsAssistantSpeaking(true);
    const generation = speechGenerationRef.current;
    stopRecognition();
    setStatus("Hermes speaking");
    while (
      mountedRef.current &&
      speechEnabledRef.current &&
      generation === speechGenerationRef.current
    ) {
      const next = speechQueueRef.current.shift();
      if (!next) break;
      await speakWithNabra(next, "jarvis", false);
    }
    if (generation !== speechGenerationRef.current) return;
    speakingRef.current = false;
    setIsAssistantSpeaking(false);
    maybeResumeListening();
  }, [maybeResumeListening, stopRecognition]);

  const drainReply = useCallback(
    (flush: boolean) => {
      const clean = cleanSpokenText(rawReplyRef.current);
      while (spokenIndexRef.current < clean.length) {
        const extracted = extractNextSpokenSentence(clean, spokenIndexRef.current);
        if (!extracted) break;
        speechQueueRef.current.push(extracted.sentence);
        spokenIndexRef.current = extracted.nextIndex;
      }
      if (flush) {
        const remainder = clean.slice(spokenIndexRef.current).trim();
        if (remainder) speechQueueRef.current.push(remainder);
        spokenIndexRef.current = clean.length;
      }
      void pumpSpeechQueue();
    },
    [pumpSpeechQueue],
  );

  const startListening = useCallback(() => {
    if (
      !liveEnabledRef.current ||
      !connected ||
      assistantBusyRef.current ||
      waitingForReplyRef.current ||
      speakingRef.current ||
      recognitionRunningRef.current
    ) {
      return;
    }
    const Recognition = speechRecognitionConstructor();
    if (!Recognition) {
      liveEnabledRef.current = false;
      setLiveEnabled(false);
      setStatus("Speech recognition is unavailable in this browser");
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    const mode = languageModeRef.current;
    let targetLang = "en-US";
    if (mode === "ar") {
      targetLang = "ar-EG";
    } else if (mode === "auto") {
      targetLang = activeAutoLangRef.current || "en-US";
    } else {
      targetLang = "en-US";
    }
    recognition.lang = targetLang;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      recognitionRunningRef.current = true;
      setListening(true);
      const label =
        mode === "ar"
          ? "Arabic"
          : mode === "en"
            ? "English"
            : `Auto (${targetLang === "ar-EG" ? "AR" : "EN"})`;
      setStatus(`Listening (${label}) — speak your command`);
    };

    recognition.onresult = (event) => {
      let finalChunks = "";
      let interimChunks = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const text = String(res?.[0]?.transcript ?? "").trim();
        if (!text) continue;
        if (res.isFinal) {
          finalChunks = finalChunks ? `${finalChunks} ${text}` : text;
        } else {
          interimChunks = interimChunks ? `${interimChunks} ${text}` : text;
        }
      }

      if (finalChunks) {
        speechAccumulatorRef.current = speechAccumulatorRef.current
          ? `${speechAccumulatorRef.current} ${finalChunks}`
          : finalChunks;
      }
      interimDraftRef.current = interimChunks;

      const currentDraft = (
        speechAccumulatorRef.current +
        (interimChunks ? ` ${interimChunks}` : "")
      ).trim();

      if (!currentDraft) return;

      setDraft(currentDraft);

      // In auto mode, adapt language on detected script
      if (languageModeRef.current === "auto") {
        const script = detectDominantScript(currentDraft);
        if (script === "ar" && activeAutoLangRef.current !== "ar-EG") {
          activeAutoLangRef.current = "ar-EG";
        } else if (script === "en" && activeAutoLangRef.current !== "en-US") {
          activeAutoLangRef.current = "en-US";
        }
      }

      clearSilenceTimer();

      const timeoutMs = getVoicePauseTimeoutMs(currentDraft, 1400);
      const deadline = Date.now() + timeoutMs;
      setPauseCountdown(Math.ceil(timeoutMs / 1000));

      countdownIntervalRef.current = setInterval(() => {
        const remainingMs = Math.max(0, deadline - Date.now());
        const remSecs = Math.ceil(remainingMs / 1000);
        setPauseCountdown(remSecs);
        if (remainingMs <= 0) {
          if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
          }
        }
      }, 200);

      silenceTimerRef.current = setTimeout(() => {
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }
        setPauseCountdown(null);
        void submitCurrentDraft();
      }, timeoutMs);
    };

    recognition.onerror = (event) => {
      const denied =
        event.error === "not-allowed" || event.error === "service-not-allowed";
      if (denied) {
        liveEnabledRef.current = false;
        setLiveEnabled(false);
        setStatus("Microphone permission was denied");
      } else if (event.error !== "aborted" && event.error !== "no-speech") {
        setStatus(`Speech recognition error: ${event.error ?? "unknown"}`);
      }
    };

    recognition.onend = () => {
      recognitionRunningRef.current = false;
      recognitionRef.current = null;
      setListening(false);
      // If idle without an active draft, seamlessly resume listening
      if (!speechAccumulatorRef.current && !interimDraftRef.current) {
        maybeResumeListening();
      }
    };

    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
      recognitionRunningRef.current = false;
      setStatus("Could not start the microphone");
    }
  }, [
    clearSilenceTimer,
    connected,
    maybeResumeListening,
    submitCurrentDraft,
  ]);

  useEffect(() => {
    startListeningRef.current = startListening;
  }, [startListening]);

  const toggleLive = useCallback(() => {
    const next = !liveEnabledRef.current;
    liveEnabledRef.current = next;
    setLiveEnabled(next);
    clearSilenceTimer();
    speechAccumulatorRef.current = "";
    interimDraftRef.current = "";
    setDraft("");
    if (next) {
      setStatus(connected ? "Starting microphone" : "Chat is reconnecting");
      window.setTimeout(() => startListeningRef.current(), 0);
      void startAudioAnalyser();
    } else {
      waitingForReplyRef.current = false;
      stopRecognition();
      stopAudioAnalyser();
      setStatus("Microphone off");
    }
  }, [clearSilenceTimer, connected, startAudioAnalyser, stopAudioAnalyser, stopRecognition]);

  const toggleLanguage = useCallback(() => {
    const prev = languageModeRef.current;
    const next: VoiceLanguageMode =
      prev === "en" ? "ar" : prev === "ar" ? "auto" : "en";
    languageModeRef.current = next;
    setLanguageMode(next);
    if (typeof window !== "undefined") {
      try {
        localStorage.setItem(VOICE_LANG_STORAGE_KEY, next);
      } catch {
        // ignore
      }
    }
    if (next === "ar") {
      activeAutoLangRef.current = "ar-EG";
    } else if (next === "en") {
      activeAutoLangRef.current = "en-US";
    }
    if (liveEnabledRef.current) {
      stopRecognition();
      window.setTimeout(() => startListeningRef.current(), 10);
    }
  }, [stopRecognition]);

  const toggleStageMode = useCallback((mode: "compact" | "expanded") => {
    setStageMode(mode);
    if (typeof window !== "undefined") {
      try {
        localStorage.setItem(ORB_MODE_STORAGE_KEY, mode);
      } catch {
        // ignore
      }
    }
  }, []);

  const toggleSpeech = useCallback(() => {
    const next = !speechEnabledRef.current;
    speechEnabledRef.current = next;
    setSpeechEnabled(next);
    if (next) {
      setStatus("Spoken replies on");
      drainReply(false);
    } else {
      speechGenerationRef.current += 1;
      speechQueueRef.current = [];
      speakingRef.current = false;
      setIsAssistantSpeaking(false);
      stopNabraAudio();
      setStatus("Spoken replies off");
      maybeResumeListening();
    }
  }, [drainReply, maybeResumeListening]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearSilenceTimer();
      stopRecognition();
      stopAudioAnalyser();
      stopNabraAudio();
    };
  }, [clearSilenceTimer, stopAudioAnalyser, stopRecognition]);

  useEffect(() => {
    if (!connected) {
      stopRecognition();
      setStatus("Chat is reconnecting");
    } else {
      maybeResumeListening();
    }
  }, [connected, maybeResumeListening, stopRecognition]);

  useEffect(() => {
    if (!liveEnabled && !speechEnabled) {
      feed.close();
      return;
    }
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = async () => {
      try {
        await feed.connect(channel);
      } catch {
        if (!disposed && feed.lastCloseCode === null) scheduleReconnect();
      }
    };
    const scheduleReconnect = () => {
      if (disposed || reconnectTimer || attempt >= EVENTS_MAX_RECONNECT_ATTEMPTS) return;
      const delay = eventsReconnectDelayMs(attempt++);
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        void connect();
      }, delay);
    };
    const offClose = feed.onClose((code) => {
      if (code !== undefined && isEventsAuthRejection(code)) return;
      if (shouldRetryEventsClose(code)) scheduleReconnect();
    });
    const offState = feed.onState((state) => {
      if (state === "open") attempt = 0;
    });
    const offStart = feed.on("message.start", () => {
      assistantBusyRef.current = true;
      waitingForReplyRef.current = true;
      rawReplyRef.current = "";
      spokenIndexRef.current = 0;
      speechGenerationRef.current += 1;
      speechQueueRef.current = [];
      speakingRef.current = false;
      setIsAssistantSpeaking(false);
      stopRecognition();
      stopNabraAudio();
      setDraft("");
      setStatus("Hermes is working");
    });
    const offDelta = feed.on("message.delta", (event) => {
      const text = typeof event.payload?.text === "string" ? event.payload.text : "";
      if (!text) return;
      rawReplyRef.current += text;
      if (speechEnabledRef.current) drainReply(false);
    });
    const offComplete = feed.on("message.complete", (event) => {
      const finalText = typeof event.payload?.text === "string" ? event.payload.text : "";
      if (finalText && finalText.startsWith(rawReplyRef.current)) {
        rawReplyRef.current = finalText;
      } else if (!rawReplyRef.current && finalText) {
        rawReplyRef.current = finalText;
      }
      assistantBusyRef.current = false;
      waitingForReplyRef.current = false;
      if (speechEnabledRef.current && event.payload?.status !== "error") {
        drainReply(true);
      } else {
        setStatus("Voice ready");
        maybeResumeListening();
      }
    });

    void connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      offClose();
      offState();
      offStart();
      offDelta();
      offComplete();
      feed.close();
    };
  }, [channel, drainReply, feed, liveEnabled, maybeResumeListening, speechEnabled, stopRecognition]);

  const voiceSupported = speechRecognitionConstructor() !== null;
  const langLabel =
    languageMode === "en" ? "EN" : languageMode === "ar" ? "عربي" : "Auto";
  const langTitle =
    languageMode === "en"
      ? "Language: English (click to switch to Arabic)"
      : languageMode === "ar"
        ? "Language: Arabic - مصرية (click to switch to Auto)"
        : "Language: Auto-Adaptive (click to switch to English)";

  return (
    <div className="mb-2 flex shrink-0 flex-col gap-1.5" style={{ color: foreground }}>
      {stageMode === "expanded" ? (
        <div className="relative w-full overflow-hidden rounded-xl border border-[#00d2c4]/30 bg-gradient-to-b from-[#00d2c4]/15 via-black/70 to-black/90 p-3 shadow-2xl backdrop-blur-md">
          <div className="flex items-center justify-between border-b border-[#00d2c4]/20 pb-2">
            <div className="flex items-center gap-2">
              <span className="size-2 rounded-full bg-[#00d2c4] animate-pulse" />
              <span className="font-mono text-xs font-bold tracking-widest text-[#00d2c4] uppercase">
                Hermes Quantum Stage
              </span>
              <span className="rounded bg-[#00d2c4]/15 px-1.5 py-0.5 font-mono text-[10px] text-[#00d2c4]/90">
                {listening ? "LISTENING" : isAssistantSpeaking ? "SPEAKING" : "STANDBY"}
              </span>
            </div>
            <Button
              ghost
              size="sm"
              onClick={() => toggleStageMode("compact")}
              title="Minimize to compact bar"
              aria-label="Minimize 3D Voice Stage"
              className="h-6 w-6 p-0 border border-white/10 hover:border-[#00d2c4]/40"
            >
              <Minimize2 className="h-3 w-3 text-white/70" />
            </Button>
          </div>

          <div className="relative my-2 flex h-[190px] w-full items-center justify-center overflow-hidden">
            <JarvisUltronVoiceOrb
              themeMode="hermes"
              isActive={liveEnabled || speechEnabled}
              isSpeaking={isAssistantSpeaking}
              isUserSpeaking={listening && Boolean(draft)}
              analyser={micAnalyser}
              className="w-full"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2 border-t border-white/10 pt-2 text-xs">
            <Button
              ghost
              size="sm"
              onClick={toggleLive}
              disabled={!voiceSupported}
              aria-pressed={liveEnabled}
              title={
                voiceSupported
                  ? "Toggle hands-free speech recognition"
                  : "Speech recognition unavailable"
              }
              className="h-7 gap-1.5 border border-current/25 px-2 hover:border-[#00d2c4]/50"
            >
              {liveEnabled ? <Mic className="h-3.5 w-3.5 text-[#00d2c4]" /> : <MicOff className="h-3.5 w-3.5" />}
              {listening ? "Listening" : liveEnabled ? "Live mic" : "Mic"}
            </Button>
            <Button
              ghost
              size="sm"
              onClick={toggleLanguage}
              title={langTitle}
              aria-label={`Speech recognition language: ${langLabel}`}
              className="h-7 gap-1.5 border border-current/25 px-2 font-mono text-[11px]"
            >
              <Globe className="h-3.5 w-3.5 opacity-80 text-[#00d2c4]" />
              <span>{langLabel}</span>
            </Button>
            <Button
              ghost
              size="sm"
              onClick={toggleSpeech}
              aria-pressed={speechEnabled}
              title="Toggle spoken Hermes replies"
              className="h-7 gap-1.5 border border-current/25 px-2"
            >
              {speechEnabled ? <Volume2 className="h-3.5 w-3.5 text-[#00d2c4]" /> : <VolumeX className="h-3.5 w-3.5" />}
              {speechEnabled ? "Voice on" : "Voice off"}
            </Button>
            {draft ? (
              <Button
                ghost
                size="sm"
                onClick={() => void submitCurrentDraft()}
                title="Send now without waiting for pause"
                className="h-7 gap-1 border border-success/40 bg-success/15 px-2 text-success hover:bg-success/25"
              >
                <Send className="h-3.5 w-3.5" />
                <span>Send{pauseCountdown ? ` (${pauseCountdown}s)` : ""}</span>
              </Button>
            ) : null}
            <span className="min-w-0 flex-1 truncate opacity-75" role="status">
              {draft || status}
            </span>
          </div>
        </div>
      ) : (
        <div
          className="flex shrink-0 flex-wrap items-center gap-2 rounded-lg border border-[#00d2c4]/25 bg-black/40 px-2 py-1.5 text-xs shadow-lg backdrop-blur-md"
          aria-label="Hermes chat voice controls"
        >
          {/* Mini 3D Voice Orb */}
          <div
            className="relative flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-full border border-[#00d2c4]/40 bg-black/60 shadow-[0_0_12px_rgba(0,210,196,0.3)] cursor-pointer hover:scale-105 transition-transform"
            onClick={() => toggleStageMode("expanded")}
            title="Click to open full 3D Quantum Voice Stage"
          >
            <JarvisUltronVoiceOrb
              isMini
              themeMode="hermes"
              isActive={liveEnabled || speechEnabled}
              isSpeaking={isAssistantSpeaking}
              isUserSpeaking={listening && Boolean(draft)}
              analyser={micAnalyser}
              className="h-full w-full pointer-events-none"
            />
          </div>

          <Button
            ghost
            size="sm"
            onClick={toggleLive}
            disabled={!voiceSupported}
            aria-pressed={liveEnabled}
            title={
              voiceSupported
                ? "Toggle hands-free speech recognition"
                : "Speech recognition unavailable"
            }
            className="h-7 gap-1.5 border border-current/25 px-2 hover:border-[#00d2c4]/50"
          >
            {liveEnabled ? <Mic className="h-3.5 w-3.5 text-[#00d2c4]" /> : <MicOff className="h-3.5 w-3.5" />}
            {listening ? "Listening" : liveEnabled ? "Live mic" : "Mic"}
          </Button>
          <Button
            ghost
            size="sm"
            onClick={toggleLanguage}
            title={langTitle}
            aria-label={`Speech recognition language: ${langLabel}`}
            className="h-7 gap-1.5 border border-current/25 px-2 font-mono text-[11px]"
          >
            <Globe className="h-3.5 w-3.5 opacity-80 text-[#00d2c4]" />
            <span>{langLabel}</span>
          </Button>
          <Button
            ghost
            size="sm"
            onClick={toggleSpeech}
            aria-pressed={speechEnabled}
            title="Toggle spoken Hermes replies"
            className="h-7 gap-1.5 border border-current/25 px-2"
          >
            {speechEnabled ? <Volume2 className="h-3.5 w-3.5 text-[#00d2c4]" /> : <VolumeX className="h-3.5 w-3.5" />}
            {speechEnabled ? "Voice on" : "Voice off"}
          </Button>
          {draft ? (
            <Button
              ghost
              size="sm"
              onClick={() => void submitCurrentDraft()}
              title="Send now without waiting for pause"
              className="h-7 gap-1 border border-success/40 bg-success/15 px-2 text-success hover:bg-success/25"
            >
              <Send className="h-3.5 w-3.5" />
              <span>Send{pauseCountdown ? ` (${pauseCountdown}s)` : ""}</span>
            </Button>
          ) : null}
          <span className="min-w-0 flex-1 truncate opacity-75" role="status">
            {draft || status}
          </span>
          <Button
            ghost
            size="sm"
            onClick={() => toggleStageMode("expanded")}
            title="Expand 3D Neural Voice Stage"
            aria-label="Expand 3D Voice Stage"
            className="h-7 w-7 p-0 border border-current/20 text-[#00d2c4] hover:border-[#00d2c4]/60"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
    </div>
  );
}


