import { Mic, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@nous-research/ui/ui/components/button";

const RESTART_DELAY_MS = 250;
const MAX_RESTARTS = 6;

type VoiceState = "idle" | "listening" | "sent" | "error";

interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly 0: { readonly transcript: string };
}

interface SpeechRecognitionEventLike {
  readonly resultIndex: number;
  readonly results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  start(): void;
  abort(): void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

interface ChatVoiceControlProps {
  connected: boolean;
  submit: (transcript: string) => void;
  onBargeIn?: () => void;
}

function recognitionConstructor(): SpeechRecognitionConstructor | undefined {
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
}

export function ChatVoiceControl({ connected, submit, onBargeIn }: ChatVoiceControlProps) {
  const [state, setState] = useState<VoiceState>("idle");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [error, setError] = useState("");
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const listeningRef = useRef(false);
  const fatalRef = useRef(false);
  const restartCountRef = useRef(0);
  const restartTimerRef = useRef<number | null>(null);
  const generationRef = useRef(0);
  const finalRef = useRef("");
  const interimRef = useRef("");

  const clearRestart = useCallback(() => {
    if (restartTimerRef.current !== null) window.clearTimeout(restartTimerRef.current);
    restartTimerRef.current = null;
  }, []);

  const updateFinal = (value: string) => {
    finalRef.current = value;
    setFinalTranscript(value);
  };

  const updateInterim = (value: string) => {
    interimRef.current = value;
    setInterimTranscript(value);
  };

  function startRecognition(generation: number) {
    if (!listeningRef.current || generation !== generationRef.current) return;
    const Constructor = recognitionConstructor();
    if (!Constructor) {
      listeningRef.current = false;
      setError("Chrome speech recognition is unavailable. Use Gboard typed input below.");
      setState("error");
      return;
    }

    const recognition = new Constructor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      if (generation !== generationRef.current || !listeningRef.current) return;
      let appendedFinal = "";
      let nextInterim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result?.[0]?.transcript ?? "";
        if (result?.isFinal) appendedFinal += text;
        else nextInterim += text;
      }
      if (appendedFinal) updateFinal(`${finalRef.current}${appendedFinal}`);
      updateInterim(nextInterim);
      setError("");
    };
    recognition.onerror = (event) => {
      if (generation !== generationRef.current) return;
      const fatal = event.error === "not-allowed" || event.error === "service-not-allowed";
      if (fatal) {
        fatalRef.current = true;
        listeningRef.current = false;
        setError(event.error === "not-allowed"
          ? "Microphone permission was denied. Allow mic access in Chrome, then tap again."
          : "Chrome speech recognition is not allowed on this device.");
        setState("error");
      } else {
        setError(`Speech recognition error: ${event.error}. Retrying…`);
      }
    };
    recognition.onend = () => {
      if (generation !== generationRef.current || !listeningRef.current || fatalRef.current) return;
      if (restartCountRef.current >= MAX_RESTARTS) {
        listeningRef.current = false;
        setError("Speech recognition kept stopping. Tap to resume or use Gboard typed input below.");
        setState("error");
        return;
      }
      const delay = RESTART_DELAY_MS * 2 ** Math.min(restartCountRef.current, 3);
      restartCountRef.current += 1;
      restartTimerRef.current = window.setTimeout(() => startRecognition(generation), delay);
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      if (generation !== generationRef.current) return;
      listeningRef.current = false;
      setError("Could not start Chrome speech recognition. Tap to try again.");
      setState("error");
    }
  }

  const begin = () => {
    if (!connected) {
      setError("Chat is not connected yet.");
      setState("error");
      return;
    }
    onBargeIn?.();
    clearRestart();
    fatalRef.current = false;
    restartCountRef.current = 0;
    listeningRef.current = true;
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    setError("");
    setState("listening");
    startRecognition(generation);
  };

  const commit = () => {
    const transcript = `${finalRef.current} ${interimRef.current}`.replace(/\s+/g, " ").trim();
    if (!transcript) {
      setError("No speech heard yet — keep talking, then tap anywhere to send.");
      return;
    }
    listeningRef.current = false;
    generationRef.current += 1;
    clearRestart();
    recognitionRef.current?.abort();
    recognitionRef.current = null;
    submit(transcript);
    updateFinal("");
    updateInterim("");
    setError("");
    setState("sent");
  };

  const cancel = useCallback(() => {
    listeningRef.current = false;
    generationRef.current += 1;
    clearRestart();
    recognitionRef.current?.abort();
    recognitionRef.current = null;
    updateFinal("");
    updateInterim("");
    setError("");
    setState("idle");
  }, [clearRestart]);

  useEffect(() => cancel, [cancel]);

  const transcript = `${finalTranscript}${interimTranscript}`.trim();
  const headline = state === "listening"
    ? (transcript || "LISTENING…")
    : state === "sent"
      ? "SENT"
      : "TAP TO TALK";

  return (
    <div className="relative flex max-h-[52dvh] min-h-[42dvh] shrink-0 flex-col overflow-hidden border border-current/25 bg-black/30 pb-[env(safe-area-inset-bottom)] text-white lg:min-h-44">
      <button
        type="button"
        aria-label="Browser voice input"
        onClick={state === "listening" ? commit : begin}
        className="flex min-h-[42dvh] w-full touch-manipulation flex-1 flex-col items-center justify-center gap-5 overflow-hidden px-5 py-8 text-center active:bg-white/10 lg:min-h-44"
      >
        <Mic className={state === "listening" ? "h-12 w-12 shrink-0 animate-pulse text-red-400 motion-reduce:animate-none" : "h-12 w-12 shrink-0"} />
        <span className="max-h-[28dvh] w-full overflow-y-auto whitespace-pre-wrap break-words text-2xl font-semibold leading-tight sm:text-3xl">
          {headline}
        </span>
        <span className="text-sm tracking-wide text-white/70">
          {state === "listening" ? "TAP ANYWHERE TO SEND" : "Chrome live speech • Gboard typed fallback remains below"}
        </span>
      </button>
      <Button
        ghost
        aria-label="Cancel browser voice input"
        onClick={cancel}
        className="absolute right-2 top-2 min-h-11 min-w-11 touch-manipulation px-2"
        prefix={<X className="h-5 w-5" />}
      >
        Cancel
      </Button>
      {error && <div role="alert" className="px-4 pb-3 text-center text-sm text-red-300">{error}</div>}
    </div>
  );
}
