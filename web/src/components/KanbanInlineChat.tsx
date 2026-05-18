import { Mic, MicOff, Send, Volume2, VolumeX } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { useI18n, type Locale } from "@/i18n";
import { parseSseChunk, type StreamEvent } from "./KanbanInlineChat.utils";

const WHISPER_MODEL = "Xenova/whisper-small";
const SAMPLE_RATE = 16_000;
const DOWNLOAD_ACK_KEY = "hermes.kanban.whisper-small-download-ack";

type InlineMessage = {
  id: string;
  role: "user" | "assistant" | "status" | "error";
  text: string;
  created?: string[];
};

type WhisperPipeline = (
  audio: Float32Array,
  options: { language: string; task: "transcribe" },
) => Promise<string | { text?: string }>;

type AudioContextConstructor = typeof AudioContext;

declare global {
  interface Window {
    webkitAudioContext?: AudioContextConstructor;
  }
}

let whisperPipelinePromise: Promise<WhisperPipeline> | null = null;

const WHISPER_LANGUAGE_BY_LOCALE: Record<Locale, string> = {
  af: "afrikaans",
  de: "german",
  en: "english",
  es: "spanish",
  fr: "french",
  ga: "irish",
  hu: "hungarian",
  it: "italian",
  ja: "japanese",
  ko: "korean",
  pt: "portuguese",
  ru: "russian",
  tr: "turkish",
  uk: "ukrainian",
  zh: "chinese",
  "zh-hant": "chinese",
};

function getAudioContextConstructor(): AudioContextConstructor | null {
  if (typeof window === "undefined") return null;
  return window.AudioContext ?? window.webkitAudioContext ?? null;
}

function voiceSupported(): boolean {
  return Boolean(
    typeof window !== "undefined"
      && typeof navigator.mediaDevices?.getUserMedia === "function"
      && typeof window.MediaRecorder !== "undefined"
      && getAudioContextConstructor()
      && typeof window.OfflineAudioContext !== "undefined",
  );
}

function messageId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function loadWhisperPipeline(): Promise<WhisperPipeline> {
  if (!whisperPipelinePromise) {
    whisperPipelinePromise = import("@xenova/transformers").then(async (mod) => {
      const pipeline = await mod.pipeline("automatic-speech-recognition", WHISPER_MODEL);
      return pipeline as WhisperPipeline;
    });
  }
  return whisperPipelinePromise;
}

async function audioBlobToMono16k(blob: Blob): Promise<Float32Array> {
  const AudioCtx = getAudioContextConstructor();
  if (!AudioCtx) throw new Error("AudioContext is not available");

  const ctx = new AudioCtx();
  try {
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    if (buffer.sampleRate === SAMPLE_RATE) {
      return buffer.getChannelData(0).slice();
    }

    const offline = new OfflineAudioContext(
      1,
      Math.ceil(buffer.duration * SAMPLE_RATE),
      SAMPLE_RATE,
    );
    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  } finally {
    void ctx.close();
  }
}

export type KanbanInlineChatProps = {
  boardId: string;
  onCardCreated?: (cardId: string) => void;
};

export function KanbanInlineChat({ boardId, onCardCreated }: KanbanInlineChatProps) {
  const { locale, t } = useI18n();
  const [text, setText] = useState("");
  const [messages, setMessages] = useState<InlineMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const supported = voiceSupported();

  const appendMessage = useCallback((message: Omit<InlineMessage, "id">) => {
    setMessages((prev) => [...prev, { id: messageId(), ...message }].slice(-10));
  }, []);

  useEffect(() => {
    return () => {
      recorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const playTts = useCallback(async (content: string) => {
    if (!ttsEnabled || !content.trim()) return;
    try {
      const res = await fetch("/api/plugins/kanban/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: content }),
      });
      if (!res.ok) throw new Error(await res.text());
      const body = await res.json() as { data_url?: string };
      if (!body.data_url) return;
      appendMessage({ role: "status", text: t.kanban.ttsPlaying });
      try {
        await new Audio(body.data_url).play();
      } catch (error) {
        appendMessage({ role: "status", text: `${t.kanban.ttsBlocked}: ${String(error)}` });
      }
    } catch (error) {
      appendMessage({ role: "status", text: `${t.kanban.ttsFailed}${String(error)}` });
    }
  }, [
    appendMessage,
    t.kanban.ttsBlocked,
    t.kanban.ttsFailed,
    t.kanban.ttsPlaying,
    ttsEnabled,
  ]);

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    if (event.type === "card_created" && event.card_id) {
      onCardCreated?.(event.card_id);
      appendMessage({
        role: "status",
        text: `${t.kanban.inlineDispatchCreated}${event.card_id}`,
        created: [event.card_id],
      });
      return "";
    }
    if (event.type === "card_status" && event.status) {
      appendMessage({ role: "status", text: event.status });
      return "";
    }
    if (event.type === "error") {
      appendMessage({ role: "error", text: event.content ?? t.kanban.inlineDispatchFailed });
      return "";
    }
    if (event.type === "result" && event.content) {
      appendMessage({ role: "assistant", text: event.content });
      return event.content;
    }
    return "";
  }, [
    appendMessage,
    onCardCreated,
    t.kanban.inlineDispatchCreated,
    t.kanban.inlineDispatchFailed,
  ]);

  const send = useCallback(async (overrideText?: string) => {
    const bodyText = (overrideText ?? text).trim();
    if (!bodyText || busy) return;
    setBusy(true);
    setText("");
    appendMessage({ role: "user", text: bodyText });
    let lastResult = "";
    try {
      const res = await fetch(
        `/api/plugins/kanban/boards/${encodeURIComponent(boardId)}/inline-dispatch?stream=true`,
        {
          method: "POST",
          headers: {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: bodyText }),
        },
      );
      if (!res.ok) throw new Error(await res.text());
      const reader = res.body?.getReader();
      if (!reader) throw new Error("stream response body is unavailable");
      const decoder = new TextDecoder();
      let pending = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(pending);
        pending = parsed.rest;
        for (const event of parsed.events) {
          lastResult = handleStreamEvent(event) || lastResult;
        }
      }
      const parsed = parseSseChunk(`${pending}\n\n`);
      for (const event of parsed.events) {
        lastResult = handleStreamEvent(event) || lastResult;
      }
      await playTts(lastResult);
    } catch (error) {
      appendMessage({ role: "error", text: `${t.kanban.inlineDispatchFailed}${String(error)}` });
    } finally {
      setBusy(false);
    }
  }, [
    appendMessage,
    boardId,
    busy,
    handleStreamEvent,
    playTts,
    t.kanban.inlineDispatchFailed,
    text,
  ]);

  const transcribe = useCallback(async (blob: Blob) => {
    setTranscribing(true);
    try {
      const audio = await audioBlobToMono16k(blob);
      const pipeline = await loadWhisperPipeline();
      const result = await pipeline(audio, {
        language: WHISPER_LANGUAGE_BY_LOCALE[locale] ?? "english",
        task: "transcribe",
      });
      const transcript = typeof result === "string" ? result : (result.text ?? "");
      setText(transcript.trim());
    } catch (error) {
      appendMessage({ role: "error", text: `${t.kanban.microphoneError} ${String(error)}` });
    } finally {
      setTranscribing(false);
    }
  }, [appendMessage, locale, t.kanban.microphoneError]);

  const toggleMic = useCallback(async () => {
    if (!supported) {
      appendMessage({ role: "error", text: t.kanban.unsupportedBrowser });
      return;
    }
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    try {
      if (!window.localStorage.getItem(DOWNLOAD_ACK_KEY)) {
        const ok = window.confirm(t.kanban.whisperDownloadConfirm);
        if (!ok) return;
        window.localStorage.setItem(DOWNLOAD_ACK_KEY, "1");
      }
      await loadWhisperPipeline();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        void transcribe(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
      };
      streamRef.current = stream;
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (error) {
      appendMessage({
        role: "error",
        text: error instanceof DOMException && error.name === "NotAllowedError"
          ? t.kanban.microphonePermissionDenied
          : `${t.kanban.microphoneError} ${String(error)}`,
      });
      setRecording(false);
    }
  }, [
    appendMessage,
    recording,
    supported,
    t.kanban.microphoneError,
    t.kanban.microphonePermissionDenied,
    t.kanban.unsupportedBrowser,
    t.kanban.whisperDownloadConfirm,
    transcribe,
  ]);

  return (
    <section className="hermes-kanban-inline-chat" aria-label={t.kanban.inlineChatHistory}>
      <form
        className="hermes-kanban-inline-chat-form"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder={t.kanban.inlineChatPlaceholder}
          aria-label={t.kanban.inlineChatPlaceholder}
          disabled={busy || transcribing}
        />
        <button
          type="button"
          onClick={() => void toggleMic()}
          disabled={!supported || busy || transcribing}
          aria-label={recording ? t.kanban.microphoneStop : t.kanban.microphoneStart}
          title={supported ? undefined : t.kanban.unsupportedBrowser}
        >
          {recording ? <MicOff aria-hidden="true" /> : <Mic aria-hidden="true" />}
        </button>
        <button
          type="button"
          onClick={() => setTtsEnabled((value) => !value)}
          aria-label={t.kanban.ttsToggle}
          aria-pressed={ttsEnabled}
        >
          {ttsEnabled ? <Volume2 aria-hidden="true" /> : <VolumeX aria-hidden="true" />}
        </button>
        <button type="submit" disabled={busy || transcribing || !text.trim()}>
          <Send aria-hidden="true" />
          <span>{busy ? t.kanban.inlineDispatchSending : t.kanban.send}</span>
        </button>
      </form>
      {!supported ? <div role="status">{t.kanban.unsupportedBrowser}</div> : null}
      {recording ? <div role="status">{t.kanban.recordingIndicator}</div> : null}
      {transcribing ? <div role="status">{t.kanban.voiceTranscribing}</div> : null}
      {messages.length > 0 ? (
        <div aria-live="polite">
          {messages.map((item) => (
            <div key={item.id} data-role={item.role}>
              {item.text}
            </div>
          ))}
          <button type="button" onClick={() => setMessages([])}>
            {t.kanban.clearHistory}
          </button>
        </div>
      ) : null}
    </section>
  );
}

export default KanbanInlineChat;
