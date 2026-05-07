export type HandsFreeState =
  | "off"
  | "listening"
  | "recording"
  | "transcribing"
  | "thinking"
  | "speaking";

export interface HandsFreeMicStats {
  level: number;
  noise: number;
  speaking: boolean;
  threshold: number;
}

export const HANDS_FREE_INITIAL_NOISE_FLOOR = 0.01;
export const HANDS_FREE_NOISE_ALPHA = 0.035;
export const HANDS_FREE_MAX_UTTERANCE_MS = 30_000;
export const HANDS_FREE_CHUNK_MAX_CHARS = 650;
export const HANDS_FREE_STREAM_CHUNK_MIN_CHARS = 520;
export const HANDS_FREE_METER_UPDATE_MS = 120;
export const HANDS_FREE_BASE_SPEECH_RMS = 0.022;
export const HANDS_FREE_BASE_SILENCE_RMS = 0.012;
export const HANDS_FREE_SPEECH_DELTA_RMS = 0.012;
export const HANDS_FREE_SILENCE_DELTA_RMS = 0.006;
export const HANDS_FREE_MIN_RECORDING_MS = 650;
export const HANDS_FREE_SILENCE_MS = 1500;
export const HANDS_FREE_MAX_SPOKEN_CHARS = 650;
export const HANDS_FREE_TRUNCATION_NOTICE = "Le detail est dans le chat.";
export const HANDS_FREE_SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=";

export const DEFAULT_HANDS_FREE_MIC_STATS: HandsFreeMicStats = {
  level: 0,
  noise: HANDS_FREE_INITIAL_NOISE_FLOOR,
  speaking: false,
  threshold: HANDS_FREE_BASE_SPEECH_RMS,
};

export function rmsFromTimeDomain(data: Uint8Array<ArrayBuffer>): number {
  let sum = 0;
  for (const sample of data) {
    const centered = (sample - 128) / 128;
    sum += centered * centered;
  }
  return Math.sqrt(sum / Math.max(1, data.length));
}

export function handsFreeSpeechThreshold(noiseFloor: number): number {
  return Math.max(HANDS_FREE_BASE_SPEECH_RMS, noiseFloor + HANDS_FREE_SPEECH_DELTA_RMS);
}

export function handsFreeSilenceThreshold(noiseFloor: number): number {
  return Math.max(HANDS_FREE_BASE_SILENCE_RMS, noiseFloor + HANDS_FREE_SILENCE_DELTA_RMS);
}

export function normalizeTranscriptCommand(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ");
}

export function isHandsFreeStopCommand(text: string): boolean {
  const normalized = normalizeTranscriptCommand(text);
  return [
    "stop",
    "pause",
    "arrete",
    "arretes",
    "attends",
    "silence",
    "tais toi",
    "coupe",
    "stoppe",
  ].includes(normalized);
}

export function isHandsFreeDisableCommand(text: string): boolean {
  const normalized = normalizeTranscriptCommand(text);
  return [
    "arrete le mode vocal",
    "arretes le mode vocal",
    "coupe le mode vocal",
    "desactive le mode vocal",
    "quitte le mode vocal",
    "stop hands free",
    "stop mode vocal",
    "stoppe le mode vocal",
  ].includes(normalized);
}

export function sanitizeHandsFreeSpeechText(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^MEDIA:.*$/gim, " ")
    .replace(/^\s*\[\[audio_as_voice\]\]\s*$/gim, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[`*_>#-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function prepareHandsFreeSpeech(text: string): string {
  let clean = sanitizeHandsFreeSpeechText(text);

  if (clean.length <= HANDS_FREE_MAX_SPOKEN_CHARS) {
    return clean;
  }

  const slice = clean.slice(0, HANDS_FREE_MAX_SPOKEN_CHARS);
  const boundary = Math.max(
    slice.lastIndexOf("."),
    slice.lastIndexOf("!"),
    slice.lastIndexOf("?"),
    slice.lastIndexOf("\n"),
  );
  const minimumBoundary = Math.min(420, Math.floor(HANDS_FREE_MAX_SPOKEN_CHARS * 0.7));
  clean = slice.slice(
    0,
    boundary > minimumBoundary ? boundary + 1 : HANDS_FREE_MAX_SPOKEN_CHARS,
  ).trim();
  return `${clean} ${HANDS_FREE_TRUNCATION_NOTICE}`;
}

export function splitHandsFreeSpeech(text: string): string[] {
  const clean = prepareHandsFreeSpeech(text);
  if (!clean) return [];

  const sentences = clean.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [clean];
  const chunks: string[] = [];
  let current = "";

  for (const rawSentence of sentences) {
    const sentence = rawSentence.trim();
    if (!sentence) continue;
    if (!current) {
      current = sentence;
      continue;
    }
    if (`${current} ${sentence}`.length <= HANDS_FREE_CHUNK_MAX_CHARS) {
      current = `${current} ${sentence}`;
      continue;
    }
    chunks.push(current);
    current = sentence;
  }

  if (current) chunks.push(current);
  return chunks.flatMap((chunk) => {
    if (chunk.length <= HANDS_FREE_CHUNK_MAX_CHARS * 1.5) return [chunk];
    const parts: string[] = [];
    for (let start = 0; start < chunk.length; start += HANDS_FREE_CHUNK_MAX_CHARS) {
      parts.push(chunk.slice(start, start + HANDS_FREE_CHUNK_MAX_CHARS).trim());
    }
    return parts.filter(Boolean);
  });
}

export function takeReadyHandsFreeSpeechChunks(
  text: string,
  force: boolean,
): { chunks: string[]; rest: string } {
  let rest = text;
  const chunks: string[] = [];

  while (rest.trim()) {
    rest = rest.trimStart();
    if (!force && rest.length < HANDS_FREE_STREAM_CHUNK_MIN_CHARS) break;

    const slice = rest.slice(0, HANDS_FREE_CHUNK_MAX_CHARS);
    const boundary = Math.max(
      slice.lastIndexOf("."),
      slice.lastIndexOf("!"),
      slice.lastIndexOf("?"),
      slice.lastIndexOf(","),
      slice.lastIndexOf(";"),
      slice.lastIndexOf(":"),
      slice.lastIndexOf(" "),
    );
    const minimumBoundary = force
      ? 0
      : Math.min(HANDS_FREE_STREAM_CHUNK_MIN_CHARS, Math.floor(HANDS_FREE_CHUNK_MAX_CHARS * 0.75));

    if (boundary > minimumBoundary || rest.length >= HANDS_FREE_CHUNK_MAX_CHARS) {
      const end = boundary > minimumBoundary ? boundary + 1 : HANDS_FREE_CHUNK_MAX_CHARS;
      chunks.push(rest.slice(0, end).trim());
      rest = rest.slice(end);
      continue;
    }

    if (force) {
      chunks.push(rest.trim());
      rest = "";
      continue;
    }

    break;
  }

  if (force && rest.trim()) {
    chunks.push(rest.trim());
    rest = "";
  }

  return {
    chunks: chunks.map(sanitizeHandsFreeSpeechText).filter(Boolean),
    rest,
  };
}

export function handsFreeStateLabel(state: HandsFreeState): string {
  switch (state) {
    case "listening":
      return "ecoute";
    case "recording":
      return "parle";
    case "transcribing":
      return "stt";
    case "thinking":
      return "agent";
    case "speaking":
      return "reponse";
    default:
      return "hands-free";
  }
}

export function handsFreeStatusText(state: HandsFreeState): string {
  return handsFreeStateLabel(state);
}

export function handsFreeMeterPercent(stats: HandsFreeMicStats): number {
  const denominator = Math.max(0.001, stats.threshold * 1.6);
  return Math.max(3, Math.min(100, Math.round((stats.level / denominator) * 100)));
}
