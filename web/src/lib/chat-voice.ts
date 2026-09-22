export type VoiceLanguageMode = "en" | "ar" | "auto";

export interface VoiceRecognitionResult {
  isFinal: boolean;
  0?: { transcript?: string };
}

export interface VoiceRecognitionEvent {
  resultIndex: number;
  results: ArrayLike<VoiceRecognitionResult>;
}

export const CONTINUATION_WORDS_AR = [
  "و", "او", "أو", "ثم", "ف", "علشان", "عشان", "بس", "لكن", "يعني",
  "مع", "في", "من", "عن", "على", "الي", "إلى", "إن", "ان", "انك", "لو",
  "لما", "حتى", "قبل", "بعد", "بدل", "زي", "معلش", "طيب", "يا", "ما"
] as const;

export const CONTINUATION_WORDS_EN = [
  "and", "or", "but", "because", "cause", "so", "then", "if", "when",
  "while", "where", "like", "with", "that", "which", "who", "also",
  "actually", "well", "um", "uh", "er", "ah", "the", "a", "an", "to"
] as const;

export function containsArabic(text: string): boolean {
  return /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(text);
}

export function containsLatin(text: string): boolean {
  return /[A-Za-z]/.test(text);
}

export function detectDominantScript(text: string): "ar" | "en" | "neutral" {
  let arCount = 0;
  let enCount = 0;
  for (const char of text) {
    if (/[\u0600-\u06FF]/.test(char)) arCount++;
    else if (/[A-Za-z]/.test(char)) enCount++;
  }
  if (arCount > 0 && arCount >= enCount) return "ar";
  if (enCount > 0 && enCount > arCount) return "en";
  return "neutral";
}

export function getVoicePauseTimeoutMs(textDraft: string, baseMs = 1400): number {
  const trimmed = textDraft.trim();
  if (!trimmed) return baseMs;

  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length === 0) return baseMs;

  const lastWord = words[words.length - 1].toLowerCase().replace(/[،,.:;!؟?]/g, "");
  const isContinuation =
    (CONTINUATION_WORDS_AR as readonly string[]).includes(lastWord) ||
    (CONTINUATION_WORDS_EN as readonly string[]).includes(lastWord) ||
    lastWord.startsWith("و") ||
    lastWord.startsWith("ف");

  const dynamicBonus = isContinuation ? 400 : 0;
  return Math.min(2400, baseMs + dynamicBonus);
}

export function normalizeVoicePrompt(text: string): string {
  return text.replace(/[\r\n\t]+/g, " ").replace(/\s+/g, " ").trim();
}

export function recognitionTranscript(event: VoiceRecognitionEvent): {
  final: string;
  interim: string;
} {
  const final: string[] = [];
  const interim: string[] = [];
  for (let index = event.resultIndex; index < event.results.length; index += 1) {
    const result = event.results[index];
    const text = String(result?.[0]?.transcript ?? "").trim();
    if (!text) continue;
    (result.isFinal ? final : interim).push(text);
  }
  return { final: final.join(" "), interim: interim.join(" ") };
}

interface VoicePromptSocket {
  readyState: number;
  send(data: string): void;
}

export function sendVoicePrompt(
  socket: VoicePromptSocket | null,
  rawText: string,
  sendReturn: (callback: () => void) => void,
  isCurrent: () => boolean,
): boolean {
  const text = normalizeVoicePrompt(rawText);
  if (!text || !socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(text);
  sendReturn(() => {
    if (isCurrent() && socket.readyState === WebSocket.OPEN) socket.send("\r");
  });
  return true;
}

