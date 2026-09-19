/**
 * Speech & Text Punctuation Utility for Web Speech Synthesis API.
 * Local Female Voice is permanently purged — all female and Arabic speech
 * is exclusively handled by the ElevenLabs API model (eleven_multilingual_v2).
 */

export interface LanguageDetectionResult {
  isArabicPredominant: boolean;
  isEnglishPredominant: boolean;
  detectedLanguage: 'Arabic' | 'English';
  arabicCount: number;
  englishCount: number;
  totalLetters: number;
  arabicRatio: number;
}

export const detectLanguageContent = (text: string): LanguageDetectionResult => {
  if (!text) {
    return {
      isArabicPredominant: false,
      isEnglishPredominant: true,
      detectedLanguage: 'English',
      arabicCount: 0,
      englishCount: 0,
      totalLetters: 0,
      arabicRatio: 0,
    };
  }

  const arabicChars = (text.match(/[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/g) || []).length;
  const englishChars = (text.match(/[a-zA-Z]/g) || []).length;
  const totalLetters = arabicChars + englishChars;

  if (totalLetters === 0) {
    const hasArabic = /[\u0600-\u06FF]/.test(text);
    return {
      isArabicPredominant: hasArabic,
      isEnglishPredominant: !hasArabic,
      detectedLanguage: hasArabic ? 'Arabic' : 'English',
      arabicCount: arabicChars,
      englishCount: englishChars,
      totalLetters: 0,
      arabicRatio: hasArabic ? 1 : 0,
    };
  }

  const arabicRatio = arabicChars / totalLetters;
  const isArabicPredominant = arabicRatio >= 0.35 || (arabicChars > 0 && englishChars === 0);
  const isEnglishPredominant = !isArabicPredominant;

  return {
    isArabicPredominant,
    isEnglishPredominant,
    detectedLanguage: isArabicPredominant ? 'Arabic' : 'English',
    arabicCount: arabicChars,
    englishCount: englishChars,
    totalLetters,
    arabicRatio,
  };
};

export const sanitizeTextForSpeech = (text: string): string => {
  if (!text) return '';

  return text
    .replace(/^[\s]*[*•\-+]\s+/gm, 'Item: ')
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/--+/g, ', ')
    .replace(/[\*\#]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
};

export const isArabic = (text: string): boolean => {
  return detectLanguageContent(text).isArabicPredominant;
};

let cachedVoices: SpeechSynthesisVoice[] = [];
let voicesPromise: Promise<SpeechSynthesisVoice[]> | null = null;

export const getVoicesSafely = (): Promise<SpeechSynthesisVoice[]> => {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return Promise.resolve([]);

  if (cachedVoices.length > 0) return Promise.resolve(cachedVoices);

  const existing = window.speechSynthesis.getVoices();
  if (existing.length > 0) {
    cachedVoices = existing;
    return Promise.resolve(existing);
  }

  if (!voicesPromise) {
    voicesPromise = new Promise<SpeechSynthesisVoice[]>((resolve) => {
      let settled = false;
      const settle = () => {
        if (settled) return;
        settled = true;
        const v = window.speechSynthesis.getVoices();
        if (v.length > 0) cachedVoices = v;
        window.speechSynthesis.removeEventListener('voiceschanged', settle);
        resolve(v);
      };
      window.speechSynthesis.addEventListener('voiceschanged', settle);
      setTimeout(settle, 1000);
    });
  }
  return voicesPromise;
};

// Eagerly trigger voice loading if in browser
if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
  try {
    const v = window.speechSynthesis.getVoices();
    if (v.length > 0) cachedVoices = v;
    window.speechSynthesis.addEventListener('voiceschanged', () => {
      const updated = window.speechSynthesis.getVoices();
      if (updated.length > 0) cachedVoices = updated;
    });
  } catch {}
}

export const pickArabicVoice = (
  voices: SpeechSynthesisVoice[],
  persona?: 'jarvis' | 'gwen'
): SpeechSynthesisVoice | undefined => {
  const isGwen = persona === 'gwen';

  if (isGwen) {
    const femaleArabic = voices.filter((v) => {
      const name = v.name.toLowerCase();
      const isFemale =
        name.includes('female') ||
        name.includes('salma') ||
        name.includes('hoda') ||
        name.includes('laila') ||
        name.includes('zeina') ||
        name.includes('mariam') ||
        name.includes('fatima') ||
        name.includes('zira') ||
        name.includes('jenny');
      return isFemale && v.lang.toLowerCase().includes('ar');
    });
    if (femaleArabic.length > 0) return femaleArabic[0];

    const anyArabicNonMale = voices.filter(
      (v) =>
        v.lang.toLowerCase().includes('ar') &&
        !v.name.toLowerCase().includes('male') &&
        !v.name.toLowerCase().includes('shakir') &&
        !v.name.toLowerCase().includes('tarik')
    );
    if (anyArabicNonMale.length > 0) return anyArabicNonMale[0];
    return voices.find((v) => v.lang.toLowerCase().includes('ar'));
  }

  // Jarvis: Male Arabic voices
  const maleArabicVoices = voices.filter((v) => {
    const name = v.name.toLowerCase();
    const isFemale =
      name.includes('female') ||
      name.includes('salma') ||
      name.includes('hoda') ||
      name.includes('laila') ||
      name.includes('zeina') ||
      name.includes('mariam') ||
      name.includes('fatima') ||
      name.includes('zira') ||
      name.includes('jenny');
    if (isFemale) return false;

    return (
      v.lang.toLowerCase().includes('ar') ||
      name.includes('arabic') ||
      name.includes('shakir') ||
      name.includes('tarik') ||
      name.includes('maged') ||
      name.includes('hamed') ||
      name.includes('bassel') ||
      name.includes('naayf') ||
      name.includes('male')
    );
  });

  if (maleArabicVoices.length === 0) {
    return voices.find((v) => v.lang.toLowerCase().includes('ar'));
  }

  const maleEg = maleArabicVoices.find(
    (v) =>
      v.lang.toLowerCase().includes('ar-eg') &&
      (v.name.toLowerCase().includes('shakir') || v.name.toLowerCase().includes('male'))
  );
  if (maleEg) return maleEg;

  const generalMale = maleArabicVoices.find(
    (v) =>
      v.name.toLowerCase().includes('shakir') ||
      v.name.toLowerCase().includes('male') ||
      v.name.toLowerCase().includes('tarik') ||
      v.name.toLowerCase().includes('hamed') ||
      v.name.toLowerCase().includes('maged') ||
      v.name.toLowerCase().includes('bassel') ||
      v.name.toLowerCase().includes('naayf')
  );
  if (generalMale) return generalMale;

  return maleArabicVoices[0];
};

export const pickEnglishVoice = (
  voices: SpeechSynthesisVoice[],
  persona?: 'jarvis' | 'gwen'
): SpeechSynthesisVoice | undefined => {
  const isGwen = persona === 'gwen';

  if (isGwen) {
    const femaleEnglish = voices.filter((v) => {
      const name = v.name.toLowerCase();
      const isFemaleName =
        name.includes('female') ||
        name.includes('zira') ||
        name.includes('jenny') ||
        name.includes('samantha') ||
        name.includes('victoria') ||
        name.includes('karen') ||
        name.includes('linda') ||
        name.includes('susan') ||
        name.includes('ava') ||
        name.includes('emma');
      return isFemaleName && v.lang.toLowerCase().startsWith('en');
    });
    if (femaleEnglish.length > 0) return femaleEnglish[0];

    const nonMale = voices.filter(
      (v) =>
        v.lang.toLowerCase().startsWith('en') &&
        !v.name.toLowerCase().includes('male') &&
        !v.name.toLowerCase().includes('guy') &&
        !v.name.toLowerCase().includes('david') &&
        !v.name.toLowerCase().includes('george')
    );
    if (nonMale.length > 0) return nonMale[0];
    return voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  }

  // Jarvis: Male English voices
  const maleEnglishVoices = voices.filter((v) => {
    const name = v.name.toLowerCase();
    const isFemale =
      name.includes('female') ||
      name.includes('zira') ||
      name.includes('jenny') ||
      name.includes('samantha') ||
      name.includes('victoria') ||
      name.includes('karen') ||
      name.includes('linda') ||
      name.includes('susan') ||
      name.includes('ava') ||
      name.includes('emma');
    if (isFemale) return false;
    return v.lang.toLowerCase().startsWith('en');
  });

  if (maleEnglishVoices.length === 0) {
    return voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  }

  // Jarvis British / sophisticated male voice
  const britishMale = maleEnglishVoices.find(
    (v) =>
      (v.lang.toLowerCase().includes('gb') || v.lang.toLowerCase().includes('uk')) &&
      (v.name.toLowerCase().includes('male') ||
        v.name.toLowerCase().includes('george') ||
        v.name.toLowerCase().includes('oliver') ||
        v.name.toLowerCase().includes('daniel'))
  );
  if (britishMale) return britishMale;

  const naturalMale = maleEnglishVoices.find(
    (v) =>
      v.name.toLowerCase().includes('david') ||
      v.name.toLowerCase().includes('mark') ||
      v.name.toLowerCase().includes('guy') ||
      v.name.toLowerCase().includes('natural') ||
      v.name.toLowerCase().includes('male')
  );
  if (naturalMale) return naturalMale;

  return maleEnglishVoices[0];
};

export const splitTextIntoSentences = (text: string): string[] => {
  if (!text) return [];
  // Split on periods, exclamation, question marks, Arabic comma/semicolon, or newlines
  const rawChunks = text.split(/([.!?،؟؛\n]+)/g);
  const sentences: string[] = [];
  let buffer = '';

  for (let i = 0; i < rawChunks.length; i += 2) {
    const segment = rawChunks[i] || '';
    const punctuation = rawChunks[i + 1] || '';
    const combined = (segment + punctuation).trim();
    if (!combined) continue;

    buffer = buffer ? `${buffer} ${combined}` : combined;
    if (buffer.length > 80 || punctuation) {
      sentences.push(buffer);
      buffer = '';
    }
  }

  if (buffer.trim()) {
    sentences.push(buffer.trim());
  }

  return sentences.filter((s) => s.trim().length > 0);
};

/**
 * Strips completed and active streaming reasoning (<think>) blocks so they are never spoken.
 */
export const cleanSpokenText = (text: string): string => {
  if (!text) return '';
  let clean = text.replace(/<think[\s\S]*?<\/think>/gi, '');
  return clean.replace(/<think[\s\S]*/gi, '');
};

export interface ExtractedSentence {
  sentence: string;
  nextIndex: number;
}

/**
 * Incrementally extracts the next clean, spoken sentence or clause from streaming LLM output.
 * Starts speech on the very first clause or sentence boundary (<200ms) for real-time voice latency.
 */
export const extractNextSpokenSentence = (
  cleanText: string,
  startIndex: number
): ExtractedSentence | null => {
  if (!cleanText || startIndex >= cleanText.length) return null;

  const unchunked = cleanText.slice(startIndex);
  const isFirstChunk = startIndex === 0;

  // 1. Check for sentence terminators and clause separators, prioritizing whichever comes first
  const sentenceMatch = unchunked.match(/([.!?؛؟\n]+)(?:\s+|$)/);
  const clauseMatch = unchunked.match(/([,،:;—]+)(?:\s+|$)/);

  // If a clause separator occurs before any sentence terminator (e.g. "Understood sir, I will..." or "تمام،")
  if (
    clauseMatch &&
    clauseMatch.index !== undefined &&
    (!sentenceMatch || sentenceMatch.index === undefined || clauseMatch.index < sentenceMatch.index)
  ) {
    const boundary = clauseMatch.index + clauseMatch[1].length;
    const raw = unchunked.slice(0, boundary).trim();
    const wordCount = raw.split(/\s+/).filter(Boolean).length;
    // For the very first chunk, emit on ANY clause boundary (even a 1-word greeting like "Yes," or "تمام،")
    // For subsequent chunks, require at least 8 chars or 2 words for natural prosody
    if (isFirstChunk ? raw.length >= 2 : (raw.length >= 8 || wordCount >= 2)) {
      return { sentence: raw, nextIndex: startIndex + boundary };
    }
  }

  // 2. Standard sentence terminators if within prompt speaking distance (<= 32 for first chunk, <= 45 for subsequent)
  if (sentenceMatch && sentenceMatch.index !== undefined) {
    const termIndex = sentenceMatch.index;
    const maxTermDistance = isFirstChunk ? 32 : 45;
    if (termIndex <= maxTermDistance) {
      const boundary = termIndex + sentenceMatch[1].length;
      const raw = unchunked.slice(0, boundary).trim();
      if (raw.length > 0) {
        return { sentence: raw, nextIndex: startIndex + boundary };
      }
    }
  }

  // 3. Ultra-fast initial speech: when starting a turn (startIndex === 0), do not wait for
  // long sentences or distant punctuation. Emit as soon as we have 3-4 words (>= 18 chars)
  // so speech output begins immediately within ~100-200ms while subsequent tokens generate.
  if (isFirstChunk && unchunked.length >= 18) {
    const maxSearch = Math.min(unchunked.length, 30);
    const spaceIndex = unchunked.lastIndexOf(' ', maxSearch);
    if (spaceIndex >= 10) {
      const raw = unchunked.slice(0, spaceIndex).trim();
      if (raw.length > 0) {
        return { sentence: raw, nextIndex: startIndex + spaceIndex + 1 };
      }
    }
  }

  // 4. Natural speaking pause for subsequent continuous text (>= 35 chars)
  if (unchunked.length >= 35) {
    const maxSearch = Math.min(unchunked.length, 45);
    const spaceIndex = unchunked.lastIndexOf(' ', maxSearch);
    if (spaceIndex >= 18) {
      const raw = unchunked.slice(0, spaceIndex).trim();
      if (raw.length > 0) {
        return { sentence: raw, nextIndex: startIndex + spaceIndex + 1 };
      }
    }
  }

  // 5. Final fallback if sentence terminator exists further out but continuous text was under 35 chars
  if (sentenceMatch && sentenceMatch.index !== undefined) {
    const boundary = sentenceMatch.index + sentenceMatch[1].length;
    const raw = unchunked.slice(0, boundary).trim();
    if (raw.length > 0) {
      return { sentence: raw, nextIndex: startIndex + boundary };
    }
  }

  return null;
};

export type QueueTask = (signal: AbortSignal) => Promise<void>;

/**
 * Pipelined sequential audio playback queue for streaming TTS sentences.
 * Ensures sentence audio plays without gaps and can be instantly aborted upon interruption.
 */
export class PipelinedAudioQueue {
  private queue: QueueTask[] = [];
  private isProcessing = false;
  private isAborted = false;
  private currentAbortController: AbortController | null = null;
  private onSpeakingChange?: (speaking: boolean) => void;

  constructor(onSpeakingChange?: (speaking: boolean) => void) {
    this.onSpeakingChange = onSpeakingChange;
  }

  public enqueue(task: QueueTask): void {
    if (this.isAborted) return;
    this.queue.push(task);
    void this.drain();
  }

  private async drain(): Promise<void> {
    if (this.isProcessing || this.isAborted) return;
    this.isProcessing = true;
    this.onSpeakingChange?.(true);

    while (this.queue.length > 0 && !this.isAborted) {
      const nextTask = this.queue.shift();
      if (nextTask) {
        this.currentAbortController = new AbortController();
        try {
          await nextTask(this.currentAbortController.signal);
        } catch (err) {
          console.warn('[PipelinedAudioQueue] sentence play notice:', err);
        } finally {
          this.currentAbortController = null;
        }
      }
    }

    this.isProcessing = false;
    if (this.queue.length === 0 && !this.isAborted) {
      this.onSpeakingChange?.(false);
    }
  }

  public abort(): void {
    this.isAborted = true;
    this.queue = [];
    if (this.currentAbortController) {
      this.currentAbortController.abort();
      this.currentAbortController = null;
    }
    this.onSpeakingChange?.(false);
  }

  public async waitUntilDone(): Promise<void> {
    while ((this.isProcessing || this.queue.length > 0) && !this.isAborted) {
      await new Promise((r) => setTimeout(r, 40));
    }
  }
}

