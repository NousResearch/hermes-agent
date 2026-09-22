import { authedFetch } from "@/lib/api";
import { getVoicesSafely, pickArabicVoice, pickEnglishVoice } from "@/lib/speechUtils";

export const sanitizeTextForSpeech = (text: string): string => {
  if (!text) return '';

  return text
    // Replace markdown bullet items (* item, - item, + item) with clean speech phrasing
    .replace(/^[\s]*[*•\-+]\s+/gm, 'Item: ')
    // Replace inline bold/italic markdown (*word*, **word**, ***word***) keeping the word intact
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    // Replace underscore bold/italic (_word_, __word__) keeping the word intact
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Strip markdown header symbols (### Header)
    .replace(/^#{1,6}\s+/gm, '')
    // Replace inline backticks (`code`) keeping content
    .replace(/`([^`]+)`/g, '$1')
    // Convert multiple hyphens (--) into a natural comma pause
    .replace(/--+/g, ', ')
    // Clean remaining isolated asterisks or hashes that might confuse speech synthesis
    .replace(/[*#]+/g, '')
    // Normalize multiple spaces into single space
    .replace(/\s+/g, ' ')
    .trim();
};

export const formatDisplayContentWithPunctuation = (text: string): string => {
  if (!text) return '';

  return text
    // Strip markdown headers (### Header) keeping the heading text
    .replace(/^#{1,6}\s+/gm, '')
    // Convert markdown bullets (* item, - item, + item) to clean bullets
    .replace(/^[\s]*[•*\-+]\s+/gm, '• ')
    // Strip bold/italic markers (**word**, *word*, ***word***) keeping the word
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    // Strip underscore emphasis (_word_, __word__) keeping the word
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Strip inline backticks (`code`) keeping content
    .replace(/`([^`]+)`/g, '$1')
    // Strip markdown links [text](url) keeping the display text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Strip remaining isolated asterisks, hashes, tildes that are not punctuation
    .replace(/[*#~]+/g, '')
    // Normalize runs of whitespace but preserve single newlines for readable structure
    .replace(/[ \t]+/g, ' ')
    // Normalize 3+ blank lines down to a single blank line
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

let activeAudio: HTMLAudioElement | null = null;

export const stopNabraAudio = (): void => {
  if (activeAudio) {
    try {
      activeAudio.pause();
    } catch {
      // ignore
    }
    activeAudio = null;
  }
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch {
      // ignore
    }
  }
};

export const speakWithNabra = async (
  text: string,
  persona: 'jarvis' | 'gwen' = 'jarvis',
  cancelExisting = true
): Promise<void> => {
  const clean = sanitizeTextForSpeech(text);
  if (!clean || typeof window === 'undefined') return;
  if (cancelExisting) {
    stopNabraAudio();
  }

  const isAr = /[\u0600-\u06FF]/.test(clean);

  // Fast path for Jarvis: Browser Web SpeechSynthesis API gives instant (<50ms) natural speech
  if (persona === 'jarvis' && 'speechSynthesis' in window) {
    try {
      const voices = await getVoicesSafely();
      const voice = isAr
        ? pickArabicVoice(voices, 'jarvis')
        : pickEnglishVoice(voices, 'jarvis');

      await new Promise<void>((resolve) => {
        const utter = new SpeechSynthesisUtterance(clean);
        utter.lang = isAr ? 'ar-EG' : 'en-US';
        utter.rate = 1.05;
        utter.pitch = 0.95;
        if (voice) utter.voice = voice;
        utter.onend = () => resolve();
        utter.onerror = () => resolve();
        if (window.speechSynthesis.paused) {
          try {
            window.speechSynthesis.resume();
          } catch {}
        }
        window.speechSynthesis.speak(utter);
        setTimeout(resolve, 25000);
      });
      return;
    } catch (e) {
      console.warn('Browser speech instant path notice:', e);
    }
  }

  // Tier 2: Try Hermes Audio backend (/api/audio/speak)
  try {
    const res = await authedFetch('/api/audio/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: clean,
        persona,
        provider: persona === 'gwen' ? 'elevenlabs' : 'edge',
        voice_id: persona === 'gwen' ? 'EXAVITQu4vr4xnSDxMaL' : (isAr ? 'ar-EG-ShakirNeural' : 'en-US-GuyNeural'),
        model_id: persona === 'gwen' ? 'eleven_multilingual_v2' : undefined,
      }),
    });

    if (res.ok) {
      const data = await res.json().catch(() => null);
      if (data?.data_url) {
        const audio = new Audio(data.data_url);
        activeAudio = audio;
        await new Promise<void>((resolve) => {
          const done = () => {
            if (activeAudio === audio) activeAudio = null;
            resolve();
          };
          audio.onended = done;
          audio.onerror = done;
          audio.play().catch(done);
          setTimeout(done, 30000);
        });
        return;
      }
    }
  } catch (err) {
    console.warn('Backend audio synthesis unavailable:', err);
  }

  // Tier 3: Browser fallback if backend was unavailable
  if ('speechSynthesis' in window) {
    try {
      const voices = await getVoicesSafely();
      const voice = isAr
        ? pickArabicVoice(voices, persona)
        : pickEnglishVoice(voices, persona);
      await new Promise<void>((resolve) => {
        const utter = new SpeechSynthesisUtterance(clean);
        utter.lang = isAr ? 'ar-EG' : 'en-US';
        utter.rate = 1.05;
        utter.pitch = 0.95;
        if (voice) utter.voice = voice;
        utter.onend = () => resolve();
        utter.onerror = () => resolve();
        window.speechSynthesis.speak(utter);
        setTimeout(resolve, 20000);
      });
    } catch {}
  }
};
