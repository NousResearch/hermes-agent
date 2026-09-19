/**
 * Wake Word and Standby Phrase Matcher for J.A.R.V.I.S. Ambient Live Mode.
 * Supports English and Egyptian / Modern Standard Arabic wake words and trailing commands.
 */

export interface WakeWordMatchResult {
  isWake: boolean;
  isStandby: boolean;
  wakeWord?: string;
  command: string;
  raw: string;
}

const WAKE_PATTERNS_EN = [
  /^(?:hey|hi|hello|ok|okay)?\s*(?:wake\s*up\s*jarvis|wake\s*up|hey\s*jarvis|hello\s*jarvis|hi\s*jarvis|ok\s*jarvis|jarvis\s*are\s*you\s*there|listen\s*jarvis|jarvis)[,،:\s-]*/i,
  /^(?:hey|hi|hello|ok|okay)?\s*(?:wake\s*up\s*gwen|hey\s*gwen|hello\s*gwen|hi\s*gwen|gwen)[,،:\s-]*/i,
];

const WAKE_PATTERNS_AR = [
  /^(?:يا\s*)?(?:اصحى|اصحي)\s*(?:يا\s*)?(?:جارفيس|جوين)[,،:\s-]*/i,
  /^(?:صباح\s*الخير|مساء\s*الخير|سامعني|الو|ألو|مرحبا|أهلاً|اهلا)\s*(?:يا\s*)?(?:جارفيس|جوين)[,،:\s-]*/i,
  /^(?:يا\s*)?(?:جارفيس|جوين)[,،:\s-]*/i,
  /^(?:اصحى|اصحي)[,،:\s-]*/i,
];

const STANDBY_PATTERNS = [
  /\b(?:standby\s*jarvis|go\s*to\s*sleep|sleep\s*mode|power\s*down|enter\s*standby|standby|goodnight\s*jarvis)\b/i,
  /(?:خليك\s*(?:على|في)\s*(?:ال\s*)?(?:standby|ستاند\s*باي|استعداد)|خلاص\s*يا\s*جارفيس|نام\s*يا\s*جارفيس|وضع\s*الاستعداد|ارتاح\s*يا\s*جارفيس)/i,
];

export function matchWakeWord(rawText: string): WakeWordMatchResult {
  if (!rawText) {
    return { isWake: false, isStandby: false, command: '', raw: '' };
  }

  const trimmed = rawText.trim();

  // Check Standby / Sleep intent first
  for (const pattern of STANDBY_PATTERNS) {
    if (pattern.test(trimmed)) {
      return {
        isWake: false,
        isStandby: true,
        command: '',
        raw: trimmed,
      };
    }
  }

  // Check English wake patterns
  for (const pattern of WAKE_PATTERNS_EN) {
    const match = trimmed.match(pattern);
    if (match && match[0]) {
      const wakeWord = match[0].trim();
      const command = trimmed.slice(match[0].length).trim();
      return {
        isWake: true,
        isStandby: false,
        wakeWord,
        command,
        raw: trimmed,
      };
    }
  }

  // Check Arabic wake patterns
  for (const pattern of WAKE_PATTERNS_AR) {
    const match = trimmed.match(pattern);
    if (match && match[0]) {
      const wakeWord = match[0].trim();
      const command = trimmed.slice(match[0].length).trim();
      return {
        isWake: true,
        isStandby: false,
        wakeWord,
        command,
        raw: trimmed,
      };
    }
  }

  return {
    isWake: false,
    isStandby: false,
    command: '',
    raw: trimmed,
  };
}
