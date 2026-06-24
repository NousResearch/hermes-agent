/**
 * Hangul input automaton for the embedded dashboard chat.
 *
 * WHY THIS EXISTS
 * ---------------
 * On iPad (all iPad browsers are WebKit) the virtual Korean keyboard does NOT
 * drive IME composition inside xterm.js's hidden textarea — xterm constantly
 * moves/clears that textarea, which resets WebKit's composition context. The
 * result is that each *committed* compatibility jamo (ㅎ U+314E, ㅏ U+314F,
 * ㄴ U+3134 …) arrives as its own `onData` event. Concatenating them yields
 * the three-letter string "ㅎㅏㄴ", which renders as three loose letters — not
 * the syllable "한" (U+D55C) a desktop OS IME would have composed before the
 * terminal ever saw it.
 *
 * This module reimplements the standard 2-beolsik input automaton: it folds a
 * stream of compatibility jamo into precomposed Hangul syllable blocks, exactly
 * like an OS IME.
 *
 * COMMIT-BASED OUTPUT (no backspaces)
 * -----------------------------------
 * An earlier version rewrote the on-screen "composing" glyph by emitting
 * backspaces (ㅎ → ⌫하 → ⌫한). That cannot work against the Hermes TUI: its
 * input tokenizer (hermes-ink termio) only breaks text runs on ESC, so a
 * backspace byte (0x7f) glued to the next syllable arrives as one token that
 * `parseKeypress` recognizes as neither backspace nor printable text — and is
 * dropped. So only the first jamo ever survived.
 *
 * Instead we do exactly what a desktop OS IME does: the terminal never sees an
 * in-progress syllable. `feed()` emits a syllable only once it is *committed*
 * (a following jamo starts a new syllable, or `flush()` is called on a space /
 * enter / any non-jamo). The in-progress syllable is held internally until then.
 * Trade-off: the syllable currently being typed isn't echoed until it commits —
 * identical to how a desktop terminal only receives finished syllables.
 */

// Compatibility-jamo code points, indexed the way the syllable formula expects.
//   syllable = 0xAC00 + (cho * 21 + jung) * 28 + jong
// `cho` 0-18, `jung` 0-20, `jong` 0-27 (0 = no final consonant).
const LEAD_CHARS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
const VOWEL_CHARS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
// Tail index 0 is "no tail"; the placeholder keeps the string 1-indexed.
const TAIL_CHARS = "_ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ";

const LEAD_IDX = new Map<string, number>(
  [...LEAD_CHARS].map((c, i): [string, number] => [c, i]),
);
const VOWEL_IDX = new Map<string, number>(
  [...VOWEL_CHARS].map((c, i): [string, number] => [c, i]),
);
const TAIL_IDX = new Map<string, number>(
  [...TAIL_CHARS]
    .map((c, i): [string, number] => [c, i])
    .filter(([c]) => c !== "_"),
);

// Compound vowels formed by typing two simple vowels in sequence.
//   key = `${jungIdx},${incomingVowelIdx}` -> resulting jung index
const VOWEL_COMPOSE = new Map<string, number>([
  ["8,0", 9], // ㅗ+ㅏ → ㅘ
  ["8,1", 10], // ㅗ+ㅐ → ㅙ
  ["8,20", 11], // ㅗ+ㅣ → ㅚ
  ["13,4", 14], // ㅜ+ㅓ → ㅝ
  ["13,5", 15], // ㅜ+ㅔ → ㅞ
  ["13,20", 16], // ㅜ+ㅣ → ㅟ
  ["18,20", 19], // ㅡ+ㅣ → ㅢ
]);

// Compound finals formed by typing two consonants while a final is pending.
//   key = `${jongIdx},${incomingLeadIdx}` -> resulting jong index
const TAIL_COMPOSE = new Map<string, number>([
  ["1,9", 3], // ㄱ+ㅅ → ㄳ
  ["4,12", 5], // ㄴ+ㅈ → ㄵ
  ["4,18", 6], // ㄴ+ㅎ → ㄶ
  ["8,0", 9], // ㄹ+ㄱ → ㄺ
  ["8,6", 10], // ㄹ+ㅁ → ㄻ
  ["8,7", 11], // ㄹ+ㅂ → ㄼ
  ["8,9", 12], // ㄹ+ㅅ → ㄽ
  ["8,16", 13], // ㄹ+ㅌ → ㄾ
  ["8,17", 14], // ㄹ+ㅍ → ㄿ
  ["8,18", 15], // ㄹ+ㅎ → ㅀ
  ["17,9", 18], // ㅂ+ㅅ → ㅄ
]);

// When a vowel follows a final consonant, that final (or the last half of a
// compound final) migrates to become the lead of the next syllable.
//   jong index -> { baseJong, leadIdx }
//   baseJong 0 means "the whole final migrated, nothing remains".
const TAIL_SPLIT = new Map<number, { baseJong: number; leadIdx: number }>([
  // simple finals: the entire final becomes the next lead
  [1, { baseJong: 0, leadIdx: 0 }], // ㄱ → ㄱ
  [2, { baseJong: 0, leadIdx: 1 }], // ㄲ → ㄲ
  [4, { baseJong: 0, leadIdx: 2 }], // ㄴ → ㄴ
  [7, { baseJong: 0, leadIdx: 3 }], // ㄷ → ㄷ
  [8, { baseJong: 0, leadIdx: 5 }], // ㄹ → ㄹ
  [16, { baseJong: 0, leadIdx: 6 }], // ㅁ → ㅁ
  [17, { baseJong: 0, leadIdx: 7 }], // ㅂ → ㅂ
  [19, { baseJong: 0, leadIdx: 9 }], // ㅅ → ㅅ
  [20, { baseJong: 0, leadIdx: 10 }], // ㅆ → ㅆ
  [21, { baseJong: 0, leadIdx: 11 }], // ㅇ → ㅇ
  [22, { baseJong: 0, leadIdx: 12 }], // ㅈ → ㅈ
  [23, { baseJong: 0, leadIdx: 14 }], // ㅊ → ㅊ
  [24, { baseJong: 0, leadIdx: 15 }], // ㅋ → ㅋ
  [25, { baseJong: 0, leadIdx: 16 }], // ㅌ → ㅌ
  [26, { baseJong: 0, leadIdx: 17 }], // ㅍ → ㅍ
  [27, { baseJong: 0, leadIdx: 18 }], // ㅎ → ㅎ
  // compound finals: keep the first half, migrate the second
  [3, { baseJong: 1, leadIdx: 9 }], // ㄳ → ㄱ + ㅅ
  [5, { baseJong: 4, leadIdx: 12 }], // ㄵ → ㄴ + ㅈ
  [6, { baseJong: 4, leadIdx: 18 }], // ㄶ → ㄴ + ㅎ
  [9, { baseJong: 8, leadIdx: 0 }], // ㄺ → ㄹ + ㄱ
  [10, { baseJong: 8, leadIdx: 6 }], // ㄻ → ㄹ + ㅁ
  [11, { baseJong: 8, leadIdx: 7 }], // ㄼ → ㄹ + ㅂ
  [12, { baseJong: 8, leadIdx: 9 }], // ㄽ → ㄹ + ㅅ
  [13, { baseJong: 8, leadIdx: 16 }], // ㄾ → ㄹ + ㅌ
  [14, { baseJong: 8, leadIdx: 17 }], // ㄿ → ㄹ + ㅍ
  [15, { baseJong: 8, leadIdx: 18 }], // ㅀ → ㄹ + ㅎ
  [18, { baseJong: 17, leadIdx: 9 }], // ㅄ → ㅂ + ㅅ
]);

/** True when `ch` is a single compatibility jamo (the only thing we compose). */
export function isCompatJamo(ch: string): boolean {
  if (ch.length !== 1) return false;
  return LEAD_IDX.has(ch) || VOWEL_IDX.has(ch) || TAIL_IDX.has(ch);
}

interface Cluster {
  cho: number; // -1 = empty
  jung: number;
  jong: number; // 0 or -1 = no final
}

function emptyCluster(): Cluster {
  return { cho: -1, jung: -1, jong: -1 };
}

function isEmpty(c: Cluster): boolean {
  return c.cho < 0 && c.jung < 0 && c.jong < 0;
}

/** Render a cluster to the single glyph it currently represents. */
function display(c: Cluster): string {
  if (c.cho >= 0 && c.jung >= 0) {
    const jong = c.jong > 0 ? c.jong : 0;
    return String.fromCharCode(0xac00 + (c.cho * 21 + c.jung) * 28 + jong);
  }
  if (c.cho >= 0) return LEAD_CHARS[c.cho]; // lone consonant
  if (c.jung >= 0) return VOWEL_CHARS[c.jung]; // lone vowel
  return "";
}

/**
 * Stateful Hangul composer. Feed it one compatibility jamo at a time; `feed()`
 * returns only text that has been *committed* (finalized syllables) and should
 * be sent to the terminal now. The syllable still under construction is held
 * internally until it commits or `flush()` is called — mirroring how a desktop
 * OS IME only hands the terminal finished syllables.
 */
export class HangulComposer {
  private cluster = emptyCluster();

  /**
   * Finalize and return the in-progress syllable (empty string if none), then
   * reset. Call when a non-jamo (space, enter, ASCII, paste) arrives so the
   * pending syllable is committed ahead of that input.
   */
  flush(): string {
    const out = display(this.cluster);
    this.cluster = emptyCluster();
    return out;
  }

  /** Drop the in-progress syllable without emitting it (e.g. user backspace). */
  discard(): void {
    this.cluster = emptyCluster();
  }

  /** Whether a syllable is currently being composed. */
  get composing(): boolean {
    return !isEmpty(this.cluster);
  }

  /** Feed one compatibility jamo; returns committed text to send now (may be ""). */
  feed(jamo: string): string {
    let committed = "";
    const c = this.cluster;

    // Finalize the current cluster and begin a fresh one.
    const startNew = (next: Cluster) => {
      committed += display(c);
      this.cluster = next;
    };

    if (VOWEL_IDX.has(jamo)) {
      const v = VOWEL_IDX.get(jamo)!;
      if (c.jung < 0) {
        if (c.jong >= 0 && c.jong > 0) {
          // LVT + V: the final migrates to a new syllable's lead.
          const { baseJong, leadIdx } = TAIL_SPLIT.get(c.jong)!;
          committed += display({ cho: c.cho, jung: c.jung, jong: baseJong });
          this.cluster = { cho: leadIdx, jung: v, jong: -1 };
        } else if (c.cho >= 0) {
          c.jung = v; // L + V
        } else {
          startNew({ cho: -1, jung: v, jong: -1 }); // bare vowel
        }
      } else {
        const compound =
          c.jong <= 0 ? VOWEL_COMPOSE.get(`${c.jung},${v}`) : undefined;
        if (compound !== undefined) {
          c.jung = compound; // ㅗ+ㅏ → ㅘ etc.
        } else if (c.jong > 0) {
          // LVT + V: migrate the final, same as above.
          const { baseJong, leadIdx } = TAIL_SPLIT.get(c.jong)!;
          committed += display({ cho: c.cho, jung: c.jung, jong: baseJong });
          this.cluster = { cho: leadIdx, jung: v, jong: -1 };
        } else {
          startNew({ cho: -1, jung: v, jong: -1 }); // LV + V, not combinable
        }
      }
    } else {
      // Consonant. A jamo may be valid as a lead, a final, or both.
      const lead = LEAD_IDX.has(jamo) ? LEAD_IDX.get(jamo)! : -1;
      const tail = TAIL_IDX.has(jamo) ? TAIL_IDX.get(jamo)! : -1;

      if (isEmpty(c)) {
        if (lead >= 0) this.cluster = { cho: lead, jung: -1, jong: -1 };
        else committed += jamo; // compound consonant typed alone: pass through
      } else if (c.jung < 0) {
        // Consonant with no vowel yet: can't stack, start a new lead.
        if (lead >= 0) startNew({ cho: lead, jung: -1, jong: -1 });
        else {
          startNew(emptyCluster());
          committed += jamo;
        }
      } else if (c.jong <= 0) {
        // LV + consonant → tentative final (if it can be one).
        if (tail >= 0) c.jong = tail;
        else if (lead >= 0) startNew({ cho: lead, jung: -1, jong: -1 });
        else {
          startNew(emptyCluster());
          committed += jamo;
        }
      } else {
        // LVT + consonant → try to form a compound final.
        const compound =
          lead >= 0 ? TAIL_COMPOSE.get(`${c.jong},${lead}`) : undefined;
        if (compound !== undefined) c.jong = compound;
        else if (lead >= 0) startNew({ cho: lead, jung: -1, jong: -1 });
        else {
          startNew(emptyCluster());
          committed += jamo;
        }
      }
    }

    return committed;
  }
}
