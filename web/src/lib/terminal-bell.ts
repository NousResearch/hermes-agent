/**
 * Audible bell for the dashboard chat terminal.
 *
 * The Chat tab is an xterm mirror of a TUI running inside the gateway, fed by
 * a PTY that carries the TUI's raw output. When a blocking prompt opens — an
 * approval, a clarify question, a sudo/secret prompt — the TUI writes C0 BEL
 * (`\x07`) into that stream (gated server-side by `display.bell_on_prompt`).
 * xterm parses the byte and fires its `onBell` event, but a browser makes no
 * sound for BEL on its own, and nothing in the dashboard was listening, so
 * "waiting on you" moments in the dashboard arrived silently unless the user
 * happened to be looking at the tab.
 *
 * `ringBell` turns that event into a short two-tone chime through WebAudio:
 * no audio asset to ship, no browser-notification permission to grant, and it
 * works in a terminal that has no real bell. Every failure path is swallowed —
 * a bell must never throw into the VT parser or the chat's hot path.
 */

/** Peak gain. Tuned up from 0.09 on 2026-10-16: audible but only faintly. */
const PEAK_GAIN = 0.3;

/** ADSR-lite: near-instant attack, brief hold, ~260ms decay. */
const ATTACK_S = 0.01;
const HOLD_S = 0.05;
const DECAY_S = 0.26;

/**
 * Two ascending blips read as a "ding" rather than a click (E6 → A6), and
 * carry over fan noise better than a single low tone.
 */
const TONES: ReadonlyArray<readonly [offsetSeconds: number, frequency: number]> = [
  [0, 1318.51],
  [0.13, 1760],
];

type AudioContextCtor = new () => AudioContext;

let audioContext: AudioContext | null = null;

function audioContextCtor(): AudioContextCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

/** Lazily create the shared context; null when the browser has no WebAudio. */
function ensureContext(): AudioContext | null {
  if (audioContext) return audioContext;
  const Ctor = audioContextCtor();
  if (!Ctor) return null;
  try {
    audioContext = new Ctor();
  } catch {
    return null; // denied by policy (iframe/kiosk) — stay silent
  }
  return audioContext;
}

/**
 * Ring the dashboard bell once. Safe to call from xterm's `onBell` handler:
 * never throws, and no-ops where WebAudio is unavailable (jsdom, a browser
 * without it, a constructor blocked by policy).
 */
export function ringBell(): void {
  const audio = ensureContext();
  if (!audio) return;

  // A fresh context starts suspended until the page sees a user gesture. A
  // user who is typing in the chat has already given one, so this mainly
  // covers a tab that has never interacted yet; retrying here means the bell
  // starts working the moment that changes, instead of staying mute for the
  // life of the page.
  if (audio.state === "suspended") {
    void audio.resume().catch(() => {});
  }

  const start = audio.currentTime;
  for (const [offset, frequency] of TONES) {
    try {
      const oscillator = audio.createOscillator();
      const gain = audio.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0, start + offset);
      gain.gain.linearRampToValueAtTime(PEAK_GAIN, start + offset + ATTACK_S);
      gain.gain.setValueAtTime(PEAK_GAIN, start + offset + ATTACK_S + HOLD_S);
      gain.gain.exponentialRampToValueAtTime(
        0.0001,
        start + offset + ATTACK_S + HOLD_S + DECAY_S,
      );
      oscillator.connect(gain).connect(audio.destination);
      oscillator.start(start + offset);
      oscillator.stop(start + offset + ATTACK_S + HOLD_S + DECAY_S + 0.02);
    } catch {
      return; // a bell is never worth an exception in the terminal's hot path
    }
  }
}

/** Drop the cached context so a test can install its own. */
export function resetBellContextForTests(): void {
  audioContext = null;
}
