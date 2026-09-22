/**
 * Procedural Iron Man & JARVIS Audio FX Synthesizer using Web Audio API.
 * Synthesizes Stark Industries HUD soundscapes without external audio files.
 */

let sharedCtx: AudioContext | null = null;

export function getSharedAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
  if (!AudioCtxClass) return null;

  if (!sharedCtx || sharedCtx.state === 'closed') {
    sharedCtx = new AudioCtxClass();
  }
  if (sharedCtx.state === 'suspended') {
    sharedCtx.resume().catch(() => {});
  }
  return sharedCtx;
}

/**
 * Tony Stark "Knock" Sound:
 * Realistic double-tap metallic/glass acoustic resonance simulating tapping the arc reactor.
 */
export function playKnockSound(): void {
  const ctx = getSharedAudioContext();
  if (!ctx || typeof ctx.createOscillator !== 'function' || typeof ctx.createGain !== 'function') return;

  const now = ctx.currentTime;

  const triggerTap = (time: number, freq: number, gainVal: number) => {
    // Fundamental strike
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, time);
    osc.frequency.exponentialRampToValueAtTime(freq * 0.4, time + 0.08);

    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(freq * 1.5, time);
    filter.Q.setValueAtTime(4.0, time);

    gain.gain.setValueAtTime(0.001, time);
    gain.gain.linearRampToValueAtTime(gainVal, time + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.09);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(time);
    osc.stop(time + 0.1);
  };

  // Tap 1
  triggerTap(now, 580, 0.45);
  // Tap 2 (slightly higher harmonic knock 110ms later)
  triggerTap(now + 0.11, 720, 0.55);
}

/**
 * Iron Man Arc Reactor Plasma Power-Up Sound:
 * Deep plasma hum frequency-sweeping upward into crystalline harmonic resonance.
 */
export function playArcReactorBootSound(): void {
  const ctx = getSharedAudioContext();
  if (!ctx || typeof ctx.createOscillator !== 'function' || typeof ctx.createGain !== 'function') return;

  const now = ctx.currentTime;
  const duration = 0.95;

  // Primary sub/plasma oscillator
  const osc1 = ctx.createOscillator();
  const gain1 = ctx.createGain();

  osc1.type = 'sawtooth';
  osc1.frequency.setValueAtTime(75, now);
  osc1.frequency.exponentialRampToValueAtTime(580, now + duration * 0.85);

  const filter1 = ctx.createBiquadFilter();
  filter1.type = 'lowpass';
  filter1.frequency.setValueAtTime(160, now);
  filter1.frequency.exponentialRampToValueAtTime(2400, now + duration * 0.8);
  filter1.Q.setValueAtTime(6.0, now);

  gain1.gain.setValueAtTime(0.001, now);
  gain1.gain.linearRampToValueAtTime(0.35, now + 0.12);
  gain1.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc1.connect(filter1);
  filter1.connect(gain1);
  gain1.connect(ctx.destination);

  osc1.start(now);
  osc1.stop(now + duration);

  // Secondary high crystal harmonic shimmer
  const osc2 = ctx.createOscillator();
  const gain2 = ctx.createGain();

  osc2.type = 'sine';
  osc2.frequency.setValueAtTime(880, now + 0.15);
  osc2.frequency.exponentialRampToValueAtTime(1760, now + duration);

  gain2.gain.setValueAtTime(0.001, now + 0.15);
  gain2.gain.linearRampToValueAtTime(0.18, now + 0.35);
  gain2.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc2.connect(gain2);
  gain2.connect(ctx.destination);

  osc2.start(now + 0.15);
  osc2.stop(now + duration);
}

/**
 * Wake Up Affirmative Cyber-Chime:
 * Ascending two-tone futuristic affirmation (C5 -> C6).
 */
export function playWakeChime(): void {
  const ctx = getSharedAudioContext();
  if (!ctx || typeof ctx.createOscillator !== 'function' || typeof ctx.createGain !== 'function') return;

  const now = ctx.currentTime;

  const tone = (time: number, freq: number, dur: number, vol: number) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, time);

    gain.gain.setValueAtTime(0.001, time);
    gain.gain.linearRampToValueAtTime(vol, time + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + dur);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(time);
    osc.stop(time + dur);
  };

  tone(now, 523.25, 0.22, 0.28); // C5
  tone(now + 0.12, 1046.5, 0.38, 0.35); // C6
}

/**
 * Standby Descent Chime:
 * Gentle descending chime indicating quiet ambient monitoring (E5 -> E4).
 */
export function playStandbyChime(): void {
  const ctx = getSharedAudioContext();
  if (!ctx || typeof ctx.createOscillator !== 'function' || typeof ctx.createGain !== 'function') return;

  const now = ctx.currentTime;

  const tone = (time: number, freq: number, dur: number, vol: number) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, time);

    gain.gain.setValueAtTime(0.001, time);
    gain.gain.linearRampToValueAtTime(vol, time + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + dur);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(time);
    osc.stop(time + dur);
  };

  tone(now, 659.25, 0.24, 0.22); // E5
  tone(now + 0.14, 329.63, 0.35, 0.18); // E4
}
