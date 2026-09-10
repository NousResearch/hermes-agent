/**
 * Mobile viewport inset tracking (shell-wide, Firefox-Mobile-safe).
 *
 * Firefox Mobile ignores `interactive-widget=resizes-content`
 * (Chromium-only) and — like iOS — leaves the layout viewport, dvh, and
 * fixed elements put while the soft keyboard and the bottom URL bar
 * overlay the content. The dashboard's mobile layout depends on the
 * visible region (`--app-h`), so this tracker measures the region below
 * the visual viewport and publishes it as the CSS custom property
 * `--hermes-mobile-bottom-inset` (px). `index.css` consumes that variable
 * inside its `@media (max-width: 768px)` block only, so desktop layouts
 * are byte-for-byte untouched.
 *
 * Two kinds of overlap are compensated (see `keyboard-inset.ts` for the
 * pure math):
 *
 *   - the soft keyboard (raw inset ≥ 80px): applied immediately — a
 *     real keyboard change is a large, unambiguous step;
 *   - a parked bottom bar (12–79px): applied only after the value has
 *     been stable across consecutive samples, which rejects the
 *     collapsing top bars of Chrome/Android (their deltas wobble with
 *     scroll and never settle) and the transient deltas of a bottom bar
 *     expanding/collapsing (Firefox's "hide bar while scrolling" mode).
 *
 * The tracker runs on a 500ms sample timer PLUS `visualViewport`
 * `resize`/`scroll` events. The timer is the backbone: on mobile
 * Firefox it is the only thing that fires when the keyboard or a bar
 * changes (the vv events don't), and it also provides the stability
 * sampling cadence.
 */
import {
  computeKeyboardInset,
  computeViewportChromeInset,
  CHROME_INSET_STABLE_SAMPLES,
  type ViewportGeometry,
} from "./keyboard-inset";

/** Sampling cadence for the inset tracker (stability window ≈ 500ms). */
export const MOBILE_INSET_SAMPLE_MS = 500;

/** CSS custom property the mobile layout consumes (px, e.g. "56px"). */
export const MOBILE_BOTTOM_INSET_VAR = "--hermes-mobile-bottom-inset";

interface Sample {
  raw: number;
  count: number;
  applied: number;
}

let activeStop: (() => void) | null = null;

/**
 * Measure the current obscured region. Returns the keyboard inset (raw ≥
 * 80px) and the raw chrome candidate (the rest, clamped to [0, 79]), or
 * null when the signal is unusable.
 */
export function sampleViewportInset(): {
  keyboardInset: number;
  rawChrome: number;
} | null {
  const vv = window.visualViewport;
  if (!vv || !Number.isFinite(vv.height) || !Number.isFinite(vv.offsetTop)) {
    return null;
  }
  const layoutHeight = window.innerHeight;
  const geometry: ViewportGeometry = { height: vv.height, offsetTop: vv.offsetTop };
  const keyboardInset = computeKeyboardInset(geometry, layoutHeight);
  const chrome = computeViewportChromeInset(geometry, layoutHeight, keyboardInset, Number.MAX_SAFE_INTEGER);
  // `computeViewportChromeInset` with infinite samples returns the raw
  // band-clamped value — exactly what the stability gate needs.
  return { keyboardInset, rawChrome: chrome };
}

/**
 * Start tracking. Idempotent (returns the existing stop function when a
 * tracker is already running). Returns a stop function that detaches the
 * timer/listeners and clears the variable.
 */
export function trackMobileViewportInset(): () => void {
  if (activeStop) return activeStop;
  if (typeof document === "undefined") return () => undefined;
  const root = document.documentElement;
  // CSS default so the var always exists before the first sample.
  if (!getComputedStyle(root).getPropertyValue(MOBILE_BOTTOM_INSET_VAR)) {
    root.style.setProperty(MOBILE_BOTTOM_INSET_VAR, "0px");
  }

  const sample: Sample = { raw: 0, count: 0, applied: 0 };

  const apply = (insetPx: number) => {
    if (insetPx === sample.applied) return;
    sample.applied = insetPx;
    root.style.setProperty(MOBILE_BOTTOM_INSET_VAR, `${insetPx}px`);
  };

  const tick = () => {
    const reading = sampleViewportInset();
    if (!reading) return; // signal unusable — keep last value
    const { keyboardInset, rawChrome } = reading;

    // Stability gate for the chrome candidate: only count consecutive
    // equal samples; anything else resets the run.
    if (rawChrome === sample.raw) {
      sample.count += 1;
    } else {
      sample.raw = rawChrome;
      sample.count = 1;
    }
    // The chrome value only counts once its stability run completes.
    const gatedChrome =
      sample.count >= CHROME_INSET_STABLE_SAMPLES ? rawChrome : 0;
    const inset = Math.max(keyboardInset, gatedChrome);
    if (keyboardInset > 0) {
      // A real keyboard dwarfs any bar and is unambiguous — apply it
      // immediately regardless of the stability run, and reset the run
      // so the bar value seen under the keyboard doesn't linger.
      sample.raw = 0;
      sample.count = 0;
    }
    apply(inset);
  };

  // Prime the stability run now (a parked bar is usually already there on
  // load): this first sample starts the run, the 500ms timer's next
  // sample confirms it — a parked bar is compensated after ~500ms.
  tick();
  const timer = window.setInterval(tick, MOBILE_INSET_SAMPLE_MS);
  const vv = window.visualViewport;
  if (vv) {
    vv.addEventListener("resize", tick);
    vv.addEventListener("scroll", tick);
  }
  window.addEventListener("resize", tick);

  let stopped = false;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    activeStop = null;
    window.clearInterval(timer);
    vv?.removeEventListener("resize", tick);
    vv?.removeEventListener("scroll", tick);
    window.removeEventListener("resize", tick);
    sample.applied = 0;
    root.style.setProperty(MOBILE_BOTTOM_INSET_VAR, "0px");
  };
  activeStop = stop;
  return stop;
}
