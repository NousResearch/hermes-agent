/**
 * Soft-keyboard / browser-chrome inset math for the dashboard (NS-434).
 *
 * Problem: when the on-screen keyboard opens on mobile, neither iOS Safari
 * nor Android Chrome (default `interactive-widget=resizes-visual`) shrinks
 * the *layout* viewport — the keyboard just overlays it. Our app shell is
 * `h-dvh`, so the terminal host's bounding box doesn't change, `fit()`
 * computes the same (cols, rows), and the PTY never re-lays-out. The Ink
 * input line — drawn at the bottom of the grid — ends up hidden underneath
 * the keyboard.
 *
 * The only reliable signal is `window.visualViewport`: its `height` shrinks
 * to the visible region above the keyboard, and `offsetTop` reflects any
 * visual-viewport scroll (iOS nudges the page when an input focuses). The
 * keyboard inset is the part of the layout viewport below the visual one:
 *
 *   inset = layoutHeight - vv.height - vv.offsetTop
 *
 * ChatPage applies this as bottom padding on the terminal wrapper, which
 * shrinks the xterm host → ResizeObserver refit → `term.onResize` sends
 * `RESIZE` to the PTY → Ink redraws the input line above the keyboard.
 * (A shell-wide `--hermes-mobile-bottom-inset` compensates the same region
 * on every mobile page; see `applyMobileViewportInset` in main.tsx and the
 * `@media (max-width: 768px)` block in index.css.)
 *
 * We also set `interactive-widget=resizes-content` in the viewport meta so
 * Android Chrome 108+ resizes the layout viewport natively; there the inset
 * computes ≈ 0 and this path is a harmless no-op. iOS and **Firefox
 * Mobile** ignore the directive (it is Chromium-only) and take the JS
 * path — FF measures exactly like iOS: layout and dvh stay put, only the
 * visual viewport shrinks.
 *
 * The second, newer problem (Firefox Mobile's bottom URL bar): the bar is
 * ~48–64px — below the keyboard threshold above — and it persists with the
 * keyboard CLOSED, so the prompt row sits under it even at rest. It is
 * compensated by `computeViewportChromeInset` (below) instead of by
 * lowering the keyboard threshold: a sub-80px inset is applied only while
 * its measured value stays STABLE across consecutive samples, which
 * matches a parked bottom bar and rejects the transient deltas that a
 * collapsing top bar (Chrome/Android) produces.
 */

/**
 * Insets smaller than this are treated as 0 for the *keyboard* path.
 * Collapsing browser chrome (URL bar show/hide) produces small transient
 * height deltas that would otherwise thrash the terminal grid; real soft
 * keyboards are ≥ ~150px.
 */
export const KEYBOARD_INSET_MIN_PX = 80;

/**
 * Chrome insets are only applied in the (MIN, MAX] band — i.e. the band
 * that is too small to be a keyboard but large enough to be a real bar.
 * 12px is below any plausible bar and above normal measurement jitter;
 * 79px stays strictly below the keyboard threshold so the two paths can
 * never both apply (double-compensation).
 */
export const CHROME_INSET_MIN_PX = 12;
export const CHROME_INSET_MAX_PX = 79;

/**
 * Consecutive equal samples required before a chrome inset is applied.
 * ~500ms at the shell's 500ms sampling cadence — long enough that a
 * collapsing/reshowing bar (whose value wobbles every sample) never
 * qualifies, short enough that a parked bar is compensated in half a
 * second. The window assumes that cadence: if the sampling interval
 * changes, revisit whether 2 samples still separates wobble from parked.
 */
export const CHROME_INSET_STABLE_SAMPLES = 2;

export interface ViewportGeometry {
  /** `visualViewport.height` — visible height above the keyboard. */
  height: number;
  /** `visualViewport.offsetTop` — visual viewport's offset into layout. */
  offsetTop: number;
}

/**
 * Height (px) of the layout viewport currently obscured by the soft
 * keyboard, or 0 when no keyboard is showing / the signal is unusable.
 */
export function computeKeyboardInset(
  viewport: ViewportGeometry | null | undefined,
  layoutHeightPx: number,
): number {
  if (!viewport || !Number.isFinite(layoutHeightPx) || layoutHeightPx <= 0) {
    return 0;
  }
  const { height, offsetTop } = viewport;
  if (!Number.isFinite(height) || !Number.isFinite(offsetTop)) return 0;
  const inset = Math.round(layoutHeightPx - height - offsetTop);
  return inset >= KEYBOARD_INSET_MIN_PX ? inset : 0;
}

/**
 * Height (px) of persistent browser chrome (e.g. Firefox Mobile's bottom
 * URL bar) that should be compensated, or 0.
 *
 * The raw obscured region is the same `layout - vv.height - vv.offsetTop`
 * the keyboard path uses, minus any keyboard inset (while a keyboard is up
 * the raw value includes the bar region, which the keyboard path already
 * compensates — clamping to the band below keeps the two disjoint), then
 * gated on *stability*: the value only counts once it has been seen
 * `stableSamples` times in a row. A parked bottom bar reports the same
 * ~56px every sample and qualifies; a collapsing top bar wobbles between
 * ~0 and ~64px as the user scrolls and never does.
 *
 * `stableSamples` is the number of consecutive equal samples seen so far
 * (maintained by the caller across its sampling loop).
 */
export function computeViewportChromeInset(
  viewport: ViewportGeometry | null | undefined,
  layoutHeightPx: number,
  keyboardInsetPx: number,
  stableSamples: number,
): number {
  if (!viewport || !Number.isFinite(layoutHeightPx) || layoutHeightPx <= 0) {
    return 0;
  }
  const { height, offsetTop } = viewport;
  if (!Number.isFinite(height) || !Number.isFinite(offsetTop)) return 0;
  const raw = Math.round(
    Math.min(
      Math.max(layoutHeightPx - height - offsetTop - keyboardInsetPx, 0),
      CHROME_INSET_MAX_PX,
    ),
  );
  // The Math.max(…, 0) is the lower guard — a negative raw (visual viewport
  // taller than the layout, seen in some iOS zoom states) yields 0 there and
  // never reaches the band check. The band check below is the effective
  // lower bound for real (non-negative) readings.
  if (raw < CHROME_INSET_MIN_PX) return 0;
  if (!Number.isFinite(stableSamples) || stableSamples < CHROME_INSET_STABLE_SAMPLES) {
    return 0;
  }
  return raw;
}

/**
 * Whether the page scroll should be pinned back to the top.
 *
 * The dashboard shell is a fixed `h-dvh` column and must never scroll, but
 * iOS Safari auto-scrolls the *page* when a focused input would sit under
 * the keyboard (xterm's hidden textarea triggers this). Pin whenever a
 * keyboard is present so the terminal chrome stays put; the terminal's own
 * scrollback handles content visibility.
 */
export function shouldPinScroll(nextInsetPx: number): boolean {
  return nextInsetPx > 0;
}
