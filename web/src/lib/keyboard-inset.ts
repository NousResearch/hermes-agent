/**
 * Soft-keyboard inset math for the dashboard chat terminal (NS-434).
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
 *
 * We also set `interactive-widget=resizes-content` in the viewport meta so
 * Android Chrome 108+ resizes the layout viewport natively; there the inset
 * computes ≈ 0 and this path is a harmless no-op. iOS ignores the directive
 * and takes the JS path.
 */

/**
 * Insets smaller than this are treated as 0. Collapsing browser chrome
 * (URL bar show/hide) produces small transient height deltas that would
 * otherwise thrash the terminal grid; real soft keyboards are ≥ ~150px.
 */
export const KEYBOARD_INSET_MIN_PX = 80;

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
