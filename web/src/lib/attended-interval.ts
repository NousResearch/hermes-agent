/**
 * `setInterval` for background refreshes that stops while nobody is looking.
 *
 * On a scale-to-zero cloud agent every dashboard request can boot the machine,
 * so a forgotten tab polling every few seconds keeps waking an agent that went
 * to sleep. Ticks are skipped while the tab is hidden, or visible with no input
 * for `ATTENDED_IDLE_MS`. When the user comes back (the tab shows again, or any
 * input after a skipped tick), `fn` runs once right away so the page catches up.
 */

/** No input for this long and a visible tab counts as unattended. */
export const ATTENDED_IDLE_MS = 2 * 60_000;

const ACTIVITY_EVENTS = [
  "pointerdown",
  "pointermove",
  "keydown",
  "wheel",
  "touchstart",
] as const;

/** Starts the interval; the returned function stops it, like `clearInterval`. */
export function setAttendedInterval(fn: () => void, ms: number): () => void {
  let lastInputAt = Date.now();
  let skipped = false;

  const attended = () =>
    document.visibilityState === "visible" &&
    Date.now() - lastInputAt < ATTENDED_IDLE_MS;

  const tick = () => {
    skipped = !attended();
    if (!skipped) fn();
  };

  // Showing the tab again counts as input: the user is looking at it.
  const onActivity = () => {
    if (document.visibilityState !== "visible") return;
    lastInputAt = Date.now();
    if (skipped) {
      skipped = false;
      fn();
    }
  };

  const id = window.setInterval(tick, ms);
  for (const type of ACTIVITY_EVENTS) {
    window.addEventListener(type, onActivity, { passive: true });
  }
  document.addEventListener("visibilitychange", onActivity);

  return () => {
    window.clearInterval(id);
    for (const type of ACTIVITY_EVENTS) {
      window.removeEventListener(type, onActivity);
    }
    document.removeEventListener("visibilitychange", onActivity);
  };
}
