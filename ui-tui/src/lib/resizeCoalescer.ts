export interface ResizeCoalescer {
  /** Call on each terminal 'resize' event. */
  schedule: () => void
  /** Drop any pending trailing reflow (effect cleanup). */
  cancel: () => void
}

/**
 * Leading + trailing throttle for terminal-resize bursts.
 *
 * A drag-resize emits a burst of 'resize' events; reflowing on every one
 * remounts the visible transcript rows each tick (they're keyed on cols so
 * yoga re-measures off live geometry), turning a smooth drag into a
 * flickering remount storm.
 *
 * `schedule()` reflows immediately on the first event (the drag stays
 * responsive), then collapses subsequent events to at most one `reflow()`
 * per `intervalMs`, and always fires a trailing `reflow()` so the final
 * width lands exactly. `cancel()` clears a pending trailing reflow.
 *
 * Uses `Date.now()` + `setTimeout` directly so it is deterministically
 * testable under fake timers.
 */
export function createResizeCoalescer(reflow: () => void, intervalMs: number): ResizeCoalescer {
  // -Infinity (not 0) so the first schedule() always satisfies the
  // elapsed >= interval leading-edge check regardless of the wall clock.
  let lastReflow = Number.NEGATIVE_INFINITY
  let trailing: ReturnType<typeof setTimeout> | undefined

  const run = () => {
    lastReflow = Date.now()
    reflow()
  }

  return {
    schedule() {
      const elapsed = Date.now() - lastReflow

      clearTimeout(trailing)
      trailing = undefined

      if (elapsed >= intervalMs) {
        run()
      } else {
        trailing = setTimeout(() => {
          trailing = undefined
          run()
        }, intervalMs - elapsed)
      }
    },
    cancel() {
      clearTimeout(trailing)
      trailing = undefined
    }
  }
}
