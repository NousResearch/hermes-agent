/**
 * Persistent false-stream suppression latch (#75637).
 *
 * Once the *first* empty-bracketed-paste speculative probe fires, the latch
 * suppresses all further speculative probes until a meaningful stream boundary
 * resets it.  Unlike a timer-based debounce, this latch does NOT re-open after
 * elapsed time — a continuing mouse-tracking fragment storm cannot cause a
 * recurring probe.
 */
export interface ClipProbeLatch {
  /**
   * Attempt a speculative clipboard probe.
   * Returns `true` when the probe may proceed (first call or post-reset),
   * `false` when it must be suppressed because a probe already fired for the
   * current stream.
   */
  tryProbe: () => boolean

  /** Reset the latch on a meaningful stream boundary (non-empty paste,
   *  explicit hotkey, or /paste).  Idempotent. */
  reset: () => void
}

export const createClipProbeLatch = (): ClipProbeLatch => {
  let latched = false

  return {
    tryProbe: () => {
      if (latched) {
        return false
      }

      latched = true

      return true
    },
    reset: () => {
      latched = false
    }
  }
}
