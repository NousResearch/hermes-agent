/**
 * Live streaming throughput ("instantaneous" t/s) for the TUI status bar.
 *
 * The server-side `usage.avg_tps` is a rolling average over *completed* API
 * calls (see `tui_gateway/server.py`: sum(output tokens) / sum(latency) over
 * the last 10 calls, "CLI status-bar parity"), so it only moves when a call
 * finishes. The status bar reads better with a rate that moves while tokens
 * are still arriving, so this counter timestamps every streamed delta over a
 * short window instead.
 *
 * Mechanics, deliberately simple and mirroring the classic CLI bar's feel:
 *   - keep delta timestamps from the last {@link WINDOW_MS} (2 s);
 *   - emit at most once per {@link THROTTLE_MS} (1 s) so the bar cannot flicker;
 *   - need at least {@link MIN_SAMPLES} deltas before a rate is meaningful;
 *   - rate = samples / window seconds, rounded to one decimal.
 *
 * Pure and dependency-free: `accumulate(now)` takes the clock as an argument so
 * tests do not need fake timers.
 */

export const WINDOW_MS = 2000
const THROTTLE_MS = 1000
const MIN_SAMPLES = 3

export interface StreamTpsCounter {
  /** Record a streamed delta at `now` (ms); returns a rate or `undefined` when muted. */
  accumulate(now: number): number | undefined
  /** Drop all state; call on turn start and turn end. */
  reset(): void
}

export function createStreamTpsCounter(): StreamTpsCounter {
  let stamps: number[] = []
  let lastEmit = 0

  return {
    accumulate(now: number): number | undefined {
      stamps.push(now)

      while (stamps.length > 0 && stamps[0]! < now - WINDOW_MS) {
        stamps.shift()
      }

      if (now - lastEmit < THROTTLE_MS) {
        return undefined
      }

      lastEmit = now

      const samples = stamps.length

      if (samples < MIN_SAMPLES) {
        return undefined
      }

      return Math.round((samples / (WINDOW_MS / 1000)) * 10) / 10
    },

    reset(): void {
      stamps = []
      lastEmit = 0
    }
  }
}
