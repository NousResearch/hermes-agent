import { isProgressReportingAvailable, wrapForMultiplexer } from '@hermes/ink'

// OSC 9;4 tab/title-bar progress while a turn runs — same bytes and gate as the classic CLI
// (hermes_cli/terminal_progress.py). Ghostty drops the indicator after ~15 s without an update,
// so the pulse is re-sent until the turn settles. Ink writes the clear itself on exit.
const SEQ = {
  clear: '\x1b]9;4;0;\x07',
  error: '\x1b]9;4;2;100\x07',
  indeterminate: '\x1b]9;4;3;\x07'
} as const

const KEEPALIVE_MS = 5_000

let keepalive: null | ReturnType<typeof setInterval> = null

const emit = (state: keyof typeof SEQ) => process.stdout.write(wrapForMultiplexer(SEQ[state]))

export const terminalProgress = {
  /** Pulse until `settle`; no-op when disabled, unsupported, or already pulsing. */
  start(enabled: boolean) {
    if (keepalive || !enabled || !isProgressReportingAvailable()) {
      return
    }

    emit('indeterminate')
    keepalive = setInterval(() => emit('indeterminate'), KEEPALIVE_MS)
    keepalive.unref?.()
  },

  /** End a pulsing turn: `clear` (done / interrupted) or `error` (red, faded by the terminal). */
  settle(state: 'clear' | 'error') {
    if (!keepalive) {
      return
    }

    clearInterval(keepalive)
    keepalive = null
    emit(state)
  }
}
