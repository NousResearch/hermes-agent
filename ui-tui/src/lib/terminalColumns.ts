const DEFAULT_COLUMNS = 80

// Some terminals settle `stdout.columns` to the final width slightly AFTER the
// last `resize` event fires (observed with Ghostty). A leading-edge-only read
// then latches the stale pre-settle width and is never corrected, so
// fixed-width chrome — notably the status bar — stays laid out wider than the
// real terminal and wraps/fragments across lines (#36666).
//
// Read on the leading edge so width tracking stays responsive, AND re-read on a
// trailing debounce so the settled width always wins. This mirrors the
// `terminal.resize` debounce already used in useMainApp.
export const RESIZE_SETTLE_MS = 100

export interface ColumnsStream {
  columns?: number
  off: (event: 'resize', listener: () => void) => void
  on: (event: 'resize', listener: () => void) => void
}

export function subscribeColumns(
  stdout: ColumnsStream,
  onColumns: (columns: number) => void,
  settleMs: number = RESIZE_SETTLE_MS
): () => void {
  const read = () => onColumns(stdout.columns ?? DEFAULT_COLUMNS)

  let settleTimer: ReturnType<typeof setTimeout> | undefined

  const onResize = () => {
    read()
    clearTimeout(settleTimer)
    settleTimer = setTimeout(read, settleMs)
  }

  stdout.on('resize', onResize)

  return () => {
    clearTimeout(settleTimer)
    stdout.off('resize', onResize)
  }
}
