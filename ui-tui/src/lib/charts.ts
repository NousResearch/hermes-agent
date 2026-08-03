/**
 * Chart primitives — pure string builders (no React), THE charting layer for
 * widget apps and demos alike. Everything returns plain strings the caller
 * colors with theme tones; everything auto-scales to the series' min/max.
 */

const BLOCKS = '▁▂▃▄▅▆▇█'

const normalize = (series: number[], window: number): { min: number; range: number; window: number[] } => {
  const view = series.slice(-Math.max(1, window))
  const min = Math.min(...view)

  return { min, range: Math.max(...view) - min || 1, window: view }
}

/** One-row sparkline: `▂▃▅▇█▆…`, last `width` samples. ALWAYS exactly
 *  `width` cells — short series pad-left so the card never resizes while
 *  history warms up (latest sample stays pinned to the right edge). */
export function sparkline(series: number[], width = series.length): string {
  if (!series.length) {
    return ' '.repeat(Math.max(0, width))
  }

  const { min, range, window } = normalize(series, width)

  return window
    .map(v => BLOCKS[Math.min(BLOCKS.length - 1, Math.floor(((v - min) / range) * BLOCKS.length))])
    .join('')
    .padStart(width)
}

/**
 * Multi-row column chart, top line first — the streams-demo panel chart.
 * Each column resolves to `rows * 8` vertical levels (full blocks below the
 * value, a partial eighth-block at it), so taller cells genuinely gain
 * resolution.
 */
export function sparkRows(series: number[], width: number, rows: number): string[] {
  if (!series.length) {
    return Array.from({ length: rows }, () => ' '.repeat(width))
  }

  const { min, range, window } = normalize(series, width)
  const levels = window.map(v => Math.max(1, Math.round(((v - min) / range) * rows * 8)))

  return Array.from({ length: rows }, (_, lineIdx) => {
    const rowFromBottom = rows - 1 - lineIdx

    return levels
      .map(level => {
        const filled = Math.min(8, Math.max(0, level - rowFromBottom * 8))

        return filled === 0 ? ' ' : BLOCKS[filled - 1]
      })
      .join('')
      .padStart(width) // warm-up pads left: chart grows from the right, card never resizes
  })
}

/** Horizontal fill gauge: `█████░░░` for a 0..1 ratio. */
export function gauge(ratio: number, width: number): string {
  const filled = Math.round(Math.min(1, Math.max(0, ratio)) * width)

  return '█'.repeat(filled) + '░'.repeat(Math.max(0, width - filled))
}

/** Horizontal bar chart: one `███▌`-style bar per value, scaled to the max,
 *  each padded to exactly `width` (stable card sizing). Eighth-block tips
 *  keep adjacent values distinguishable. */
export function hbars(values: number[], width: number): string[] {
  const max = Math.max(...values, 0) || 1

  return values.map(v => {
    const cells = (Math.min(max, Math.max(0, v)) / max) * width
    const full = Math.floor(cells)
    const rest = Math.round((cells - full) * 8)

    return ('█'.repeat(full) + (rest > 0 ? '▏▎▍▌▋▊▉█'[rest - 1] : '')).padEnd(width)
  })
}
