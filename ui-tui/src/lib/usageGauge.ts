/** Split a `/usage` line so `[||||    ]` can be colored like the model-picker gauges. */

export type GaugeStyle = 'green' | 'red' | 'yellow'

// Exactly the 10-cell bar `render_bar` draws (same rule as the desktop and curses
// pickers) — so a markdown `[ ]` checkbox, a bare `[]` or a stray `[|x]` is not a gauge.
const GAUGE_BAR_RE = /\[(?=[| ]{10}\])\|* *\]/g

/** True when `text` holds at least one `[||||    ]` bar (a `/usage` row). */
export const hasUsageGauge = (text: string): boolean => new RegExp(GAUGE_BAR_RE.source).test(text)

export function gaugeBarStyle(bar: string): GaugeStyle {
  const inner = bar.slice(1, -1)
  // Fill is what is USED (matches the portal's "N% used"): red once 80%+ is gone.
  const used = inner.length ? (inner.match(/\|/g) ?? []).length / inner.length : 0

  if (used > 0.8) {
    return 'red'
  }

  if (used > 0.5) {
    return 'yellow'
  }

  return 'green'
}

export function splitUsageGaugeLine(text: string): Array<{ style: GaugeStyle | null; text: string }> {
  const segs: Array<{ style: GaugeStyle | null; text: string }> = []
  let last = 0

  for (const match of text.matchAll(GAUGE_BAR_RE)) {
    const start = match.index ?? 0
    const bar = match[0]
    const inner = bar.slice(1, -1)
    const pipes = (inner.match(/\|/g) ?? []).join('')
    const empty = inner.slice(pipes.length)
    const style = gaugeBarStyle(bar)

    if (start > last) {
      segs.push({ style: null, text: text.slice(last, start) })
    }

    // Color only the `|` fill. wrap-ansi treats a colored `[||||||||]` as CSI
    // and prints `?[32m` — brackets stay uncolored on purpose.
    segs.push({ style: null, text: '[' })

    if (pipes) {
      segs.push({ style, text: pipes })
    }

    if (empty) {
      segs.push({ style: null, text: empty })
    }

    segs.push({ style: null, text: ']' })
    last = start + bar.length
  }

  if (last < text.length) {
    segs.push({ style: null, text: text.slice(last) })
  }

  return segs.length ? segs : [{ style: null, text }]
}
