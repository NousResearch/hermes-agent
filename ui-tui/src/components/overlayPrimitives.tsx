import type { Key } from '@hermes/ink'
import { Text, useInput } from '@hermes/ink'
import { type ReactNode, useState } from 'react'

import type { UsageModelData } from '../gatewayTypes.js'
import { liftForContrast, mix } from '../lib/color.js'
import type { Theme } from '../theme.js'

/**
 * Overlay width clamp: prefer `preferred`, honor the caller's `maxWidth`
 * ABSOLUTELY (a grid cell knows its budget — overflowing it clips at the
 * terminal edge), and keep a usability floor only when the cap allows it.
 */
export function clampOverlayWidth(preferred: number, maxWidth?: number, min = 24): number {
  const cap = maxWidth === undefined ? Number.MAX_SAFE_INTEGER : Math.max(1, Math.trunc(maxWidth))

  return Math.max(Math.min(min, cap), Math.min(preferred, cap))
}

/**
 * THE scrollbar treatment (transcript + overlays): thumb rides the theme
 * base, accent while interacting; track recedes via an explicit blend toward
 * the surface. Never SGR dim — terminal-interpreted, it renders as a black
 * slab on transparent profiles (terminal.background #00000000).
 */
export function scrollbarColors(t: Theme, hover: boolean, grabbed: boolean): { thumb: string; track: string } {
  return {
    thumb: grabbed || hover ? t.color.accent : t.color.primary,
    track: mix(hover ? t.color.border : t.color.muted, t.color.completionBg, hover ? 0.25 : 0.55)
  }
}

export interface MenuRowSpec {
  color?: string
  label: string
  run: () => void
}

/**
 * ↑/↓ + Enter + number-key selection over `rows`; Esc runs `onEscape`.
 * `onKey`, when given, runs first on every keypress — return `true` to mark
 * the key fully handled and skip the default escape/arrow/enter/number
 * handling for that keypress (e.g. a screen with a text-input sub-mode).
 */
export function useMenu(rows: MenuRowSpec[], onEscape: () => void, onKey?: (ch: string, key: Key) => boolean): number {
  const [sel, setSel] = useState(0)

  useInput((ch, key) => {
    if (onKey?.(ch, key)) {
      return
    }

    if (key.escape) {
      return onEscape()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < rows.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return rows[sel]?.run()
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= rows.length) {
      return rows[n - 1]?.run()
    }
  })

  return Math.min(sel, Math.max(0, rows.length - 1))
}

/**
 * THE selected-row treatment for every list surface (completions popover,
 * session switcher, pickers): a `selection` chip on the active row, nothing
 * painted otherwise. Panels never paint their own full background — floating
 * surfaces are `opaque` (terminal-native canvas) and only the active row
 * carries color, so list surfaces cannot disagree about "selected" and stay
 * correct on any terminal background. Callers own layout; this owns color.
 */
export function listRowStyle(t: Theme, active: boolean): { backgroundColor?: string; color?: string } {
  if (!active) {
    return {}
  }

  const backgroundColor = t.color.completionCurrentBg

  // The chip guarantees its own ink: a cross-polarity theme (dark palette on
  // a light terminal) pairs pale text with a light chip, so lift the ink
  // against the ACTUAL chip fill with the xterm algorithm.
  return { backgroundColor, color: liftForContrast(t.color.text, backgroundColor, 4.5) }
}

/** Spreadable props for a selectable row: chip bg + ink + bold when active.
 *  Spread AFTER `color` so the chip ink wins on the active row. Replaces
 *  `inverse`, which swaps against the terminal's unknowable default colors
 *  (a black slab on transparent profiles). */
export function chipRowProps(t: Theme, active: boolean): { backgroundColor?: string; bold: boolean; color?: string } {
  const row = listRowStyle(t, active)

  return { backgroundColor: row.backgroundColor, bold: active, ...(row.color ? { color: row.color } : {}) }
}

/** A numbered menu row with the ▸ cursor (mirrors ClarifyPrompt). Active rows
 *  carry the shared list-row selection chip — same treatment as completions
 *  and the session switcher — instead of `inverse`, whose contrast depends on
 *  the terminal's unknowable default colors. */
export function MenuRow({ active, index, label, t }: { active: boolean; index: number; label: string; t: Theme }) {
  const row = listRowStyle(t, active)

  return (
    <Text>
      <Text
        backgroundColor={row.backgroundColor}
        bold={active}
        color={active ? (row.color ?? t.color.label) : t.color.muted}
      >
        {active ? '▸ ' : '  '}
        {index}. {label}
      </Text>
    </Text>
  )
}

/** Plain (non-numbered) action row with the ▸ cursor (confirm screens). */
export function ActionRow({ active, label, color, t }: { active: boolean; label: string; color?: string; t: Theme }) {
  return (
    <Text>
      <Text color={active ? t.color.accent : t.color.muted}>{active ? '▸ ' : '  '}</Text>
      <Text bold={active} color={active ? (color ?? t.color.text) : t.color.muted}>
        {label}
      </Text>
    </Text>
  )
}

export const BAR_CELLS = 10

/** ratio in [0,1] -> { bar: '█…░…', pct: 0-100 } using `cells` cells. */
export function barCells(ratio: number, cells: number = BAR_CELLS): { bar: string; pct: number } {
  const r = Math.max(0, Math.min(1, ratio))

  const filled = Math.round(r * cells)

  return { bar: '█'.repeat(filled) + '░'.repeat(cells - filled), pct: Math.round(r * 100) }
}

/**
 * Two-bar dollar usage view (decided with the user over a crammed three-segment
 * bar: at terminal widths a single fill glyph per full-resolution bar is the
 * only legible option). The plan bar is labeled with the plan name and shows
 * the allowance detail + % used; the top-up bar shows purchased dollars (no
 * denominator, renders full = balance, rolls over). Dollars only — never
 * "credits". Each row:
 *   `Plus    [██████░░░░]  $14.00 of $20.00 · 30% used`
 * Renders nothing for a free account (no bars to draw — caller shows upsell).
 */
export function UsageBars({ model, t }: { model: undefined | UsageModelData; t: Theme }) {
  if (!model || !model.available) {
    return null
  }

  const rows: ReactNode[] = []
  // Label the plan bar with the plan name (padded for column alignment with the
  // top-up row). Falls back to 'plan' when the name is absent.
  const planLabel = (model.plan_name || 'plan').padEnd(8).slice(0, 8)

  if (model.plan_bar) {
    const b = model.plan_bar
    const { bar } = barCells(b.fill_fraction)
    const pct = b.pct_used == null ? '' : ` · ${b.pct_used}% used`

    rows.push(
      <Text color={t.color.text} key="plan">
        {planLabel}
        <Text color={t.color.muted}>[</Text>
        <Text color={t.color.accent}>{bar}</Text>
        <Text color={t.color.muted}>]</Text>
        {`  ${b.remaining_display} left of ${b.total_display}${pct}`}
      </Text>
    )
  }

  if (model.topup_bar) {
    const b = model.topup_bar
    const { bar } = barCells(1)

    rows.push(
      <Text color={t.color.text} key="topup">
        {'top-up  '}
        <Text color={t.color.muted}>[</Text>
        <Text color={t.color.ok}>{bar}</Text>
        <Text color={t.color.muted}>]</Text>
        {`  ${b.remaining_display} · never expires`}
      </Text>
    )
  }

  if (rows.length === 0) {
    return null
  }

  return <>{rows}</>
}

/**
 * Plain-text version of the two-bar usage view, for text-only surfaces (the
 * /usage transcript panel). Returns one string per line: a plan bar, a top-up
 * bar, and a total-spendable summary, whichever apply. Dollars only.
 */
export function usageBarsText(model: undefined | UsageModelData): string[] {
  if (!model || !model.available) {
    return []
  }

  const lines: string[] = []
  const planLabel = (model.plan_name || 'plan').padEnd(8).slice(0, 8)

  if (model.plan_bar) {
    const b = model.plan_bar
    const { bar } = barCells(b.fill_fraction)
    const pct = b.pct_used == null ? '' : ` · ${b.pct_used}% used`

    lines.push(`${planLabel}[${bar}]  ${b.remaining_display} left of ${b.total_display}${pct}`)
  }

  if (model.topup_bar) {
    const b = model.topup_bar
    const { bar } = barCells(1)

    lines.push(`top-up  [${bar}]  ${b.remaining_display} · never expires`)
  }

  if (model.total_spendable_display && model.has_topup) {
    lines.push(`Total spendable: ${model.total_spendable_display}`)
  }

  return lines
}

export const footer = (extra: string, t: Theme) => <Text color={t.color.muted}>{extra}</Text>
