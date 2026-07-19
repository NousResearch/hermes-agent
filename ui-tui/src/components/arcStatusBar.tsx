import { Box, Text } from '@hermes/ink'
import type { ReactNode } from 'react'

import { fmtDuration } from '../domain/messages.js'
import type { RateCredits, RateWindow, Usage } from '../types.js'

// ── Arc Raiders palette (truecolor) ──────────────────────────────────────────
// Ported verbatim from the user's Claude Code statusline (~/.claude/usage-tracker)
// so the Hermes bar reads identically. Kept local (not theme-derived) because the
// whole point of `arc` mode is this specific neo-industrial look regardless of skin.
export const ARC = {
  teal: '#2C7A7B', // borders, structure
  cyan: '#4FD1C5', // data values, accent
  white: '#CBD5E0', // primary text
  dim: '#718096', // secondary / muted
  green: '#38A169', // OK / input flow
  amber: '#D69E2E', // warning / output flow
  red: '#E53E3E' // critical
} as const

// ── Pure helpers (unit-tested in arcStatusBar.test.ts) ────────────────────────

// Tokens → 'average book' pages. 300 words/page ÷ 0.75 words/token = 400 tok/page.
// A ~200k context ≈ 500 pages. Single tunable constant, mirrors usage.py.
export const TOKENS_PER_PAGE = 400

export function fmtPages(n: number): string {
  const p = n / TOKENS_PER_PAGE
  if (p < 0.1) return '0p'
  if (p < 1) return `${p.toFixed(1)}p`
  if (p >= 1000) return `${(p / 1000).toFixed(1)}Kp`
  return `${Math.round(p)}p`
}

export function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`
  return String(n)
}

// Format a reset timestamp (epoch seconds or ISO string) as compact "1H05M".
export function fmtUntil(resetsAt: number | string | undefined, now = Date.now()): string {
  if (resetsAt == null || resetsAt === 0) return ''
  let ts: number
  if (typeof resetsAt === 'number') {
    ts = resetsAt * 1000
  } else {
    const parsed = Date.parse(resetsAt)
    if (Number.isNaN(parsed)) return ''
    ts = parsed
  }
  const secs = (ts - now) / 1000
  if (secs <= 0) return '0M'
  const mins = Math.ceil(secs / 60)
  const h = Math.floor(mins / 60)
  const m = mins % 60
  return h ? `${h}H${String(m).padStart(2, '0')}M` : `${m}M`
}

export interface GaugeParts {
  filled: string
  empty: string
  pctText: string
  color: string
  warn: boolean
}

// Colored capacity gauge. Threshold colors match usage.py: <50 cyan, <80 amber,
// ≥80 red (+ radioactive warn glyph). Returns parts so the renderer can tint
// each segment independently through Ink <Text> nodes.
export function gauge(pct: number, w = 10): GaugeParts {
  const p = Math.max(0, Math.min(100, Math.round(pct)))
  const filledN = Math.max(0, Math.min(Math.round((p / 100) * w), w))
  let color: string = ARC.cyan
  let warn = false
  if (p >= 80) {
    color = ARC.red
    warn = true
  } else if (p >= 50) {
    color = ARC.amber
  }
  return {
    filled: '▓'.repeat(filledN),
    empty: '░'.repeat(w - filledN),
    pctText: `${p}`.padStart(3, ' ') + '%',
    color,
    warn
  }
}

// Cost badge: estimated costs get a leading `~`, actual/reconciled costs don't,
// `included` (subscription) reads "incl", and unknown pricing hides the segment.
export function fmtCost(usage: Pick<Usage, 'cost_status' | 'cost_usd'>): null | string {
  const status = usage.cost_status
  if (status === 'unknown' || status == null) {
    // Fall back to showing a number only if one was sent without a status.
    if (typeof usage.cost_usd !== 'number') return null
  }
  if (status === 'included') return 'incl'
  if (typeof usage.cost_usd !== 'number') return null
  const prefix = status === 'estimated' ? '~$' : '$'
  return `${prefix}${usage.cost_usd.toFixed(2)}`
}

// ── React component ──────────────────────────────────────────────────────────

function Gauge({ pct, w = 10 }: { pct: number; w?: number }) {
  const g = gauge(pct, w)
  return (
    <Text>
      <Text color={ARC.teal}>[</Text>
      <Text color={g.color}>{g.filled}</Text>
      <Text color={ARC.teal}>{g.empty}</Text>
      <Text color={ARC.teal}>]</Text>
      <Text color={g.color}> {g.pctText}</Text>
      {g.warn ? <Text color={ARC.red}> ☢</Text> : null}
    </Text>
  )
}

function chev(text: ReactNode, label: string) {
  return (
    <Text wrap="truncate-end">
      <Text color={ARC.teal}>«« </Text>
      <Text color={ARC.white}>{label}</Text>
      <Text color={ARC.teal}> :: </Text>
      {text}
      <Text color={ARC.teal}> »»</Text>
    </Text>
  )
}

function WindowSeg({ label, w }: { label: string; w: RateWindow }) {
  const reset = fmtUntil(w.resets_at)
  return (
    <Text>
      <Text color={ARC.dim}>{label} </Text>
      <Gauge pct={w.used_percentage} />
      {reset ? (
        <Text>
          {' '}
          <Text color={ARC.dim}>↻</Text>
          <Text color={ARC.cyan}>{reset}</Text>
        </Text>
      ) : null}
    </Text>
  )
}

function CreditsSeg({ c }: { c: RateCredits }) {
  if (typeof c.used_percentage === 'number') {
    const bal =
      typeof c.remaining_usd === 'number'
        ? ` $${c.remaining_usd.toFixed(2)}${typeof c.limit_usd === 'number' ? `/$${c.limit_usd.toFixed(0)}` : ''}`
        : ''
    return (
      <Text>
        <Text color={ARC.dim}>{c.label ?? 'CREDIT'} </Text>
        <Gauge pct={c.used_percentage} />
        {bal ? <Text color={ARC.cyan}>{bal}</Text> : null}
      </Text>
    )
  }
  if (typeof c.remaining_usd === 'number') {
    return (
      <Text>
        <Text color={ARC.dim}>{c.label ?? 'CREDIT'} </Text>
        <Text color={ARC.cyan}>${c.remaining_usd.toFixed(2)} left</Text>
      </Text>
    )
  }
  return <Text color={ARC.dim}>NO RATE DATA</Text>
}

export interface ArcStatusBarProps {
  model: string
  usage: Usage
  sessionStartedAt?: null | number
  now?: number
}

// Three-line neo-industrial status bar:
//   IDENT    : model // cost // duration
//   CAPACITY : 5H/7D windows or credit balance (whatever the provider reports)
//   FLUX     : ▲input/~pages ▼output/~pages :: context gauge
export function ArcStatusBar({ model, usage, sessionStartedAt, now = Date.now() }: ArcStatusBarProps) {
  const modelLabel = model.replace(/^claude[- ]/i, '').replace(/\s+/g, '.').toUpperCase()
  const cost = fmtCost(usage)
  const dur = sessionStartedAt ? fmtDuration(now - sessionStartedAt) : ''
  const rl = usage.rate_limits

  // ── LINE 1: IDENT ──
  const line1 = (
    <Text wrap="truncate-end">
      <Text color={ARC.teal}>«« </Text>
      <Text bold color={ARC.cyan}>{modelLabel}</Text>
      {cost ? (
        <Text>
          <Text color={ARC.teal}> // </Text>
          <Text color={ARC.cyan}>{cost}</Text>
        </Text>
      ) : null}
      {dur ? (
        <Text>
          <Text color={ARC.teal}> // </Text>
          <Text color={ARC.white}>{dur} UPLINK</Text>
        </Text>
      ) : null}
      <Text color={ARC.teal}> »»</Text>
    </Text>
  )

  // ── LINE 2: CAPACITY ──
  let capBody: React.ReactNode
  if (rl?.five_hour || rl?.seven_day) {
    capBody = (
      <Text>
        {rl.five_hour ? <WindowSeg label="5H" w={rl.five_hour} /> : null}
        {rl.five_hour && rl.seven_day ? <Text color={ARC.teal}> :: </Text> : null}
        {rl.seven_day ? <WindowSeg label="7D" w={rl.seven_day} /> : null}
      </Text>
    )
  } else if (rl?.credits) {
    capBody = <CreditsSeg c={rl.credits} />
  } else {
    capBody = <Text color={ARC.dim}>NO RATE DATA</Text>
  }
  const line2 = chev(capBody, 'RATE')

  // ── LINE 3: FLUX ──
  const inTok = usage.input ?? 0
  const outTok = usage.output ?? 0
  const flux = (
    <Text>
      {inTok || outTok ? (
        <Text>
          <Text color={ARC.green}>▲</Text>
          <Text color={ARC.cyan}>{fmtTok(inTok)}</Text>
          <Text color={ARC.dim}>/~{fmtPages(inTok)}</Text>
          <Text> </Text>
          <Text color={ARC.amber}>▼</Text>
          <Text color={ARC.cyan}>{fmtTok(outTok)}</Text>
          <Text color={ARC.dim}>/~{fmtPages(outTok)}</Text>
          <Text color={ARC.teal}> :: </Text>
        </Text>
      ) : null}
      <Text color={ARC.dim}>CONTEXT </Text>
      <Gauge pct={usage.context_percent ?? 0} />
    </Text>
  )
  const line3 = chev(flux, 'FLUX')

  return (
    <Box flexDirection="column">
      {line1}
      {line2}
      {line3}
    </Box>
  )
}
