import { Box, type ScrollBoxHandle, stringWidth, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import { $delegationState } from '../app/delegationStore.js'
import type { IndicatorStyle, Notice } from '../app/interfaces.js'
import { useTurnSelector } from '../app/turnStore.js'
import { DEV_CREDITS_MODE } from '../config/env.js'
import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { buildSubagentTree, treeTotals, widthByDepth } from '../lib/subagentTree.js'
import { fmtK } from '../lib/text.js'
import { useScrollbarSnapshot, useViewportSnapshot } from '../lib/viewportStore.js'
import type { Theme } from '../theme.js'
import type { Msg, Usage } from '../types.js'

const FACE_TICK_MS = 2500
const HEART_COLORS = ['#ff5fa2', '#ff4d6d']

// Keep verb segment width stable so status-bar content to the right doesn't
// jitter when the ticker rotates between short/long verbs.
export const VERB_PAD_LEN = VERBS.reduce((max, v) => Math.max(max, v.length), 0) + 1 // + ellipsis
export const padVerb = (verb: string) => `${verb}…`.padEnd(VERB_PAD_LEN, ' ')

// Compact alternates for the `emoji` and `ascii` indicator styles.
// Each entry is a fixed-width (display-width) glyph.
const EMOJI_FRAMES = ['⚕ ', '🌀', '🤔', '✨', '🍵', '🔮']
const ASCII_FRAMES = ['|', '/', '-', '\\']

// Faster tick for spinner-style indicators — they read as motion only
// at frame rates closer to their authored interval.
const SPINNER_TICK_MS = 100

interface IndicatorRender {
  frame: string
  intervalMs: number
  // When false, FaceTicker hides the rotating verb and just shows the
  // glyph + duration.  Lets `unicode` stay minimal while the other
  // styles keep the verb-rotation flavour users associate with the
  // running… status.
  showVerb: boolean
}

const renderIndicator = (style: IndicatorStyle, tick: number): IndicatorRender => {
  if (style === 'kaomoji') {
    return { frame: FACES[tick % FACES.length] ?? '', intervalMs: FACE_TICK_MS, showVerb: true }
  }

  if (style === 'emoji') {
    return {
      frame: EMOJI_FRAMES[tick % EMOJI_FRAMES.length] ?? '⚕ ',
      intervalMs: SPINNER_TICK_MS * 6,
      showVerb: true
    }
  }

  if (style === 'ascii') {
    return {
      frame: ASCII_FRAMES[tick % ASCII_FRAMES.length] ?? '|',
      intervalMs: SPINNER_TICK_MS,
      showVerb: true
    }
  }

  // 'unicode' — braille spinner (fixed 1-col).  Authored interval is
  // ~80ms; honour it but bound below at a safe minimum so React
  // re-renders stay reasonable.  This style is for users who want
  // the cleanest possible status, so no verb rotation either.
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  return { frame, intervalMs: Math.max(SPINNER_TICK_MS, spinner.interval), showVerb: false }
}

// `FACES` / `EMOJI_FRAMES` are static, so measure their widest glyph once at
// module load instead of rescanning on every status render.
const KAOMOJI_FRAME_WIDTH = FACES.reduce((max, f) => Math.max(max, stringWidth(f)), 1)
const EMOJI_FRAME_WIDTH = EMOJI_FRAMES.reduce((max, f) => Math.max(max, stringWidth(f)), 1)

const indicatorFrameWidth = (style: IndicatorStyle): number => {
  if (style === 'kaomoji') {
    return KAOMOJI_FRAME_WIDTH
  }

  if (style === 'emoji') {
    return EMOJI_FRAME_WIDTH
  }

  // 'ascii' and 'unicode' are single-column glyphs.
  return 1
}

// Bounded width of the elapsed-time clock, derived from `fmtDuration` itself so
// the reservation/budget stays consistent with what actually renders (it emits
// a space between units, e.g. `59m 59s` / `99h 59m`). Durations beyond this
// (100h+) are left to clip rather than reserving unbounded width.
export const MAX_DURATION_WIDTH = Math.max(
  stringWidth(fmtDuration(59 * 60_000 + 59_000)), // "59m 59s"
  stringWidth(fmtDuration(99 * 3_600_000 + 59 * 60_000)) // "99h 59m"
)

// Display width to reserve for the busy indicator so its verb + elapsed-time
// tail can't shove the model off-screen on narrow terminals. Style-aware:
// `unicode` is a bare 1-col braille spinner with no verb, while kaomoji/emoji/
// ascii add a fixed-width verb; any style adds a bounded elapsed-time tail.
// Mirrors FaceTicker's `frame + verbSegment + durationSegment` layout.
export const busyIndicatorWidth = (style: IndicatorStyle, hasDuration: boolean): number => {
  const { showVerb } = renderIndicator(style, 0)
  const verb = showVerb ? 1 + VERB_PAD_LEN : 0
  // ` · ` plus the bounded clock (e.g. `59m 59s`).
  const duration = hasDuration ? stringWidth(' · ') + MAX_DURATION_WIDTH : 0

  return indicatorFrameWidth(style) + verb + duration
}

function FaceTicker({ color, startedAt, style }: { color: string; startedAt?: null | number; style: IndicatorStyle }) {
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000))
  const [verbTick, setVerbTick] = useState(() => Math.floor(Math.random() * VERBS.length))
  const [now, setNow] = useState(() => Date.now())

  // Pre-compute cadence + verb-visibility for the active style so an
  // `/indicator` switch re-arms the interval (and skips the verb timer
  // for verb-less styles like `unicode`) without leaving the previous
  // timer dangling.
  const { intervalMs, showVerb } = renderIndicator(style, 0)

  useEffect(() => {
    const glyph = setInterval(() => setTick(n => n + 1), intervalMs)
    const clock = setInterval(() => setNow(Date.now()), 1000)
    // Verb timer is gated on `showVerb` — `unicode` style hides the verb
    // entirely, so cycling `verbTick` would be an avoidable re-render.
    const verb = showVerb ? setInterval(() => setVerbTick(n => n + 1), FACE_TICK_MS) : null

    return () => {
      clearInterval(glyph)
      clearInterval(clock)

      if (verb !== null) {
        clearInterval(verb)
      }
    }
  }, [intervalMs, showVerb])

  const { frame } = renderIndicator(style, tick)
  const verb = VERBS[verbTick % VERBS.length] ?? ''
  const verbSegment = showVerb ? ` ${padVerb(verb)}` : ''
  // Leading space keeps a gap between the frame and the duration when the
  // verb segment is hidden (e.g. `unicode` spinner style).  When the verb
  // IS shown, its trailing padding already provides the gap, so the extra
  // space is harmless.
  const durationSegment = startedAt ? ` · ${fmtDuration(now - startedAt)}` : ''

  return (
    <Text color={color}>
      {frame}
      {verbSegment}
      {durationSegment}
    </Text>
  )
}

function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.muted
  }

  if (pct >= 95) {
    return t.color.statusCritical
  }

  if (pct > 80) {
    return t.color.statusBad
  }

  if (pct >= 50) {
    return t.color.statusWarn
  }

  return t.color.statusGood
}

function statusSessionCountLabel(count: number) {
  return `${count} ${count === 1 ? 'session' : 'sessions'}`
}

// Colour a credits notice by its level. The notice TEXT already carries its
// own glyph (⚠ • ✕ ✓) from the Python policy — we only tint it here, never
// prepend another glyph. `success` maps to the theme's green status colour.
function noticeColor(level: Notice['level'], t: Theme): string {
  if (level === 'error') {
    return t.color.error
  }

  if (level === 'warn') {
    return t.color.warn
  }

  if (level === 'success') {
    return t.color.statusGood
  }

  // 'info' / undefined — keep it readable but understated.
  return t.color.accent
}

function ctxBar(pct: number | undefined, w = 10) {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const filled = Math.round((p / 100) * w)

  return '█'.repeat(filled) + '░'.repeat(w - filled)
}

// `minLeftContent` is the display width of the high-priority left segments
// (status indicator + model + context). Reserving it makes the cwd/branch
// segment on the right yield FIRST on narrow terminals, instead of squeezing
// the loading indicator and model down to nothing.
export function statusRuleWidths(cols: number, cwdLabel: string, minLeftContent = 0) {
  const width = Math.max(1, Math.floor(cols || 1))
  const desiredSeparatorWidth = width >= 24 ? 3 : 1
  const baseMinLeft = width >= 24 ? 8 : 1
  // Never reserve more than the terminal width; never less than the historical
  // floor. With the default `minLeftContent = 0` this is identical to the old
  // behaviour, so callers that don't pass content are unaffected.
  const minLeftWidth = Math.min(width, Math.max(baseMinLeft, Math.floor(minLeftContent)))
  const maxRightWidth = Math.max(0, width - desiredSeparatorWidth - minLeftWidth)

  if (!cwdLabel || maxRightWidth <= 0) {
    return { leftWidth: width, rightWidth: 0, separatorWidth: 0 }
  }

  const rightWidth = Math.max(0, Math.min(stringWidth(cwdLabel), maxRightWidth))
  const separatorWidth = rightWidth > 0 ? desiredSeparatorWidth : 0
  const leftWidth = Math.max(1, width - separatorWidth - rightWidth)

  return { leftWidth, rightWidth, separatorWidth }
}

// Progressive disclosure for the status rule's lower-priority tail segments.
// As the terminal narrows we shed the least important pieces first (cost →
// bg → voice → compressions → duration → context bar), and below the bar
// breakpoint the context read-out collapses to a bare token count. Status and
// model are never gated here — they're guaranteed room by `statusRuleWidths`.
export interface StatusBarSegments {
  bar: boolean
  bg: boolean
  compactCtx: boolean
  compressions: boolean
  cost: boolean
  duration: boolean
  provider: boolean
  subagents: boolean
  tools: boolean
  voice: boolean
}

export function statusBarSegments(cols: number): StatusBarSegments {
  const w = Math.max(1, Math.floor(cols || 1))

  return {
    compactCtx: w < 72,
    bar: w >= 72,
    duration: w >= 76,
    compressions: w >= 80,
    voice: w >= 84,
    bg: w >= 88,
    subagents: w >= 92,
    tools: w >= 96,
    cost: w >= 104,
    provider: w >= 112
  }
}

function SpawnHud({ t }: { t: Theme }) {
  // Tight HUD that only appears when the session is actually fanning out.
  // Colour escalates to warn/error as depth or concurrency approaches the cap.
  const delegation = useStore($delegationState)
  const subagents = useTurnSelector(state => state.subagents)

  const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])

  if (!totals.descendantCount && !delegation.paused) {
    return null
  }

  const maxDepth = delegation.maxSpawnDepth
  const maxConc = delegation.maxConcurrentChildren
  const depth = Math.max(0, totals.maxDepthFromHere)
  const active = totals.activeCount

  // `max_concurrent_children` is a per-parent cap, not a global one.
  // `activeCount` sums every running agent across the tree and would
  // over-warn for multi-orchestrator runs.  The widest level of the tree
  // is a closer proxy to "most concurrent spawns that could be hitting a
  // single parent's slot budget".
  const widestLevel = widthByDepth(tree).reduce((a, b) => Math.max(a, b), 0)
  const depthRatio = maxDepth ? depth / maxDepth : 0
  const concRatio = maxConc ? widestLevel / maxConc : 0
  const ratio = Math.max(depthRatio, concRatio)

  const color = delegation.paused || ratio >= 1 ? t.color.error : ratio >= 0.66 ? t.color.warn : t.color.muted

  const pieces: string[] = []

  if (delegation.paused) {
    pieces.push('⏸ paused')
  }

  if (totals.descendantCount > 0) {
    const depthLabel = maxDepth ? `${depth}/${maxDepth}` : `${depth}`
    pieces.push(`d${depthLabel}`)

    if (active > 0) {
      // Label pairs the widest-level count (drives concRatio above) with
      // the total active count for context.  `W/cap` triggers the warn,
      // `+N` is everything else currently running across the tree.
      const extra = Math.max(0, active - widestLevel)
      const widthLabel = maxConc ? `${widestLevel}/${maxConc}` : `${widestLevel}`
      const suffix = extra > 0 ? `+${extra}` : ''
      pieces.push(`⚡${widthLabel}${suffix}`)
    }
  }

  const atCap = depthRatio >= 1 || concRatio >= 1

  return (
    <Text color={color}>
      {atCap ? ' │ ⚠ ' : ' │ '}
      {pieces.join(' ')}
    </Text>
  )
}

function SessionDuration({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [startedAt])

  return fmtDuration(now - startedAt)
}

function IdleSince({ endedAt }: { endedAt: number }) {
  // Time since the last final agent response. Re-ticks every second like
  // SessionDuration so the read-out stays live while the session idles.
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [endedAt])

  return `✓ ${fmtDuration(now - endedAt)}`
}

const effortLabel = (effort?: string) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  return value && value !== 'medium' && value !== 'normal' && value !== 'default' ? `R:${value}` : ''
}

const providerLabel = (provider?: string) => {
  const value = String(provider ?? '').trim()

  return value ? `via ${value}` : ''
}

const costLabel = (usage: Usage) => {
  if (typeof usage.cost_usd !== 'number' || !Number.isFinite(usage.cost_usd) || usage.cost_usd <= 0) {
    return ''
  }

  const prefix = usage.cost_status === 'estimated' ? '~' : ''

  return `${prefix}$${usage.cost_usd.toFixed(3)}`
}

const shortModelLabel = (model: string) =>
  model
    .split('/')
    .pop()!
    .replace(/^claude[-_]/, '')
    .replace(/^anthropic[-_]/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b(\d+)\s+(\d+)\b/g, '$1.$2')
    .trim()

const modelLabel = (model: string, effort?: string, fast?: boolean) =>
  [shortModelLabel(model), effortLabel(effort), fast ? 'fast' : ''].filter(Boolean).join(' ')

export function GoodVibesHeart({ tick, t }: { tick: number; t: Theme }) {
  const [active, setActive] = useState(false)
  const [color, setColor] = useState(t.color.accent)

  useEffect(() => {
    if (tick <= 0) {
      return
    }

    const palette = [t.color.error, t.color.warn, t.color.accent]
    setColor(palette[Math.floor(Math.random() * palette.length)]!)
    setActive(true)

    const id = setTimeout(() => setActive(false), 650)

    return () => clearTimeout(id)
  }, [t.color.accent, t.color.error, t.color.warn, tick])

  if (!active) {
    return null
  }

  return <Text color={color}>♥</Text>
}

export function StatusRule({
  cwdLabel,
  cols,
  busy,
  status,
  statusColor,
  model,
  modelProvider,
  modelFast,
  modelReasoningEffort,
  indicatorStyle = 'kaomoji',
  notice,
  usage,
  bgCount,
  lastTurnEndedAt,
  liveSessionCount,
  sessionStartedAt,
  turnStartedAt,
  voiceLabel,
  onSessionCountClick,
  t
}: StatusRuleProps) {
  const pct = usage.context_percent
  const barColor = ctxBarColor(pct, t)
  const segs = statusBarSegments(cols)

  // On narrow terminals the context read-out collapses to a bare token count
  // (`12k tok`) and the visual fill bar is dropped entirely.
  const ctxLabel = usage.context_max
    ? segs.compactCtx
      ? `${fmtK(usage.context_used ?? 0)} tok`
      : `${fmtK(usage.context_used ?? 0)}/${fmtK(usage.context_max)}`
    : usage.total > 0
      ? `${fmtK(usage.total)} tok`
      : ''

  const bar = !segs.compactCtx && usage.context_max ? ctxBar(pct) : ''
  const modelText = modelLabel(model, modelReasoningEffort, modelFast)

  // A credits notice replaces the status/verb slot, but only when idle —
  // while busy the FaceTicker always wins (R1 render priority). The notice
  // text carries its own glyph; we only tint it (R1) and let it shrink (R3-M7).
  const showNotice = !busy && !!notice?.text
  // The notice slot is shrinkable (flexShrink={1}, truncate-end), so reserve
  // only a small bounded width for it in the essentials budget — enough that
  // a short notice never gets crushed, but a long one ellipsizes instead of
  // shoving `model │ ctx` off-screen (R3-M7). Cap at the notice's own width
  // so short notices reserve exactly what they need.
  const NOTICE_RESERVE_MAX = 24
  const noticeReserve = showNotice ? Math.min(stringWidth(notice!.text), NOTICE_RESERVE_MAX) : 0

  // Width of the must-keep left segments (indicator + model + context). They
  // are pinned (never shrink) and reserved so the cwd/branch on the right
  // yields first. The busy face width depends on the active /indicator style
  // (kaomoji is wide + verb; unicode is a bare 1-col spinner). When a notice
  // occupies the slot it reserves only `noticeReserve` (it shrinks/truncates).
  const slotWidth = busy
    ? busyIndicatorWidth(indicatorStyle, turnStartedAt != null)
    : showNotice
      ? noticeReserve
      : stringWidth(status)

  const essentialWidth =
    stringWidth('─ ') +
    slotWidth +
    stringWidth(' │ ') +
    stringWidth(modelText) +
    (ctxLabel ? stringWidth(' │ ') + stringWidth(ctxLabel) : 0)

  const { leftWidth, rightWidth, separatorWidth } = statusRuleWidths(cols, cwdLabel, essentialWidth)

  // Whole-segment progressive disclosure for the tail: a segment renders only
  // if it fits in the space left after the pinned essentials, evaluated in
  // descending priority order — bar, duration, compressions, voice, session
  // count, bg, cost. Lower-priority segments drop first and nothing truncates
  // mid-segment, so status/model/context are never crushed.
  const SEP = stringWidth(' │ ')
  let tailBudget = Math.max(0, leftWidth - essentialWidth)

  const fits = (w: number) => {
    if (tailBudget >= w) {
      tailBudget -= w

      return true
    }

    return false
  }

  const sessionCountText = liveSessionCount > 0 ? statusSessionCountLabel(liveSessionCount) : ''
  const compressions = typeof usage.compressions === 'number' ? usage.compressions : 0
  const toolCalls = typeof usage.calls === 'number' ? usage.calls : 0
  const toolCallsText = toolCalls > 0 ? `tools ${toolCalls}` : ''
  const costText = costLabel(usage)
  const providerText = providerLabel(modelProvider)

  // Dev-only readout (HERMES_DEV_CREDITS). The server omits the key entirely unless the
  // flag is on, so this segment self-hides for normal users. micros→cents is allowed money
  // math (display formatting) — never parseFloat a *_usd. Signed: a mid-session top-up that
  // raises remaining nets a negative Δ (honest).
  const devCreditsText =
    typeof usage.dev_credits_spent_micros === 'number'
      ? `Δ ${(usage.dev_credits_spent_micros / 10000).toFixed(1)}¢`
      : ''

  const showBar = !!bar && fits(SEP + stringWidth(`[${bar}] ${pct != null ? `${pct}%` : ''}`))
  const showDuration = segs.duration && !!sessionStartedAt && fits(SEP + MAX_DURATION_WIDTH)

  // Idle clock — time since the last final agent response. Hidden while busy
  // (the FaceTicker's elapsed tail covers the live turn) and before the first
  // turn completes. Shares the duration breakpoint and width reservation.
  const showIdle =
    segs.duration && !busy && lastTurnEndedAt != null && fits(SEP + stringWidth('✓ ') + MAX_DURATION_WIDTH)

  const showCompressions = segs.compressions && compressions > 0 && fits(SEP + stringWidth(`cmp ${compressions}`))
  const showVoice = segs.voice && !!voiceLabel && fits(SEP + stringWidth(voiceLabel))
  const showSessionCount = !!sessionCountText && fits(SEP + stringWidth(sessionCountText))
  const showBg = segs.bg && bgCount > 0 && fits(SEP + stringWidth(`${bgCount} bg`))
  const subagentCount = typeof usage.active_subagents === 'number' ? usage.active_subagents : 0
  const showSubagents = segs.subagents && subagentCount > 0 && fits(SEP + stringWidth(`⛓ ${subagentCount}`))
  const showTools = segs.tools && !!toolCallsText && fits(SEP + stringWidth(toolCallsText))
  const showCost = segs.cost && !!costText && fits(SEP + stringWidth(costText))
  const showProvider = segs.provider && !!providerText && fits(SEP + stringWidth(providerText))

  // Parked-background reassurance: a top-level delegate_task runs in the
  // background, so the turn ends (idle) while the subagent keeps working and its
  // result re-enters as a fresh turn later. When idle with work still in flight,
  // spell out that the agent resumes on its own — no spinner, nothing to poll.
  // Width-budgeted like every tail segment, so it drops first on a tight
  // terminal where ⛓ already carries the signal.
  const resumeHintText =
    subagentCount === 1 ? '↩ resumes when subagent finishes' : `↩ resumes when ${subagentCount} subagents finish`

  const showResumeHint = !busy && subagentCount > 0 && fits(SEP + stringWidth(resumeHintText))
  // Dev-gated readout (HERMES_DEV_CREDITS), lowest priority,
  // so it consumes tail budget LAST and drops first on a narrow terminal.
  const showDevCredits = !!devCreditsText && fits(SEP + stringWidth(devCreditsText))

  const handleSessionCountClick = (event: { stopImmediatePropagation?: () => void }) => {
    event.stopImmediatePropagation?.()
    onSessionCountClick?.()
  }

  const sessionCountNode = onSessionCountClick ? (
    <Box flexShrink={0} onClick={handleSessionCountClick}>
      <Text color={t.color.accent}> │ {sessionCountText}</Text>
    </Box>
  ) : (
    <Text color={t.color.muted}> │ {sessionCountText}</Text>
  )

  return (
    <Box height={1}>
      <Box flexDirection="row" flexShrink={1} overflow="hidden" width={leftWidth}>
        {/* Leading pinned chrome: border + busy face / idle status. When a
            notice occupies the slot the status text is dropped — the notice
            renders as a separate shrinkable box below so a long notice
            ellipsizes instead of crushing model │ ctx (R3-M7). */}
        <Box flexDirection="row" flexShrink={0}>
          <Text color={t.color.border}>{'─ '}</Text>
          {busy ? (
            <FaceTicker color={statusColor} startedAt={turnStartedAt} style={indicatorStyle} />
          ) : showNotice ? null : (
            <Text color={statusColor} wrap="truncate-end">
              {status}
            </Text>
          )}
        </Box>
        {/* Notice slot — the only shrinkable left element (R3-M7). Sits in a
            flexShrink={1} box with truncate-end so it yields/ellipsizes
            before the pinned model │ ctx box ever clips. */}
        {showNotice ? (
          <Box flexDirection="row" flexShrink={1} overflow="hidden">
            <Text color={noticeColor(notice!.level, t)} wrap="truncate-end">
              {notice!.text}
            </Text>
          </Box>
        ) : null}
        {/* Pinned essentials — model + context never shrink, always visible. */}
        <Box flexDirection="row" flexShrink={0}>
          {DEV_CREDITS_MODE ? (
            <Text color={t.color.warn} wrap="truncate-end">
              {' (dev credits)'}
            </Text>
          ) : null}
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            {modelText}
          </Text>
          {ctxLabel ? (
            <Text color={t.color.muted} wrap="truncate-end">
              {' │ '}
              {ctxLabel}
            </Text>
          ) : null}
        </Box>
        {showBar ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            <Text color={barColor}>[{bar}]</Text> <Text color={barColor}>{pct != null ? `${pct}%` : ''}</Text>
          </Text>
        ) : null}
        {showDuration ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            <SessionDuration startedAt={sessionStartedAt!} />
          </Text>
        ) : null}
        {showIdle ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            <IdleSince endedAt={lastTurnEndedAt!} />
          </Text>
        ) : null}
        {showCompressions ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            <Text color={compressions >= 10 ? t.color.error : compressions >= 5 ? t.color.warn : t.color.muted}>
              cmp {compressions}
            </Text>
          </Text>
        ) : null}
        {showVoice ? (
          <Text
            color={
              voiceLabel!.startsWith('●') ? t.color.error : voiceLabel!.startsWith('◉') ? t.color.warn : t.color.muted
            }
            wrap="truncate-end"
          >
            {' │ '}
            {voiceLabel}
          </Text>
        ) : null}
        {showSessionCount ? sessionCountNode : null}
        {showBg ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            {bgCount} bg
          </Text>
        ) : null}
        {showSubagents ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}⛓ {subagentCount}
          </Text>
        ) : null}
        {showTools ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            {toolCallsText}
          </Text>
        ) : null}
        {showCost ? (
          <Text color={usage.cost_status === 'exact' ? t.color.statusGood : t.color.muted} wrap="truncate-end">
            {' │ '}
            {costText}
          </Text>
        ) : null}
        {showProvider ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {' │ '}
            {providerText}
          </Text>
        ) : null}
        {showResumeHint ? (
          <Text color={t.color.muted} dim wrap="truncate-end">
            {' │ '}
            {resumeHintText}
          </Text>
        ) : null}
        {showDevCredits ? (
          <Text color={t.color.accent} wrap="truncate-end">
            {' │ '}
            {devCreditsText}
          </Text>
        ) : null}
        {/* SpawnHud isn't part of the tail budget (its width is dynamic), so it
            renders last — any overflow truncates the HUD itself rather than the
            budgeted segments before it. It self-hides when no delegation runs. */}
        <SpawnHud t={t} />
      </Box>

      {rightWidth > 0 ? (
        <>
          <Text color={t.color.border}>{separatorWidth >= 3 ? ' ─ ' : ' '}</Text>
          <Box flexShrink={0} width={rightWidth}>
            <Text color={t.color.label} wrap="truncate-end">
              {cwdLabel}
            </Text>
          </Box>
        </>
      ) : null}
    </Box>
  )
}

export function FloatBox({ children, color }: { children: ReactNode; color: string }) {
  return (
    <Box
      alignSelf="flex-start"
      borderColor={color}
      borderStyle="double"
      flexDirection="column"
      marginTop={1}
      opaque
      paddingX={1}
    >
      {children}
    </Box>
  )
}

export function StickyPromptTracker({ messages, offsets, scrollRef, onChange }: StickyPromptTrackerProps) {
  const { atBottom, bottom, top } = useViewportSnapshot(scrollRef)
  const text = stickyPromptFromViewport(messages, offsets, top, bottom, atBottom)

  useEffect(() => onChange(text), [onChange, text])

  return null
}

export function TranscriptScrollbar({ scrollRef, t }: TranscriptScrollbarProps) {
  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<number | null>(null)
  const grabRef = useRef<number | null>(null)
  const { scrollHeight: total, top: pos, viewportHeight: vp } = useScrollbarSnapshot(scrollRef)

  if (!vp) {
    return <Box width={1} />
  }

  const s = scrollRef.current
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const thumbColor = grab !== null ? t.color.primary : hover ? t.color.accent : t.color.border
  const trackColor = hover ? t.color.border : t.color.muted

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)

        grabRef.current = off
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grabRef.current ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => {
        grabRef.current = null
        setGrab(null)
      }}
      width={1}
    >
      {!scrollable ? (
        <Text color={trackColor} dim>
          {' \n'.repeat(Math.max(0, vp - 1))}{' '}
        </Text>
      ) : (
        <>
          {thumbTop > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {`${'│\n'.repeat(Math.max(0, thumbTop - 1))}${thumbTop > 0 ? '│' : ''}`}
            </Text>
          ) : null}
          {thumb > 0 ? (
            <Text color={thumbColor}>{`${'┃\n'.repeat(Math.max(0, thumb - 1))}${thumb > 0 ? '┃' : ''}`}</Text>
          ) : null}
          {vp - thumbTop - thumb > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {`${'│\n'.repeat(Math.max(0, vp - thumbTop - thumb - 1))}${vp - thumbTop - thumb > 0 ? '│' : ''}`}
            </Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

interface StatusRuleProps {
  bgCount: number
  lastTurnEndedAt?: null | number
  liveSessionCount: number
  busy: boolean
  cols: number
  cwdLabel: string
  model: string
  modelProvider?: string
  modelFast?: boolean
  modelReasoningEffort?: string
  indicatorStyle?: IndicatorStyle
  notice?: Notice | null
  sessionStartedAt?: null | number
  status: string
  statusColor: string
  t: Theme
  turnStartedAt?: null | number
  usage: Usage
  voiceLabel?: string
  onSessionCountClick?: () => void
}

interface StickyPromptTrackerProps {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  onChange: (text: string) => void
  scrollRef: RefObject<ScrollBoxHandle | null>
}

interface TranscriptScrollbarProps {
  scrollRef: RefObject<ScrollBoxHandle | null>
  t: Theme
}
