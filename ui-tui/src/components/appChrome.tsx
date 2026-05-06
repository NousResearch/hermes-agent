import { Box, type ScrollBoxHandle, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, type RefObject, useEffect, useMemo, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import { $delegationState } from '../app/delegationStore.js'
import type { IndicatorStyle } from '../app/interfaces.js'
import { $sessionTodos, useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { buildSubagentTree, treeTotals, widthByDepth } from '../lib/subagentTree.js'
import { fmtK, isToolTrailResultLine, parseToolTrailResultLine, splitToolDuration, toolTrailLabel } from '../lib/text.js'
import { useGitStatus } from '../lib/useGitStatus.js'
import { useViewportSnapshot } from '../lib/viewportStore.js'
import type { Theme } from '../theme.js'
import type { Msg, TodoItem, Usage } from '../types.js'

const FACE_TICK_MS = 2500
const HEART_COLORS = ['#ff5fa2', '#ff4d6d']

// Keep verb segment width stable so status-bar content to the right doesn't
// jitter when the ticker rotates between short/long verbs.
export const VERB_PAD_LEN = VERBS.reduce((max, v) => Math.max(max, v.length), 0) + 1 // + ellipsis
export const DURATION_PAD_LEN = 7 // e.g. "  9s", "1m 05s", "59m 59s"
export const padVerb = (verb: string) => `${verb}…`.padEnd(VERB_PAD_LEN, ' ')
export const padTickerDuration = (ms: number) => fmtDuration(ms).padStart(DURATION_PAD_LEN, ' ')

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

function FaceTicker({ color, now, startedAt }: { color: string; now: number; startedAt?: null | number }) {
  const ui = useStore($uiState)
  const style = ui.indicatorStyle
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000))
  const [verbTick, setVerbTick] = useState(() => Math.floor(Math.random() * VERBS.length))

  // Pre-compute cadence + verb-visibility for the active style so an
  // `/indicator` switch re-arms the interval (and skips the verb timer
  // for verb-less styles like `unicode`) without leaving the previous
  // timer dangling.
  const { intervalMs, showVerb } = renderIndicator(style, 0)

  useEffect(() => {
    const glyph = setInterval(() => setTick(n => n + 1), intervalMs)
    // Verb timer is gated on `showVerb` — `unicode` style hides the verb
    // entirely, so cycling `verbTick` would be an avoidable re-render.
    const verb = showVerb ? setInterval(() => setVerbTick(n => n + 1), FACE_TICK_MS) : null

    return () => {
      clearInterval(glyph)

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
  const durationSegment = startedAt ? ` · ${padTickerDuration(now - startedAt)}` : ''

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

function ctxBar(pct: number | undefined, w = 10) {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const filled = Math.round((p / 100) * w)

  return '█'.repeat(filled) + '░'.repeat(w - filled)
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

function SessionDuration({ now, startedAt }: { now: number; startedAt: number }) {
  return fmtDuration(now - startedAt)
}

const effortLabel = (effort?: string) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  return value && value !== 'medium' && value !== 'normal' && value !== 'default' ? value : ''
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

// ── Todo progress: "✓ 2/5" or "✓ 3" when all done ──
// Reads from the persistent $sessionTodos atom so the count survives
// turn resets (unlike $turnState.todos which is cleared at turn end).
function TodoSummary() {
  const todos = useStore($sessionTodos)

  if (!todos.length) {
    return null
  }

  const done = todos.filter(t => t.status === 'completed').length
  const total = todos.length

  return (
    <Text color={done === total ? t_color_good : t_color_muted}>
      {' │ ✓ '}
      {done}
      {done < total ? `/${total}` : ''}
    </Text>
  )
}

// ── Agent HUD panel: tool usage summary + MCP count + recent trail ──
function AgentHudPanel({ cols }: { cols: number }) {
  const ui = useStore($uiState)
  const trail = useTurnSelector(state => state.turnTrail)
  const skills = ui.info?.skills
  const mcpServers = ui.info?.mcp_servers

  // 1. Aggregate tool counts from trail: "✓ Edit ×10 | ✓ Write ×5 | ✓ Read ×2"
  const counts = new Map<string, { fail: number; ok: number }>()
  for (const line of trail) {
    if (!isToolTrailResultLine(line)) continue
    const parsed = parseToolTrailResultLine(line)
    if (!parsed) continue
    const { label } = splitToolDuration(parsed.call)
    const entry = counts.get(label) ?? { fail: 0, ok: 0 }
    if (parsed.mark === '✗') entry.fail++
    else entry.ok++
    counts.set(label, entry)
  }

  const toolSummaryPieces: string[] = []
  // Sort by total count desc, then alphabetical
  const sorted = [...counts.entries()].sort((a, b) => {
    const ta = a[1].ok + a[1].fail
    const tb = b[1].ok + b[1].fail
    return tb !== ta ? tb - ta : a[0].localeCompare(b[0])
  })

  for (const [name, { fail, ok }] of sorted) {
    const total = ok + fail
    const prefix = fail > 0 ? '✗' : '✓'
    const display = name.length > 16 ? name.slice(0, 15) + '…' : name
    toolSummaryPieces.push(`${prefix} ${display} ×${total}`)
  }

  // 2. MCP count
  const mcpCount = mcpServers?.length ?? 0

  // 3. Skill count
  const skillCount = skills
    ? Object.values(skills).flat().filter(Boolean).length
    : 0

  // 4. Recent trail lines (last 3 non-transient)
  const recentLines = trail
    .filter(l => isToolTrailResultLine(l) && l !== 'analyzing tool output…')
    .slice(-3)

  // Nothing to show
  if (!toolSummaryPieces.length && !mcpCount && !skillCount && !recentLines.length) {
    return null
  }

  // Build rows — adaptive to available width
  const rows: ReactNode[] = []

  // Row A: Tool usage summary (e.g. "✓ Edit ×10 | ✓ Write ×5 | ✓ Read ×2")
  if (toolSummaryPieces.length) {
    const maxPieces = cols > 100 ? 6 : cols > 70 ? 4 : 2
    const visible = toolSummaryPieces.slice(0, maxPieces)
    const rest = toolSummaryPieces.length - maxPieces
    rows.push(
      <Text key="tools" color={t_color_muted}>
        {visible.join(' │ ')}
        {rest > 0 ? <Text> │ +{rest}</Text> : null}
      </Text>
    )
  }

  // Row B: MCP + Skills counts
  const metaPieces: string[] = []
  if (mcpCount > 0) metaPieces.push(`${mcpCount} MCP`)
  if (skillCount > 0) metaPieces.push(`${skillCount} skills`)
  if (metaPieces.length) {
    rows.push(
      <Text key="meta" color={t_color_muted}>
        {metaPieces.join(' · ')}
      </Text>
    )
  }

  // Row C: Recent trail (up to 2 lines, compact)
  if (recentLines.length) {
    const maxTrail = cols > 100 ? 2 : 1
    for (const line of recentLines.slice(-maxTrail)) {
      const parsed = parseToolTrailResultLine(line)
      if (!parsed) continue
      const { label, duration } = splitToolDuration(parsed.call)
      const isError = parsed.mark === '✗'
      const displayLabel = label.length > 24 ? label.slice(0, 23) + '…' : label
      const durationStr = duration || ''
      const detail = parsed.detail
        ? parsed.detail.length > 40
          ? parsed.detail.slice(0, 39) + '…'
          : parsed.detail
        : ''
      const fullLine = `${parsed.mark} ${displayLabel}${durationStr}${detail ? ` :: ${detail}` : ''}`
      rows.push(
        <Text key={line} color={isError ? t_color_good : t_color_muted} dim={!isError}>
          {fullLine.length > cols - 4 ? fullLine.slice(0, cols - 5) + '…' : fullLine}
        </Text>
      )
    }
  }

  if (!rows.length) return null

  return (
    <Box flexDirection="column" marginLeft={1}>
      {rows}
    </Box>
  )
}

// Module-level theme color references — resolved lazily from the Theme
// object passed via props to avoid needing context or global state.
// These are overwritten by the first StatusRule render.
let t_color_good = '#5f875f'
let t_color_muted = '#888'

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
  }, [t.color.accent, tick])

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
  modelFast,
  modelReasoningEffort,
  usage,
  bgCount,
  sessionStartedAt,
  showCost,
  turnStartedAt,
  voiceLabel,
  collapsed,
  t
}: StatusRuleProps) {
  // Resolve theme colors for child components that read from stores directly.
  t_color_good = t.color.statusGood
  t_color_muted = t.color.muted

  // Single shared clock for all time-dependent children — avoids N
  // independent setInterval timers (FaceTicker, SessionDuration).
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [])

  // Git status from cwd
  const ui = useStore($uiState)
  const cwd = ui.info?.cwd ?? process.cwd()
  const git = useGitStatus(cwd)

  const pct = usage.context_percent

  // context_used may be 0 right after a model switch (before next API call).
  // Fall back to session input tokens so the bar stays meaningful.
  const effectiveCtxUsed = (usage.context_used ?? 0) || usage.input || 0

  // Recalculate percent with the effective value when context_used was stale.
  const effectivePct = usage.context_max && usage.context_used === 0 && usage.input > 0
    ? Math.max(0, Math.min(100, Math.round((usage.input / usage.context_max) * 100)))
    : pct

  const barColor = ctxBarColor(effectivePct, t)

  const ctxLabel = usage.context_max
    ? `${fmtK(effectiveCtxUsed)}/${fmtK(usage.context_max)}`
    : usage.total > 0
      ? `${fmtK(usage.total)} tok`
      : ''

  // Adaptive context bar width based on terminal columns.
  const barWidth = cols > 120 ? 14 : cols > 80 ? 10 : 6
  const bar = usage.context_max ? ctxBar(effectivePct, barWidth) : ''

  const modelTag = modelLabel(model, modelReasoningEffort, modelFast)

  // Token breakdown: always show in/out when data is available
  const tokenDetail = (usage.input > 0 || usage.output > 0)
    ? `(in: ${fmtK(usage.input)}, out: ${fmtK(usage.output)})`
    : ''

  // Todo + tools (hooks must be unconditional)
  // Use $sessionTodos so the count persists across turns.
  const todos = useStore($sessionTodos)
  const todoDone = todos.filter(td => td.status === 'completed').length
  const todoTotal = todos.length

  // ── Collapsed mode: single-line compact summary ──
  if (collapsed) {
    const corePieces: string[] = []

    if (busy) {
      corePieces.push(turnStartedAt ? fmtDuration(now - turnStartedAt) : '…')
    } else {
      corePieces.push(status)
    }

    if (usage.provider) {
      corePieces.push(usage.provider)
    }

    if (git.branch) {
      corePieces.push(`git:(${git.branch}${git.dirty ? '*' : ''})`)
    }

    if (ctxLabel) {
      corePieces.push(`${ctxLabel} ${effectivePct != null ? `${effectivePct}%` : ''}`)
    }

    if ((usage.cache_read ?? 0) > 0) {
      corePieces.push(`cache: ${fmtK(usage.cache_read ?? 0)}`)
    }

    if (showCost && typeof usage.cost_usd === 'number') {
      corePieces.push(`$${usage.cost_usd.toFixed(2)}`)
    }

    // Compressions in collapsed
    if (usage.compressions !== undefined) {
      corePieces.push(`${usage.compressions ?? 0} compress`)
    }

    // Todo count in collapsed
    if (todoTotal > 0) {
      corePieces.push(`✓${todoDone}/${todoTotal}`)
    }

    return (
      <Box height={1}>
        <Text color={t.color.border} wrap="truncate-end">
          {'─ '}
          <Text color={statusColor}>{corePieces.join(' · ')}</Text>
          {bar ? (
            <Text color={t.color.muted}>
              {' '}
              <Text color={barColor}>[{bar}]</Text>
            </Text>
          ) : null}
        </Text>
        <Text color={t.color.border}> ─ </Text>
        <Text color={t.color.label}>{cwdLabel}</Text>
      </Box>
    )
  }

  // ── Full mode (2 rows when cols >= 60, 1 row otherwise) ──
  const useTwoRows = cols >= 60

  const leftWidth = Math.max(12, cols - cwdLabel.length - 3)

  // Provider label (from usage)
  const providerLabel = usage.provider ? `${usage.provider}` : ''

  // Cache display
  const cacheLabel = (usage.cache_read ?? 0) > 0 ? `cache: ${fmtK(usage.cache_read ?? 0)}` : ''

  // Output speed
  const speedLabel = (usage.output_speed ?? 0) > 0 ? `out: ${(usage.output_speed ?? 0).toFixed(1)} tok/s` : ''

  // Cost display
  const costLabel = showCost && typeof usage.cost_usd === 'number' ? `$${usage.cost_usd.toFixed(2)}` : ''

  // Compressions
  const compressLabel = (usage.compressions ?? 0) >= 0 && 'compressions' in usage
    ? `${usage.compressions ?? 0} compress`
    : ''

  // ── Row 1: status, model, git, context ──
  const row1 = (
    <Text color={t.color.border} wrap="truncate-end">
      {'─ '}
      {busy ? (
        <FaceTicker color={statusColor} now={now} startedAt={turnStartedAt} />
      ) : (
        <Text color={statusColor}>{status}</Text>
      )}
      <Text color={t.color.muted}> │ {modelTag}</Text>
      {providerLabel ? (
        <Text color={t.color.muted}> · {providerLabel}</Text>
      ) : null}
      {git.branch ? (
        <Text color={git.dirty ? t.color.warn : t.color.muted}>
          {' │ git:('}
          {git.branch}
          {git.dirty ? '*' : ''}
          {')'}
        </Text>
      ) : null}
      {ctxLabel ? <Text color={t.color.muted}> │ {ctxLabel}</Text> : null}
      {bar ? (
        <Text color={t.color.muted}>
          {' │ '}
          <Text color={barColor}>[{bar}]</Text> <Text color={barColor}>{effectivePct != null ? `${effectivePct}%` : ''}</Text>
        </Text>
      ) : null}
      {tokenDetail ? <Text color={t.color.muted}> {tokenDetail}</Text> : null}
      <SpawnHud t={t} />
      <TodoSummary />
    </Text>
  )

  // ── Row 2: calls, cache, compress, session duration, speed, cost, voice, bg ──
  const row2 = (
    <Text color={t.color.muted} wrap="truncate-end">
      {usage.calls > 0 ? <Text> │ {usage.calls} calls</Text> : null}
      {cacheLabel ? <Text> │ {cacheLabel}</Text> : null}
      {compressLabel ? <Text> │ {compressLabel}</Text> : null}
      {sessionStartedAt ? (
        <Text>
          {' │ '}
          <SessionDuration now={now} startedAt={sessionStartedAt} />
        </Text>
      ) : null}
      {speedLabel ? <Text> │ {speedLabel}</Text> : null}
      {costLabel ? <Text> │ {costLabel}</Text> : null}
      {voiceLabel ? (
        <Text
          color={
            voiceLabel.startsWith('●') ? t.color.error : voiceLabel.startsWith('◉') ? t.color.warn : t.color.muted
          }
        >
          {' │ '}
          {voiceLabel}
        </Text>
      ) : null}
      {bgCount > 0 ? <Text> │ {bgCount} bg</Text> : null}
    </Text>
  )

  // ── Row 3+: Agent HUD panel (tool summary, MCP, recent trail) ──
  const hudPanel = <AgentHudPanel cols={cols} />

  if (useTwoRows) {
    return (
      <Box flexDirection="column">
        <Box height={1}>
          <Box flexShrink={1} width={leftWidth}>{row1}</Box>
          <Text color={t.color.border}> ─ </Text>
          <Text color={t.color.label}>{cwdLabel}</Text>
        </Box>
        <Box height={1}>
          <Box flexShrink={1} width={leftWidth}>{row2}</Box>
          <Text color={t.color.border}> ─ </Text>
          <Text color={t.color.muted}>{cwdLabel}</Text>
        </Box>
        {hudPanel}
      </Box>
    )
  }

  // Fallback: single row for very narrow terminals
  return (
    <Box flexDirection="column">
      <Box height={1}>
        <Box flexShrink={1} width={leftWidth}>
          {row1}
          {row2}
        </Box>
        <Text color={t.color.border}> ─ </Text>
        <Text color={t.color.label}>{cwdLabel}</Text>
      </Box>
      {hudPanel}
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
  const { scrollHeight: total, top: pos, viewportHeight: vp } = useViewportSnapshot(scrollRef)

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
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
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
  busy: boolean
  collapsed?: boolean
  cols: number
  cwdLabel: string
  model: string
  modelFast?: boolean
  modelReasoningEffort?: string
  sessionStartedAt?: null | number
  showCost: boolean
  status: string
  statusColor: string
  t: Theme
  turnStartedAt?: null | number
  usage: Usage
  voiceLabel?: string
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
