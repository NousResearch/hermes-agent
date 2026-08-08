import { Box, stringWidth, Text, useStdout } from '@hermes/ink'
import React, { memo, useEffect, useState } from 'react'

import { useTurnSelector } from '../app/turnStore.js'
import { type DockRow, type DockTone, type DockView, projectAgentDock } from '../lib/agentDock.js'
import { isRunning } from '../lib/subagentTree.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

/**
 * B3 "Framed Simple Ledger" — the approved resting dock. One continuous
 * muted single-line perimeter around the agent region; the `agents` label,
 * truthful header counts, and the `/agents ↗` affordance are integrated into
 * the top edge. At most three projection rows plus one truthful aggregate line
 * render inside. Below `DOCK_NARROW_WIDTH` the frame is omitted entirely and
 * the tested one-line summary renders instead.
 *
 * All layout math uses `stringWidth` (terminal cells), never `.length`, so
 * status glyphs and bounded labels cannot shear the right border. Colored ink
 * frame is bounded: one status glyph per row; header label and affordance
 * keep their established accent/label tones; everything else is muted.
 */

export interface AgentDockProps {
  cols: number
  onOpen: () => void
  t: Theme
}

export interface AgentDockViewProps extends AgentDockProps {
  nowMs: number
  subagents: readonly SubagentProgress[]
  summaryOnly?: boolean
}

interface Seg {
  bold?: boolean
  color?: string
  text: string
}

const CALLSIGN_CELL = 12
/** Minimum run of `─` before the top-right corner. */
const MIN_TOP_FILL = 1
/** Preserve transcript/composer room on unusually short terminals. */
const MIN_FULL_DOCK_ROWS = 16

function useTerminalRows(): number {
  const { stdout } = useStdout()
  const stream = stdout ?? process.stdout
  const [rows, setRows] = useState(() => stream.rows ?? 24)

  useEffect(() => {
    const syncRows = () => setRows(stream.rows ?? 24)

    syncRows()
    stream.on('resize', syncRows)

    return () => {
      stream.off('resize', syncRows)
    }
  }, [stream])

  return rows
}

const toneColor = (tone: DockTone, t: Theme): string => {
  switch (tone) {
    case 'accent':
      return t.color.accent

    case 'error':
      return t.color.error

    case 'statusGood':
      return t.color.statusGood

    case 'warn':
      return t.color.warn

    case 'muted':
      return t.color.muted
  }
}

const segsWidth = (segs: readonly Seg[]): number => segs.reduce((total, item) => total + stringWidth(item.text), 0)

/** Truncate to a cell budget with a terminal-cell walk, ellipsis included. */
const clipToWidth = (text: string, maxWidth: number): string => {
  if (stringWidth(text) <= maxWidth) {
    return text
  }

  let out = ''
  let used = 0

  for (const char of text) {
    const charWidth = stringWidth(char)

    if (used + charWidth > maxWidth - 1) {
      break
    }

    out += char
    used += charWidth
  }

  return `${out}…`
}

const padToWidth = (text: string, width: number): string => {
  const remaining = width - stringWidth(text)

  return remaining > 0 ? text + ' '.repeat(remaining) : text
}

function FrameLine({ segs }: { segs: readonly Seg[] }) {
  return (
    <Text wrap="truncate-end">
      {segs.map((item, index) => (
        <Text bold={item.bold} color={item.color} key={index}>
          {item.text}
        </Text>
      ))}
    </Text>
  )
}

/** `╭─ agents · {header} · /agents ↗ ──…──╮`, exactly `cols` cells. When the
 * header counts cannot fit, their trailing segments drop one by one before
 * the corners or the `/agents ↗` affordance are ever clipped. */
function topEdge(view: DockView, t: Theme, cols: number): Seg[] {
  const inner = cols - 2
  const affordance = '/agents ↗'
  const fixedWidth = stringWidth('─ ') + stringWidth('agents') + stringWidth(' · ') + stringWidth(affordance) + stringWidth(' ')

  const allParts = view.header.length > 0 ? view.header.split(' · ') : []
  let parts = allParts
  let omitted = false

  const partsWidth = (items: readonly string[]): number =>
    items.reduce((total, item) => total + stringWidth(' · ') + stringWidth(item), 0)

  const omissionWidth = () => (omitted ? stringWidth(' · …') : 0)

  while (parts.length > 0 && fixedWidth + partsWidth(parts) + omissionWidth() > inner - MIN_TOP_FILL) {
    parts = parts.slice(0, -1)
    omitted = true
  }

  const fill = Math.max(MIN_TOP_FILL, inner - fixedWidth - partsWidth(parts) - omissionWidth())

  return [
    { color: t.color.border, text: '╭─ ' },
    { bold: true, color: t.color.accent, text: 'agents' },
    ...parts.flatMap((part): Seg[] => [
      { color: t.color.muted, text: ' · ' },
      { color: t.color.muted, text: part }
    ]),
    ...(omitted
      ? [
          { color: t.color.muted, text: ' · ' },
          { color: t.color.muted, text: '…' }
        ]
      : []),
    { color: t.color.muted, text: ' · ' },
    { color: t.color.label, text: affordance },
    { color: t.color.border, text: ` ${'─'.repeat(fill)}╮` }
  ]
}

function bottomEdge(t: Theme, cols: number): Seg[] {
  return [{ color: t.color.border, text: `╰${'─'.repeat(Math.max(0, cols - 2))}╯` }]
}

/** Wrap interior content with `│ … │`, padded to exactly `cols` cells. */
function interior(content: readonly Seg[], t: Theme, cols: number): Seg[] {
  const inner = cols - 2
  const pad = Math.max(0, inner - segsWidth(content))

  return [
    { color: t.color.border, text: '│' },
    ...content,
    { text: ' '.repeat(pad) },
    { color: t.color.border, text: '│' }
  ]
}

function rowContent(row: DockRow, t: Theme, cols: number): Seg[] {
  const indent = '  '.repeat(Math.min(Math.max(row.depth, 0), 2))

  const lead: Seg[] = [
    { text: ` ${indent}` },
    { color: toneColor(row.tone, t), text: row.glyph },
    { text: ' ' },
    { color: t.color.text, text: padToWidth(row.callsign, CALLSIGN_CELL) },
    { text: '  ' }
  ]

  const trail = row.elapsed ? `${row.activity} · ${row.elapsed}` : row.activity
  const budget = cols - 2 - segsWidth(lead)

  return [...lead, { color: t.color.muted, text: clipToWidth(trail, Math.max(0, budget)) }]
}

/** Projection-owned truthful aggregate, ellipsis-indented and clipped so the
 * framed line stays exact even for long summaries. */
function aggregateContent(view: DockView, t: Theme, cols: number): Seg[] {
  return [{ color: t.color.muted, text: clipToWidth(` … ${view.overflowSummary}`, Math.max(0, cols - 2)) }]
}

/**
 * Render-only half of the dock. Keeping projection inputs explicit makes
 * hidden, narrow, framed, overflow, and click behavior deterministic in
 * tests without a live store or terminal.
 */
export function AgentDockView({ cols, nowMs, onOpen, subagents, summaryOnly = false, t }: AgentDockViewProps) {
  const view = projectAgentDock(subagents, { forceSummary: summaryOnly, nowMs, width: cols })

  if (view.hidden) {
    return null
  }

  const open = (event?: { stopImmediatePropagation?: () => void }) => {
    event?.stopImmediatePropagation?.()
    onOpen()
  }

  if (view.summaryOnly) {
    // No marginTop here: a narrow terminal cannot spare a blank row, and the
    // one-line summary must stay exactly one line.
    return (
      <Box flexDirection="column" flexShrink={0} onClick={open}>
        <Text wrap="truncate-end">
          <Text bold color={t.color.accent}>
            agents
          </Text>
          <Text color={t.color.label}> · /agents ↗</Text>
          <Text color={t.color.muted}> · {view.summary}</Text>
        </Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} onClick={open}>
      <FrameLine segs={topEdge(view, t, cols)} />

      {view.rows.map(row => (
        <FrameLine key={row.id} segs={interior(rowContent(row, t, cols), t, cols)} />
      ))}

      {view.overflowSummary !== '' ? <FrameLine segs={interior(aggregateContent(view, t, cols), t, cols)} /> : null}

      <FrameLine segs={bottomEdge(t, cols)} />
    </Box>
  )
}

/**
 * Live store wrapper. A one-second clock is active only while at least one
 * row has a running/queued start time without a terminal duration.
 */
export const AgentDock = memo(function AgentDock({ cols, onOpen, t }: AgentDockProps) {
  const subagents = useTurnSelector(state => state.subagents)
  const terminalRows = useTerminalRows()
  const [nowMs, setNowMs] = useState(() => Date.now())
  const needsClock = subagents.some(item => item.startedAt != null && item.durationSeconds == null && isRunning(item))

  useEffect(() => {
    if (!needsClock) {
      return
    }

    setNowMs(Date.now())
    const timer = setInterval(() => setNowMs(Date.now()), 1000)

    return () => clearInterval(timer)
  }, [needsClock])

  return (
    <AgentDockView
      cols={cols}
      nowMs={nowMs}
      onOpen={onOpen}
      subagents={subagents}
      summaryOnly={terminalRows < MIN_FULL_DOCK_ROWS}
      t={t}
    />
  )
})
