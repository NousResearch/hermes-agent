import { Box, stringWidth, Text, useStdout } from '@hermes/ink'
import { mix } from '@hermes/shared/color'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $agentDockCollapsed, useAgentRoster } from '../app/agentRoster.js'
import { type ProcessRow, useProcessRows } from '../app/processRoster.js'
import { $uiState } from '../app/uiStore.js'
import { type AgentRows, buildAgentRows, dockRowLimit } from '../lib/agentRows.js'
import { processGlyph } from '../lib/processGlyph.js'
import { statusGlyph } from '../lib/subagentGlyph.js'
import { fmtDuration } from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export interface ProcessBlock {
  /** Rows that exist but were cut by the height bound. */
  hidden: number
  rows: ProcessRow[]
  running: number
  total: number
}

export const EMPTY_PROCESSES: ProcessBlock = { hidden: 0, rows: [], running: 0, total: 0 }

/** Height-bound the process rows the way `buildAgentRows` bounds agents; callers
 * split one dock budget between the two blocks. */
export const buildProcessBlock = (rows: readonly ProcessRow[], maxRows: number): ProcessBlock => {
  const shown = maxRows > 0 ? rows.slice(0, maxRows) : [...rows]

  return {
    hidden: rows.length - shown.length,
    rows: shown,
    running: rows.filter(r => r.status === 'running').length,
    total: rows.length
  }
}

/** Split a dock row budget when both blocks are present so neither hides the other. */
export const splitDockBudget = (
  limit: number,
  agents: number,
  processes: number
): { agents: number; processes: number } => {
  if (!processes) {
    return { agents: limit, processes: 0 }
  }

  const agentBudget = Math.min(agents, Math.max(1, limit - Math.max(1, Math.floor(limit / 2))))

  return { agents: agentBudget, processes: Math.max(1, limit - agentBudget) }
}

export const processSummary = (block: ProcessBlock): string => {
  const done = block.total - block.running

  return [block.running ? `${block.running} running` : '', done ? `${done} done` : ''].filter(Boolean).join(' · ')
}

/** One `⚙ command · 42s · last: …` line per process; the exit verdict replaces the
 * activity once the process has finished. */
export function ProcessRowLine({ cols, row, t }: { cols: number; row: ProcessRow; t: Theme }) {
  const glyph = processGlyph(row.status, t)
  const activity = row.status === 'running' ? `${fmtDuration(row.elapsedSeconds)} · ${row.detail}` : row.detail
  const commandWidth = Math.max(8, cols - stringWidth(activity) - 6)

  return (
    <Text wrap="truncate-end">
      <Text color={glyph.color}>{glyph.glyph} </Text>
      <Text color={t.color.text}>{compactPreview(row.command, commandWidth)}</Text>
      <Text color={t.color.muted}> · {activity}</Text>
    </Text>
  )
}

export function AgentsPanelView({
  collapsed = false,
  cols,
  hidden,
  processes = EMPTY_PROCESSES,
  rows,
  running,
  t
}: AgentRows & { collapsed?: boolean; cols: number; processes?: ProcessBlock; t: Theme }) {
  if (!running && !processes.total) {
    return null
  }

  const counts = [
    running ? `${running} live agents` : '',
    processes.total ? (processes.running ? `${processes.running} procs` : `${processes.total} done`) : ''
  ]
    .filter(Boolean)
    .join(' · ')

  const summary = `▸ ${counts}`
  const hints = ' · Ctrl+T expand · Ctrl+R restore'
  const activityWidth = cols - stringWidth(summary + hints) - 3
  const firstDetail = rows[0]?.detail ?? processes.rows[0]?.detail ?? ''
  const activity = firstDetail && activityWidth >= 12 ? ` · ${compactPreview(firstDetail, activityWidth)}` : ''

  return (
    <Box
      backgroundColor={mix(t.color.statusBg, t.color.shellDollar, 0.12)}
      flexDirection="column"
      flexShrink={0}
      width={cols}
    >
      {collapsed ? (
        <Text bold color={t.color.accent} wrap="truncate-end">
          {summary + activity + hints}
        </Text>
      ) : null}
      {!collapsed && running ? (
        <Text bold color={t.color.accent} wrap="truncate-end">
          {`▾ ${running} live agents${hidden ? ` · +${hidden} more` : ''} · Ctrl+T expand · Ctrl+R collapse`}
        </Text>
      ) : null}
      {!collapsed &&
        rows.map(row => (
          <Box flexDirection="column" key={row.key}>
            <Text wrap="truncate-end">
              <Text color={statusGlyph(row.status, t).color}>{statusGlyph(row.status, t).glyph} </Text>
              <Text color={t.color.text}>{compactPreview(row.goal, Math.max(8, cols - 18))}</Text>
              <Text color={t.color.muted}> {row.elapsedSeconds == null ? '' : fmtDuration(row.elapsedSeconds)}</Text>
            </Text>
            <Text color={t.color.muted} wrap="truncate-end">{`  ↳ ${compactPreview(row.detail, cols - 4)}`}</Text>
          </Box>
        ))}
      {!collapsed && processes.total ? (
        <Text bold color={t.color.accent} wrap="truncate-end">
          {`▾ Processes · ${processSummary(processes)}${processes.hidden ? ` · +${processes.hidden} more` : ''}${
            running ? '' : ' · Ctrl+T expand · Ctrl+R collapse'
          }`}
        </Text>
      ) : null}
      {!collapsed && processes.rows.map(row => <ProcessRowLine cols={cols} key={row.id} row={row} t={t} />)}
    </Box>
  )
}

export function LiveAgentsPanel({ cols }: { cols: number }) {
  const { theme } = useStore($uiState)
  const collapsed = useStore($agentDockCollapsed)
  const { stdout } = useStdout()
  const subagents = useAgentRoster()
  const live = subagents.some(s => s.status === 'running' || s.status === 'queued')
  const [now, setNow] = useState(Date.now)
  const processRows = useProcessRows(now)
  // Process rows carry `Ns ago` / elapsed text, so the clock ticks while any are shown.
  const ticking = live || processRows.length > 0
  useEffect(() => {
    if (!ticking) {
      return
    }

    const timer = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(timer)
  }, [ticking])

  const budget = splitDockBudget(dockRowLimit(stdout?.rows ?? 24), subagents.length, processRows.length)

  return (
    <AgentsPanelView
      collapsed={collapsed}
      cols={cols}
      {...buildAgentRows(subagents, [], now, budget.agents)}
      processes={buildProcessBlock(processRows, budget.processes)}
      t={theme}
    />
  )
}
