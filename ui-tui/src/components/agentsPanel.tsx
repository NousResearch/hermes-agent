import { Box, stringWidth, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $agentDockCollapsed, useAgentRoster } from '../app/agentRoster.js'
import { $uiState } from '../app/uiStore.js'
import { type AgentRows, buildAgentRows, dockRowLimit } from '../lib/agentRows.js'
import { mix } from '../lib/color.js'
import { statusGlyph } from '../lib/subagentGlyph.js'
import { fmtDuration } from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export function AgentsPanelView({
  collapsed = false,
  cols,
  hidden,
  rows,
  running,
  t
}: AgentRows & { collapsed?: boolean; cols: number; t: Theme }) {
  if (!running) {
    return null
  }

  const summary = `▸ ${running} live agents`
  const hints = ' · Ctrl+T expand · F7 restore'
  const activityWidth = cols - stringWidth(summary + hints) - 3
  const activity = rows[0]?.detail && activityWidth >= 12 ? ` · ${compactPreview(rows[0].detail, activityWidth)}` : ''

  return (
    <Box
      backgroundColor={mix(t.color.statusBg, t.color.shellDollar, 0.12)}
      flexDirection="column"
      flexShrink={0}
      width={cols}
    >
      <Text bold color={t.color.accent} wrap="truncate-end">
        {collapsed
          ? summary + activity + hints
          : `▾ ${running} live agents${hidden ? ` · +${hidden} more` : ''} · Ctrl+T expand · F7 collapse`}
      </Text>
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
  useEffect(() => {
    if (!live) {
      return
    }

    const timer = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(timer)
  }, [live])

  return (
    <AgentsPanelView
      collapsed={collapsed}
      cols={cols}
      {...buildAgentRows(subagents, [], now, dockRowLimit(stdout?.rows ?? 24))}
      t={theme}
    />
  )
}
