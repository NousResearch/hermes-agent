import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useAgentRoster } from '../app/agentRoster.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiState } from '../app/uiStore.js'
import { type AgentRows, buildAgentRows, dockRowLimit } from '../lib/agentRows.js'
import { statusGlyph } from '../lib/subagentGlyph.js'
import { fmtDuration } from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export function AgentsPanelView({ cols, hidden, rows, running, t }: AgentRows & { cols: number; t: Theme }) {
  if (!running) {return null}

  return (
    <Box flexDirection="column" flexShrink={0} width={cols}>
      <Text bold color={t.color.accent} onClick={() => patchOverlayState({ agents: true })} wrap="truncate-end">
        {`▾ ${running} live agents${hidden ? ` · +${hidden} more` : ''} · Ctrl+T expand`}
      </Text>
      {rows.map(row => (
        <Box flexDirection="column" key={row.key} onClick={() => patchOverlayState({ agents: true })}>
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
  const { stdout } = useStdout()
  const subagents = useAgentRoster()
  const live = subagents.some(s => s.status === 'running' || s.status === 'queued')
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!live) {return}
    const timer = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(timer)
  }, [live])

  return (
    <AgentsPanelView cols={cols} {...buildAgentRows(subagents, [], now, dockRowLimit(stdout?.rows ?? 24))} t={theme} />
  )
}
