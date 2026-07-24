import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo } from 'react'

import { $asyncDelegations } from '../app/delegationStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { toggleAgentsCollapsed, useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { type AgentRow, buildAgentRows } from '../lib/agentRows.js'
import { statusGlyph } from '../lib/subagentGlyph.js'
import { fmtDuration } from '../lib/subagentTree.js'
import type { Theme } from '../theme.js'

const fmtElapsed = (seconds: null | number): string =>
  seconds == null || seconds < 0 ? '' : fmtDuration(seconds)

/** Pure presentational panel — no store access, so it renders in tests as a
 * plain function call (mirrors StatusRule). Not memo-wrapped so it stays
 * directly callable; the connected LiveAgentsPanel below carries the memo. */
export function AgentsPanelView({
  collapsed,
  done,
  onOpenTree,
  onToggle,
  rows,
  running,
  t
}: {
  collapsed: boolean
  done: number
  onOpenTree?: () => void
  onToggle?: () => void
  rows: AgentRow[]
  running: number
  t: Theme
}) {
  if (!rows.length) {
    return null
  }

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box onClick={onToggle}>
        <Text color={t.color.muted}>
          <Text color={t.color.accent}>{collapsed ? '▸ ' : '▾ '}</Text>
          <Text bold color={t.color.text}>
            agents
          </Text>{' '}
          <Text color={t.color.statusFg} dim>
            · {running} running · {done} done
          </Text>
        </Text>
        {onOpenTree && (
          <Text color={t.color.muted} dim onClick={onOpenTree}>
            {'  '}^a tree
          </Text>
        )}
      </Box>

      {!collapsed && (
        <Box flexDirection="column" marginLeft={2}>
          {rows.map((row, i) => {
            const g = statusGlyph(row.status, t)
            const elapsed = fmtElapsed(row.elapsedSeconds)

            return (
              <Text color={t.color.statusFg} key={row.key}>
                <Text color={t.color.muted}>{String(i + 1).padStart(2, ' ')} </Text>
                <Text color={g.color}>{g.glyph} </Text>
                {row.name && <Text color={t.color.text}>{row.name} </Text>}
                {row.goal && <Text color={t.color.statusFg}>{row.goal}</Text>}
                {elapsed && <Text color={t.color.muted} dim>{'  '}{elapsed}</Text>}
                {row.detail && (
                  <Text color={row.resultReady ? t.color.statusGood : t.color.muted} dim>
                    {'  '}
                    {row.detail}
                    {row.resultReady ? ' ⏎' : ''}
                  </Text>
                )}
              </Text>
            )
          })}
        </Box>
      )}
    </Box>
  )
}

/** Store-connected docked panel. Rides above the composer next to the todo
 * panel; renders nothing when there is no live or background agent. */
export const LiveAgentsPanel = memo(function LiveAgentsPanel() {
  const ui = useStore($uiState)
  const subagents = useTurnSelector(state => state.subagents)
  const collapsed = useTurnSelector(state => state.agentsCollapsed)
  const asyncDelegations = useStore($asyncDelegations)

  const { done, rows, running } = buildAgentRows(subagents, asyncDelegations, Date.now())

  return (
    <AgentsPanelView
      collapsed={collapsed}
      done={done}
      onOpenTree={() => patchOverlayState({ agents: true })}
      onToggle={toggleAgentsCollapsed}
      rows={rows}
      running={running}
      t={ui.theme}
    />
  )
})
