import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { $turnState } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import type { SubagentProgress } from '../types.js'

import { SubagentAccordion } from './thinking.js'

export interface SwarmOverlayModel {
  empty: boolean
  headline: string
  summary: string
  total: number
}

const STATUS_ORDER = ['running', 'completed', 'failed', 'interrupted'] as const

type SwarmStatus = (typeof STATUS_ORDER)[number]

export function buildSwarmOverlayModel(subagents: SubagentProgress[]): SwarmOverlayModel {
  const counts: Record<SwarmStatus, number> = {
    completed: 0,
    failed: 0,
    interrupted: 0,
    running: 0
  }

  for (const item of subagents) {
    const key = (STATUS_ORDER as readonly string[]).includes(item.status) ? (item.status as SwarmStatus) : 'running'
    counts[key] += 1
  }

  const total = subagents.length
  const fragments = STATUS_ORDER.filter(status => counts[status] > 0).map(status => `${counts[status]} ${status}`)

  return {
    empty: total === 0,
    headline: total === 0 ? 'Swarm · idle' : `Swarm · ${total} subagent${total === 1 ? '' : 's'}`,
    summary: fragments.length ? fragments.join(' · ') : 'No active subagents yet',
    total
  }
}

export function SwarmOverlay() {
  const turn = useStore($turnState)
  const ui = useStore($uiState)
  const model = buildSwarmOverlayModel(turn.subagents)

  return (
    <Box flexDirection="column" minWidth={56} paddingX={1} paddingY={1}>
      <Text bold color={ui.theme.color.gold}>
        {model.headline}
      </Text>
      <Text color={ui.theme.color.dim}>{model.summary}</Text>

      {model.empty ? (
        <Box marginTop={1}>
          <Text color={ui.theme.color.dim}>No active subagents yet. Run delegated work, then use /swarm to focus the live board.</Text>
        </Box>
      ) : (
        <Box flexDirection="column" marginTop={1}>
          {turn.subagents.map((item, index) => (
            <SubagentAccordion
              branch={index === turn.subagents.length - 1 ? 'last' : 'mid'}
              expanded
              item={item}
              key={item.id}
              t={ui.theme}
            />
          ))}
        </Box>
      )}

      <Box marginTop={1}>
        <Text color={ui.theme.color.dim}>Use /swarm close to dismiss.</Text>
      </Box>
    </Box>
  )
}
