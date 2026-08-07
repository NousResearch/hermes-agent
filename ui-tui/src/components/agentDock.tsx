import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { $turnState } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import type { SubagentNode, SubagentProgress } from '../types.js'
import { buildSubagentTree, fmtDuration, fmtTokens, topLevelSubagents } from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'

// ── Agent Dock ────────────────────────────────────────────────────────────────
// Always-visible (non-modal) strip of live subagent cards directly under the
// composer — Claude-Code-style. Up to 4 top-level agents (matches the
// delegation.maxConcurrentChildren cap). Each card shows status, goal,
// runtime, token depth and tool depth. Click a card to open that agent's full
// detail in the existing agentsOverlay. ESC returns to the compact dock.
// The dock auto-hides 15s after the last agent finishes.

const MAX_COLUMNS = 4
const AUTO_CLOSE_MS = 15_000

const STATUS_GLYPH: Record<string, string> = {
  running: '●',
  queued: '○',
  completed: '✓',
  interrupted: '■',
  failed: '✗',
  timeout: '⌛',
  error: '⚠'
}

const statusGlyph = (status: string): string => STATUS_GLYPH[status] ?? STATUS_GLYPH.error

function isRunning(item: Pick<SubagentProgress, 'status'>): boolean {
  return item.status === 'running' || item.status === 'queued'
}

function Card({
  active,
  index,
  node,
  onOpen,
  t,
  width
}: {
  active: boolean
  index: number
  node: SubagentNode
  onOpen: (index: number) => void
  t: Theme
  width: number
}) {
  const { item, aggregate: agg } = node
  const glyph = statusGlyph(item.status)
  const color =
    item.status === 'running'
      ? t.color.accent
      : item.status === 'completed'
        ? t.color.statusGood
        : item.status === 'error' || item.status === 'failed'
          ? t.color.error
          : t.color.muted

  const goal = compactPreview(item.goal || 'subagent', width - 4)
  const tokens = fmtTokens((item.inputTokens ?? 0) + (item.outputTokens ?? 0))
  const elapsed = item.durationSeconds != null ? fmtDuration(item.durationSeconds) : ''
  const depth = `d${agg.maxDepthFromHere}·${agg.totalTools}t`

  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="round"
      borderColor={active ? t.color.accent : t.color.border}
      paddingX={1}
      onClick={() => onOpen(index)}
    >
      <Text bold={active} color={active ? t.color.accent : t.color.text} wrap="truncate-end">
        <Text color={color}>{glyph} </Text>
        {goal}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {elapsed ? `${elapsed} · ` : ''}
        {tokens} tok · {depth}
      </Text>
    </Box>
  )
}

export function AgentDock() {
  const ui = useStore($uiState)
  const turn = useStore($turnState)
  const { stdout } = useStdout()
  const cols = (stdout?.columns ?? 80) - 2

  // Live subagents come from the same store the agentsOverlay reads.
  const subagents = turn.subagents ?? []

  const tops = useMemo(() => {
    const tree = buildSubagentTree(subagents)
    return topLevelSubagents(subagents).slice(0, MAX_COLUMNS).map(id => {
      const found = tree.find(n => n.item.id === id.id)
      return found ?? ({ item: id, aggregate: { totalTools: id.toolCount ?? 0, maxDepthFromHere: id.depth, inputTokens: id.inputTokens ?? 0, outputTokens: id.outputTokens ?? 0 } as never, children: [] } as SubagentNode)
    })
  }, [subagents])

  const anyRunning = useMemo(() => tops.some(n => isRunning(n.item)), [tops])

  const [visible, setVisible] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const [autoCloseAt, setAutoCloseAt] = useState<number | null>(null)

  // Visibility + auto-close timer.
  useEffect(() => {
    if (tops.length === 0) {
      setVisible(false)
      setAutoCloseAt(null)
      return
    }

    if (anyRunning) {
      setVisible(true)
      setAutoCloseAt(null)
      return
    }

    // No agents running: start the 15s auto-close window (once).
    if (autoCloseAt == null) {
      setAutoCloseAt(Date.now() + AUTO_CLOSE_MS)
    }
    setVisible(true)
  }, [tops.length, anyRunning, autoCloseAt])

  useEffect(() => {
    if (autoCloseAt == null) {
      return
    }

    const tick = setInterval(() => {
      if (Date.now() >= autoCloseAt) {
        setVisible(false)
        setAutoCloseAt(null)
        setSelected(null)
      }
    }, 500)

    return () => clearInterval(tick)
  }, [autoCloseAt])

  // ESC returns to the compact dock (closes any open detail overlay).
  useInput((_input, key) => {
    if (key.escape && selected != null) {
      setSelected(null)
      patchOverlayState({ agents: false, agentsInitialHistoryIndex: 0 })
    }
  })

  if (!visible || tops.length === 0) {
    return null
  }

  const cardWidth = Math.max(16, Math.floor((cols - (tops.length - 1) * 1) / tops.length))

  const openAgent = (index: number) => {
    setSelected(index)
    patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
  }

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} paddingX={1}>
      <Text color={ui.theme.color.muted} dim>
        {anyRunning ? '⚡ agents live' : '✓ agents done · closing…'}
        {autoCloseAt != null ? ` (${Math.ceil((autoCloseAt - Date.now()) / 1000)}s)` : ''}
      </Text>
      <Box flexDirection="row" gap={1}>
        {tops.map((node, i) => (
          <Card
            active={selected === i}
            index={i}
            key={node.item.id}
            node={node}
            onOpen={openAgent}
            t={ui.theme}
            width={cardWidth}
          />
        ))}
      </Box>
    </Box>
  )
}
