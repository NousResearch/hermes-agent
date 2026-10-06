import { stringWidth, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { type GoalLine, useGoalLine } from '../app/goalStatus.js'
import { $uiState } from '../app/uiStore.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

/** One `⊙ goal · 3/20 turns · <title>` row above the live-work dock while a /goal is standing. */
export function GoalBarView({ cols, line, t }: { cols: number; line: GoalLine | null; t: Theme }) {
  if (!line) {
    return null
  }

  const head = `${line.glyph} ${line.label} · ${line.detail} · `

  return (
    <Text wrap="truncate-end">
      <Text color={line.glyph === '⊙' ? t.color.accent : t.color.warn}>{`${line.glyph} ${line.label}`}</Text>
      <Text color={t.color.muted}>{` · ${line.detail} · `}</Text>
      <Text color={t.color.text}>{compactPreview(line.title, Math.max(8, cols - stringWidth(head)))}</Text>
    </Text>
  )
}

export function GoalBar({ cols }: { cols: number }) {
  const { theme } = useStore($uiState)

  return <GoalBarView cols={cols} line={useGoalLine()} t={theme} />
}
