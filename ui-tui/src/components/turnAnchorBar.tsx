import { Box, NoSelect, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, useEffect, useRef } from 'react'

import { $uiState } from '../app/uiStore.js'
import { userDisplay } from '../domain/messages.js'
import { type TurnAnchor, turnAnchorFromViewport } from '../domain/viewport.js'
import { useViewportSnapshot } from '../lib/viewportStore.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

/** The bar always occupies exactly one row so viewport compensation is ±1. */
export const TURN_ANCHOR_BAR_HEIGHT = 1

/** Scroll compensation when the bar's presence changes: the transcript viewport
 *  shrinks by one row on appear (content must move up: -1) and regrows on
 *  disappear (+1). 0 while the presence is unchanged. */
export const anchorCompensation = (wasPresent: boolean, isPresent: boolean): number => {
  if (wasPresent === isPresent) {
    return 0
  }

  return isPresent ? -1 : 1
}

const GLYPH = '↳'
const HISTORY_HINT = '] = live'
const GLYPH_COLS = 2 // glyph + one space
const HINT_COLS = 3 // ' · ' separator + hint

interface TurnAnchorBarProps {
  anchor: TurnAnchor | null
  cols: number
  t: Theme
}

// First line of the pinned user message, with a continuation count when the
// original prompt spanned more lines (`+N` = lines after the first).
const anchorFirstLine = (text: string): { count: number; first: string } => {
  const display = userDisplay(text.trim())
  const lines = display.split('\n')
  const first = (lines[0] ?? '').trim()
  const count = Math.max(0, lines.length - 1)

  return { count, first }
}

// Pinned turn header (approach A): sits ABOVE the transcript ScrollBox, never
// inside it. Live mode = the streaming turn's opening prompt; history mode =
// the user message that opens the turn currently on screen. One row, always.
export const TurnAnchorBar = memo(function TurnAnchorBar({ anchor, cols, t }: TurnAnchorBarProps) {
  if (!anchor || cols <= 0) {
    return null
  }

  const { count, first } = anchorFirstLine(anchor.msg.text)
  const hint = anchor.mode === 'history' ? HISTORY_HINT : ''
  const hintCols = hint ? hint.length + HINT_COLS : 0
  // cols - 1 keeps the row inside the terminal width after the left gutter.
  const textBudget = Math.max(4, cols - 1 - GLYPH_COLS - hintCols)

  return (
    <NoSelect>
      <Box height={TURN_ANCHOR_BAR_HEIGHT} paddingLeft={1} width={cols}>
        <Text bold color={t.color.accent}>
          {GLYPH}{' '}
        </Text>
        <Text color={t.color.text} wrap="truncate-end">
          {first.slice(0, textBudget)}
          {count > 0 ? ` +${count}` : ''}
        </Text>
        {hint ? (
          <Text color={t.color.muted}>
            {' · '}
            {hint}
          </Text>
        ) : null}
      </Box>
    </NoSelect>
  )
})

interface PinnedTurnAnchorProps {
  cols: number
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  scrollRef: { current: null | { adjustScrollTop: (dy: number) => void } }
  t: Theme
}

/** Mounted above the transcript ScrollBox (approach A). Owns the anchor
 *  computation (viewport snapshot + streaming flag) and the one-row scroll
 *  compensation on bar appear/disappear. `display.turn_anchor` gates it. */
export const PinnedTurnAnchor = memo(function PinnedTurnAnchor({
  cols,
  messages,
  offsets,
  scrollRef,
  t
}: PinnedTurnAnchorProps) {
  const ui = useStore($uiState)
  const { atBottom, bottom, top } = useViewportSnapshot(scrollRef as never)

  const anchor =
    ui.turnAnchor === 'pinned'
      ? turnAnchorFromViewport(messages, offsets, top, bottom, atBottom, ui.streaming)
      : null

  const presentRef = useRef(anchor !== null)

  useEffect(() => {
    const present = anchor !== null
    const dy = anchorCompensation(presentRef.current, present)

    if (dy !== 0) {
      presentRef.current = present
      scrollRef.current?.adjustScrollTop(dy)
    }
  }, [anchor, scrollRef])

  return <TurnAnchorBar anchor={anchor} cols={cols} t={t} />
})
