import { Box, NoSelect, stringWidth, Text } from '@hermes/ink'
import type { ClickEvent } from '@hermes/ink'
import React, { memo, useCallback, useEffect, useRef, useState } from 'react'

import { copyText } from '../lib/copyText.js'
import type { Theme } from '../theme.js'

// Copy state machine states
type CopyState = 'idle' | 'copying' | 'copied' | 'failed'

// 3-cell copy icon — visible enough to be clearly clickable.
const COPY_ICON = '⧉⧉⧉'

// Feedback labels shown in a compact single-row header.
const FEEDBACK: Record<CopyState, string> = {
  idle: '',
  copying: '…',
  copied: '✓',
  failed: '!'
}

const COPY_FEEDBACK_MS = 1200
const FAIL_FEEDBACK_MS = 1500

interface CopyBloxProps {
  children: React.ReactNode
  closed: boolean
  compact?: boolean
  language: string
  rawContent: string
  theme: Theme
  cols: number
}

const NARROW_CODE_BLOCK_COLS = 20

const truncateToWidth = (value: string, maxWidth: number): string => {
  if (stringWidth(value) <= maxWidth) {
    return value
  }

  const ellipsis = '…'
  const budget = Math.max(0, maxWidth - stringWidth(ellipsis))

  const segments =
    typeof Intl !== 'undefined' && 'Segmenter' in Intl
      ? [...new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(value)].map(({ segment }) => segment)
      : Array.from(value)

  let result = ''

  for (const segment of segments) {
    if (stringWidth(result + segment) > budget) {
      break
    }

    result += segment
  }

  return result + ellipsis
}

export const CopyBlox = memo(function CopyBlox({
  children,
  closed,
  compact = false,
  language,
  rawContent,
  theme,
  cols
}: CopyBloxProps) {
  const [copyState, setCopyState] = useState<CopyState>('idle')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const rawContentRef = useRef(rawContent)
  const busyRef = useRef(false)

  // Always keep ref current so click handler sees latest content
  rawContentRef.current = rawContent

  const doCopy = useCallback(async () => {
    if (busyRef.current) {
      return
    }

    busyRef.current = true

    setCopyState('copying')

    const text = rawContentRef.current
    const result = await copyText(text)

    if (result.success) {
      setCopyState('copied')
    } else {
      setCopyState('failed')
    }

    const ms = result.success ? COPY_FEEDBACK_MS : FAIL_FEEDBACK_MS

    if (timerRef.current) {
      clearTimeout(timerRef.current)
    }

    const timer = setTimeout(() => {
      setCopyState('idle')
      busyRef.current = false
    }, ms)

    timerRef.current = timer
  }, [])

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [])

  const t = theme.color
  const label = language || 'text'
  const isStreaming = !closed
  const isNarrow = compact || cols < NARROW_CODE_BLOCK_COLS

  const w = (s: string) => stringWidth(s)
  const fill = (n: number) => '─'.repeat(Math.max(1, n))

  // ── Code body rows ─────────────────────────────────────────────────
  const codeRows: React.ReactNode[] = []
  let codeIdx = 0
  React.Children.forEach(children, child => {
    if (child == null) {
      return
    }

    codeRows.push(
      <Box key={codeIdx++} width={isNarrow ? Math.max(1, cols - 2) : cols}>
        {!isNarrow && <Text color={t.border}>{'\u2502'}</Text>}
        {child}
        {!isNarrow && <Text color={t.border}>{'\u2502'}</Text>}
      </Box>
    )
  })

  // ── Bottom border ──────────────────────────────────────────────────
  const bottomFillWidth = Math.max(1, cols - 2) // └ + ┘

  // ── Header ─────────────────────────────────────────────────────────
  // All headers are single-row: ┌─label─spacer─icon─┐
  // The 3-cell icon (⧉⧉⧉) is wide enough to be clearly visible.

  const suffixW = isStreaming || copyState !== 'idle' ? 1 : 3
  const displayLabel = truncateToWidth(label, Math.max(1, cols - (isNarrow ? 2 : 3) - suffixW))
  const labelW = w(displayLabel)
  const iconW = 3 // 3 cells wide

  if (isNarrow) {
    const onClick = isStreaming
      ? undefined
      : (e: ClickEvent) => {
          e.stopImmediatePropagation()
          void doCopy()
        }

    return (
      <Box
        borderBottom={false}
        borderColor={t.border}
        borderLeft
        borderRight={false}
        borderStyle="single"
        borderTop={false}
        flexDirection="column"
        paddingLeft={1}
        width={cols}
      >
        <NoSelect onClick={onClick}>
          <Box>
            <Text color={t.accent}>{displayLabel}</Text>
            <Text color={isStreaming ? t.muted : t.accent}>
              {isStreaming ? '\u27f3' : copyState === 'idle' ? COPY_ICON : FEEDBACK[copyState]}
            </Text>
          </Box>
        </NoSelect>

        <Box flexDirection="column">{codeRows}</Box>
      </Box>
    )
  }

  if (isStreaming) {
    // ── Single-row streaming header ──────────────────────────────────
    const fixed = 1 + 1 + labelW + iconW + 1 // ┌─label─icon─┐
    const spacerW = Math.max(0, cols - fixed)

    return (
      <Box flexDirection="column" width={cols}>
        <NoSelect>
          <Box>
            <Text color={t.border}>{'\u250c'}</Text>
            <Text color={t.border}>{'\u2500'}</Text>
            <Text color={t.accent}>{displayLabel}</Text>
            {spacerW > 0 ? <Text color={t.border}>{fill(spacerW)}</Text> : null}
            <Text color={t.muted}>{'\u27f3'}</Text>
            <Text color={t.border}>{'\u2510'}</Text>
          </Box>
        </NoSelect>

        <Box flexDirection="column">{codeRows}</Box>

        <Box>
          <Text color={t.border}>{'\u2514'}</Text>
          <Text color={t.border}>{fill(bottomFillWidth)}</Text>
          <Text color={t.border}>{'\u2518'}</Text>
        </Box>
      </Box>
    )
  }

  if (copyState !== 'idle') {
    // ── Single-row feedback header ───────────────────────────────────
    const fixed = 1 + 1 + labelW + 4 + 1 // ┌─label─" … "─┐
    const spacerW = Math.max(0, cols - fixed)

    return (
      <Box flexDirection="column" width={cols}>
        <NoSelect
          onClick={(e: ClickEvent) => {
            e.stopImmediatePropagation()
            doCopy()
          }}
        >
          <Box>
            <Text color={t.border}>{'\u250c'}</Text>
            <Text color={t.border}>{'\u2500'}</Text>
            <Text color={t.accent}>{displayLabel}</Text>
            {spacerW > 0 ? <Text color={t.border}>{fill(spacerW)}</Text> : null}
            <Text color={t.border}> </Text>
            <Text color={t.accent}>{FEEDBACK[copyState]}</Text>
            <Text color={t.border}> </Text>
            <Text color={t.border}>{'\u2510'}</Text>
          </Box>
        </NoSelect>

        <Box flexDirection="column">{codeRows}</Box>

        <Box>
          <Text color={t.border}>{'\u2514'}</Text>
          <Text color={t.border}>{fill(bottomFillWidth)}</Text>
          <Text color={t.border}>{'\u2518'}</Text>
        </Box>
      </Box>
    )
  }

  // ── Single-row idle header with copy icon ──────────────────────────
  const fixed = 1 + 1 + labelW + iconW + 1 // ┌─label─icon─┐
  const spacerW = Math.max(0, cols - fixed)

  return (
    <Box flexDirection="column" width={cols}>
      <NoSelect
        onClick={(e: ClickEvent) => {
          e.stopImmediatePropagation()
          doCopy()
        }}
      >
        <Box>
          <Text color={t.border}>{'\u250c'}</Text>
          <Text color={t.border}>{'\u2500'}</Text>
          <Text color={t.accent}>{displayLabel}</Text>
          {spacerW > 0 ? <Text color={t.border}>{fill(spacerW)}</Text> : null}
          <Text color={t.accent}>{COPY_ICON}</Text>
          <Text color={t.border}>{'\u2510'}</Text>
        </Box>
      </NoSelect>

      {/* Code body with side borders — not inside click target */}
      <Box flexDirection="column">{codeRows}</Box>

      {/* Bottom border */}
      <Box>
        <Text color={t.border}>{'\u2514'}</Text>
        <Text color={t.border}>{fill(bottomFillWidth)}</Text>
        <Text color={t.border}>{'\u2518'}</Text>
      </Box>
    </Box>
  )
})
