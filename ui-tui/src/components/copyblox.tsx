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
  language: string
  rawContent: string
  theme: Theme
  cols: number
}

export const CopyBlox = memo(function CopyBlox({ children, closed, language, rawContent, theme, cols }: CopyBloxProps) {
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
      <Box key={codeIdx++}>
        <Text color={t.border}>{'\u2502'}</Text>
        {child}
        <Text color={t.border}>{'\u2502'}</Text>
      </Box>
    )
  })

  // ── Bottom border ──────────────────────────────────────────────────
  const bottomFillWidth = Math.max(1, cols - 2) // └ + ┘

  // ── Header ─────────────────────────────────────────────────────────
  // All headers are single-row: ┌─label─spacer─icon─┐
  // The 3-cell icon (⧉⧉⧉) is wide enough to be clearly visible.

  const labelW = w(label)
  const iconW = 3 // 3 cells wide

  if (isStreaming) {
    // ── Single-row streaming header ──────────────────────────────────
    const fixed = 1 + 1 + labelW + iconW + 1 // ┌─label─icon─┐
    const spacerW = Math.max(0, cols - fixed)

    return (
      <Box flexDirection="column">
        <NoSelect>
        <Box>
          <Text color={t.border}>{'\u250c'}</Text>
          <Text color={t.border}>{'\u2500'}</Text>
          <Text color={t.accent}>{label}</Text>
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
      <Box flexDirection="column">
        <NoSelect onClick={(e: ClickEvent) => { e.stopImmediatePropagation(); doCopy() }}>
        <Box>
          <Text color={t.border}>{'\u250c'}</Text>
          <Text color={t.border}>{'\u2500'}</Text>
          <Text color={t.accent}>{label}</Text>
          {spacerW > 0 ? <Text color={t.border}>{fill(spacerW)}</Text> : null}
          <Text color={t.border}>{' '}</Text>
          <Text color={t.accent}>{FEEDBACK[copyState]}</Text>
          <Text color={t.border}>{' '}</Text>
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
    <Box flexDirection="column">
      <NoSelect onClick={(e: ClickEvent) => { e.stopImmediatePropagation(); doCopy() }}>
      <Box>
        <Text color={t.border}>{'\u250c'}</Text>
        <Text color={t.border}>{'\u2500'}</Text>
        <Text color={t.accent}>{label}</Text>
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
