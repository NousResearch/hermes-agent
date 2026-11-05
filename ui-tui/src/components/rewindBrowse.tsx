import { Box, type Key, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { RewindOverlay } from '../app/interfaces.js'
import { $rewindFocus, $rewindScrollRestore } from '../app/rewindFocusStore.js'
import { useT } from '../i18n/useT.js'
import { clampOffset, ensureVisible, fitBrowseHint, messageRange, pagerLines, pagerMessages } from '../lib/transcriptPager.js'
import type { Theme } from '../theme.js'

// Rows the native pager leaves for the hint, the composer chrome and the status rule.
const PAGER_RESERVED_ROWS = 8

/**
 * Codex-style transcript browsing (Esc Esc on an idle, empty composer, or `/rewind`). The latest
 * prompt starts selected; ←/h and →/l move to older/newer prompts (clamped), ↵ puts the selected
 * prompt into the composer as a pending edit, `f` forks from before it, Esc goes back (scroll
 * restored), `q` closes, and any other printable key closes and is typed into the composer.
 *
 * Default (alt-screen) mode browses the real transcript: the selected prompt is scrolled into view
 * and drawn in reverse video there, ↑↓/jk scroll it (the global handler owns that scroll), and this
 * component is the one-line hint. `--tui-native` mode cannot scroll its transcript (it lives in the
 * terminal's scrollback), so this component draws a pager of the session's prompts and replies
 * above the hint instead, with its own ↑↓/jk and PgUp/PgDn scroll.
 */
export function RewindBrowse({
  cols = 80,
  native = false,
  onBack,
  onClose,
  onEdit,
  onFork,
  onType,
  req,
  t
}: RewindBrowseProps) {
  const T = useT()
  const { stdout } = useStdout()
  const [idx, setIdx] = useState(req.targets.length - 1)
  const target = req.targets[idx]

  useEffect(() => {
    $rewindFocus.set(target ? { text: target.text, turn: target.turn } : null)
  }, [target])

  useEffect(() => () => $rewindFocus.set(null), [])

  const rows = Math.max(4, (stdout?.rows ?? 24) - PAGER_RESERVED_ROWS)
  const messages = useMemo(() => pagerMessages(req.messages ?? []), [req.messages])
  const lines = useMemo(() => (native ? pagerLines(messages, Math.max(20, cols - 4)) : []), [cols, messages, native])

  // The selected prompt is the `turn`-th user message of the pager (same rule as the transcript focus).
  const selectedMsg = useMemo(() => {
    const users = messages.flatMap((msg, i) => (msg.role === 'user' ? [i] : []))
    const byTurn = target ? users[target.turn - 1] : undefined

    return byTurn !== undefined && messages[byTurn]!.text.trim() === target?.text.trim()
      ? byTurn
      : (users.findLast(i => messages[i]!.text.trim() === target?.text.trim()) ?? -1)
  }, [messages, target])

  const [offset, setOffset] = useState(Number.MAX_SAFE_INTEGER)

  useEffect(() => {
    setOffset(prev => ensureVisible(prev, messageRange(lines, selectedMsg), rows, lines.length))
  }, [lines, rows, selectedMsg])

  useInput((ch, key) => {
    // A lone Esc can arrive meta-flagged, so it is read before the chord guard.
    if (key.escape) {
      $rewindScrollRestore.set(true)

      return onBack()
    }

    // Ctrl+C closes from the global handler; other chords are not browse keys.
    if (key.ctrl || key.meta) {
      return
    }

    const action = browseAction(ch, key)

    if (action === 'older' || action === 'newer') {
      return setIdx(i => Math.max(0, Math.min(req.targets.length - 1, i + (action === 'older' ? -1 : 1))))
    }

    if (typeof action === 'number') {
      // Default mode: the transcript scroll belongs to the global handler.
      return (
        native && setOffset(prev => clampOffset(prev + action * (Math.abs(action) > 1 ? rows : 1), lines.length, rows))
      )
    }

    const run = {
      close: onClose,
      edit: () => target && onEdit(target),
      fork: () => target && onFork(target),
      type: () => onType(ch)
    }

    action && run[action]()
  })

  // The overlay box pads one column each side.
  const hint = (
    <Text color={t.color.muted} wrap="truncate-end">
      {fitBrowseHint(T.prompt.rewind.browse, Math.max(1, cols - 2))}
    </Text>
  )

  if (!native) {
    return hint
  }

  const top = clampOffset(offset, lines.length, rows)

  return (
    <Box flexDirection="column">
      {lines.slice(top, top + rows).map((line, i) => (
        <Text
          color={line.user ? t.color.text : t.color.muted}
          inverse={line.msg === selectedMsg && line.msg >= 0}
          key={top + i}
          wrap="truncate-end"
        >
          {line.text || ' '}
        </Text>
      ))}
      {hint}
    </Box>
  )
}

type BrowseAction = 'close' | 'edit' | 'fork' | 'newer' | 'older' | 'type' | number | null

const PRINTABLE_EXCLUDED: (keyof Key)[] = ['backspace', 'delete', 'return', 'tab', 'wheelDown', 'wheelUp']

/** A browse key's meaning; numbers scroll (±1 a row, ±2 a page). */
function browseAction(ch: string, key: Key): BrowseAction {
  if (key.leftArrow || ch === 'h') {
    return 'older'
  }

  if (key.rightArrow || ch === 'l') {
    return 'newer'
  }

  if (key.return) {
    return 'edit'
  }

  const scroll =
    key.upArrow || ch === 'k' ? -1 : key.downArrow || ch === 'j' ? 1 : key.pageUp ? -2 : key.pageDown ? 2 : 0

  if (scroll) {
    return scroll
  }

  const named: Record<string, BrowseAction> = { f: 'fork', q: 'close' }

  return named[ch] ?? (ch && !PRINTABLE_EXCLUDED.some(name => key[name]) ? 'type' : null)
}

interface RewindBrowseProps {
  cols?: number
  native?: boolean
  onBack: () => void
  onClose: () => void
  onEdit: (target: RewindOverlay['targets'][number]) => void
  onFork: (target: RewindOverlay['targets'][number]) => void
  onType: (text: string) => void
  req: RewindOverlay
  t: Theme
}
