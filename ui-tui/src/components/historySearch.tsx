import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { $overlayState, hasPromptOpen } from '../app/overlayStore.js'
import { useT } from '../i18n/useT.js'
import * as inputHistory from '../lib/history.js'
import { historyPreview, searchHistory } from '../lib/historySearch.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 10
const MIN_WIDTH = 40
const MAX_WIDTH = 110

/**
 * Ctrl+R reverse search over the shared `.hermes_history` (earlier sessions and the classic CLI
 * included). Typing filters, Ctrl+R / ↓ steps to an older match, Enter or Tab puts the match in
 * the composer for editing (it is never sent from here), Esc / Ctrl+G cancels.
 */
export function HistorySearch({ initialQuery = '', maxWidth, onAccept, onClose, t }: HistorySearchProps) {
  const [query, setQuery] = useState(initialQuery)
  const [idx, setIdx] = useState(0)
  const T = useT()
  const H = T.pickers.history
  const C = T.pickers.common
  const { stdout } = useStdout()
  const width = clampOverlayWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6)), maxWidth)
  const matches = useMemo(() => searchHistory(inputHistory.load(), query), [query])
  // An agent question or a confirm owns the keyboard while it is up: Enter must not also recall.
  const promptOpen = hasPromptOpen(useStore($overlayState))

  useInput((input, key) => {
    if (promptOpen) {
      return
    }

    if (key.escape || (key.ctrl && input === 'g')) {
      return onClose()
    }

    if (key.return || key.tab) {
      const pick = matches[idx]

      return pick ? onAccept(pick) : onClose()
    }

    if (key.upArrow) {
      return setIdx(i => Math.max(0, i - 1))
    }

    if (key.downArrow || (key.ctrl && input === 'r')) {
      return setIdx(i => Math.min(Math.max(0, matches.length - 1), i + 1))
    }

    if (key.backspace || key.delete) {
      setQuery(q => q.slice(0, -1))

      return setIdx(0)
    }

    if (input && !key.ctrl && !key.meta && [...input].every(ch => ch >= ' ')) {
      setQuery(q => q + input)
      setIdx(0)
    }
  })

  // The prompt renders in the flow under this floating panel: draw nothing until it is answered,
  // staying mounted so the query and selection survive.
  if (promptOpen) {
    return null
  }

  const { items, offset } = windowItems(matches, idx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {H.title}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {query ? C.filter(query) : C.typeToFilter} · {matches.length === 1 ? H.countOne(1) : H.countOther(matches.length)}
      </Text>

      {offset > 0 && <Text color={t.color.muted}>{C.moreAbove(offset)}</Text>}

      {matches.length === 0 ? (
        <Text color={t.color.muted}>{query ? H.noMatches(query) : H.empty}</Text>
      ) : (
        items.map((entry, i) => {
          const at = offset + i === idx

          return (
            <Text color={t.color.text} {...chipRowProps(t, at)} key={`${offset + i}:${entry}`} wrap="truncate-end">
              {at ? '▸ ' : '  '}
              {historyPreview(entry)}
            </Text>
          )
        })
      )}

      {offset + VISIBLE < matches.length && (
        <Text color={t.color.muted}>{C.moreBelow(matches.length - offset - VISIBLE)}</Text>
      )}

      <OverlayHint t={t}>{H.hint}</OverlayHint>
    </Box>
  )
}

interface HistorySearchProps {
  initialQuery?: string
  maxWidth?: number
  onAccept: (entry: string) => void
  onClose: () => void
  t: Theme
}
