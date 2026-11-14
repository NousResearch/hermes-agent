import { Box, type Key, type ScrollBoxHandle, Text, useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type RefObject, useEffect, useMemo, useState } from 'react'

import type { FindOverlay } from '../app/interfaces.js'
import { $findFocus, $findScrollRestore, findSnippet, findTranscriptHits } from '../app/transcriptFindStore.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { FocusRowScroller } from './rewindFocus.js'

const TYPING_EXCLUDED: (keyof Key)[] = ['return', 'tab', 'wheelDown', 'wheelUp', 'pageUp', 'pageDown']

/**
 * `/find`: incremental search over the transcript. Typing narrows the query; the newest match is
 * selected, scrolled into view and marked. ↑/↓ step to older/newer matches, ↵ closes and stays
 * there, Esc closes and returns to where the transcript was. PgUp/PgDn and the wheel still scroll.
 */
export function TranscriptFind({ cols = 80, onClose, req, t }: TranscriptFindProps) {
  const T = useT().prompt.find
  const [query, setQuery] = useState(req.query)
  const rows = useMemo(() => req.getRows(), [req])
  const hits = useMemo(() => findTranscriptHits(rows, query), [rows, query])
  const [cur, setCur] = useState(hits.length - 1)

  useEffect(() => setCur(hits.length - 1), [hits])

  const row = hits[cur] ?? -1

  useEffect(() => $findFocus.set(row), [row])
  useEffect(() => () => $findFocus.set(null), [])

  useInput((ch, key) => {
    if (key.escape) {
      $findScrollRestore.set(true)

      return onClose()
    }

    if (key.return) {
      return onClose()
    }

    if (key.upArrow || key.downArrow) {
      return setCur(i => Math.max(0, Math.min(hits.length - 1, i + (key.upArrow ? -1 : 1))))
    }

    if (key.backspace || key.delete) {
      return setQuery(q => q.slice(0, -1))
    }

    if (ch && !key.ctrl && !key.meta && !TYPING_EXCLUDED.some(name => key[name])) {
      setQuery(q => q + ch.replace(/\s+/g, ' '))
    }
  })

  const width = Math.max(10, cols - 4)
  const count = query ? (hits.length ? T.count(String(cur + 1), String(hits.length)) : T.none) : ''
  const hit = row >= 0 ? rows[row] : undefined

  return (
    <Box flexDirection="column">
      {hit && (
        <Text color={t.color.muted} wrap="truncate-end">
          {findSnippet(hit.text, query, width)}
        </Text>
      )}
      <Text wrap="truncate-end">
        <Text color={t.color.accent}>{T.label} </Text>
        <Text>{query}▏</Text>
        <Text color={t.color.muted}>
          {count ? `  ${count}` : ''} · {T.hint}
        </Text>
      </Text>
    </Box>
  )
}

const takeFindRestore = () => {
  const restore = $findScrollRestore.get()

  $findScrollRestore.set(false)

  return restore
}

/** Keeps the `/find` match in view (the transcript side of `TranscriptFind`). */
export function TranscriptFindScroller({ offsets, scrollRef }: TranscriptFindScrollerProps) {
  const focus = useStore($findFocus)

  return (
    <FocusRowScroller
      active={focus !== null}
      offsets={offsets}
      restore={takeFindRestore}
      row={focus ?? -1}
      scrollRef={scrollRef}
    />
  )
}

/** The transcript row `/find` has selected, -1 when none (drawn like the rewind focus). */
export const useFindFocusIndex = () => useStore($findFocus) ?? -1

interface TranscriptFindProps {
  cols?: number
  onClose: () => void
  req: FindOverlay
  t: Theme
}

interface TranscriptFindScrollerProps {
  offsets: ArrayLike<number>
  scrollRef: RefObject<null | ScrollBoxHandle>
}
