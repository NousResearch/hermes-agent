import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { useT } from '../i18n/useT.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { relativeSessionAge, selectedSessionRowStyle } from './activeSessionSwitcher.js'
import { windowOffset } from './overlayControls.js'
import { TextInput } from './textInput.js'

// "Hesitation" debounce: only search once typing pauses, so three quick
// keystrokes cost one RPC, not three.
const DEBOUNCE_MS = 300
const RESULT_LIMIT = 15
// Two lines per result, so a shorter window than the switcher's 12 keeps the
// panel inside a normal viewport once snippets are drawn.
const VISIBLE = 8
const SNIPPET_MAX = 160

interface SessionSearchRow {
  id: string
  preview?: null | string
  snippet?: null | string
  source?: null | string
  started_at?: number
  title?: null | string
}

interface SessionSearchResponse {
  results?: SessionSearchRow[]
}

// Preferred payload keys, in order: tool turns store the human-readable text
// under one of these while the surrounding JSON is transport scaffolding.
const SNIPPET_KEYS = ['output', 'content', 'text', 'snippet'] as const

const capSnippet = (text: string) => (text.length > SNIPPET_MAX ? `${text.slice(0, SNIPPET_MAX - 1)}…` : text)

const firstPayloadString = (node: unknown): string | null => {
  if (Array.isArray(node)) {
    for (const item of node) {
      const found = firstPayloadString(item)

      if (found !== null) {
        return found
      }
    }

    return null
  }

  if (node && typeof node === 'object') {
    const record = node as Record<string, unknown>

    for (const key of SNIPPET_KEYS) {
      const value = record[key]

      if (typeof value === 'string' && value.trim()) {
        return value
      }
    }

    for (const value of Object.values(record)) {
      const found = firstPayloadString(value)

      if (found !== null) {
        return found
      }
    }
  }

  return null
}

/**
 * Collapse a raw search snippet to one readable line. Snippets come from
 * stored transcript text, so tool turns arrive as raw JSON like
 * {"output": "=== GITHUB …"} — when that parses, surface the payload string
 * instead of the JSON scaffolding; otherwise keep the collapsed raw text.
 */
export const cleanSnippet = (raw: string): string => {
  const collapsed = raw.replace(/\s+/g, ' ').trim()

  if (collapsed.startsWith('{') || collapsed.startsWith('[')) {
    try {
      const payload = firstPayloadString(JSON.parse(collapsed))

      if (payload !== null) {
        return capSnippet(payload.replace(/\s+/g, ' ').trim())
      }
    } catch {
      // Broken JSON is just transcript text that happens to start with a brace.
    }
  }

  return capSnippet(collapsed)
}

export function SessionSearchOverlay({ gw, initialQuery, maxWidth, onCancel, onResume, t }: SessionSearchOverlayProps) {
  const tr = useT()
  const T = tr.slashCmd.session.search
  const C = tr.pickers.common
  const { stdout } = useStdout()

  const [query, setQuery] = useState(initialQuery ?? '')
  const [results, setResults] = useState<SessionSearchRow[]>([])
  const [searching, setSearching] = useState(false)
  // Query whose search last completed — "no results" is only honest for the
  // text currently in the box, not for a query still waiting on the debounce.
  const [completedQuery, setCompletedQuery] = useState('')
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)

  // Monotonic request seq: a response landing after a newer request was issued
  // is stale and must not overwrite the newer result set. In-flight RPCs
  // cannot be aborted — their results are simply dropped on arrival.
  const seqRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // A late TextInput key-burst flush can deliver onChange AFTER Esc closed
  // the overlay; the alive flag keeps that ghost from arming a search.
  const aliveRef = useRef(true)

  const search = useCallback(
    (value: string) => {
      const seq = ++seqRef.current

      setSearching(true)
      setErr('')

      gw.request<SessionSearchResponse>('session.search', { limit: RESULT_LIMIT, query: value })
        .then(r => {
          if (!aliveRef.current || seqRef.current !== seq) {
            return
          }

          setSearching(false)
          setCompletedQuery(value)
          setResults(r?.results ?? [])
          setSel(0)
        })
        .catch((e: unknown) => {
          if (!aliveRef.current || seqRef.current !== seq) {
            return
          }

          setSearching(false)
          setResults([])
          setErr(rpcErrorMessage(e))
        })
    },
    [gw]
  )

  // `/search <text>` prefills the box and fires once immediately — the
  // debounce guards keystrokes, not the initial fill.
  useEffect(() => {
    if (initialQuery) {
      search(initialQuery)
    }

    // Cleanup cancels the pending debounce timer; the alive flag (checked on
    // every response) orphans anything in flight, so a closed overlay never
    // mutates stale state.
    return () => {
      aliveRef.current = false

      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [initialQuery, search])

  const onQueryChange = (value: string) => {
    setQuery(value)
    setSel(0)

    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }

    // Clearing the box returns to the idle hint; the seq bump also drops
    // whatever in-flight search the cleared text had spawned.
    if (!value.trim()) {
      seqRef.current++
      setSearching(false)
      setResults([])
      setErr('')

      return
    }

    timerRef.current = setTimeout(() => {
      timerRef.current = null

      if (aliveRef.current) {
        search(value)
      }
    }, DEBOUNCE_MS)
  }

  useInput((_ch, key) => {
    if (key.escape) {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }

      return onCancel()
    }

    if (key.upArrow) {
      return setSel(s => Math.max(0, s - 1))
    }

    if (key.downArrow) {
      return setSel(s => Math.min(Math.max(0, results.length - 1), s + 1))
    }

    if (key.return) {
      const row = results[sel]

      if (row) {
        onResume(row.id)
      }
    }
  })

  const width = maxWidth ?? stdout?.columns ?? 80
  const snippetWidth = Math.max(20, width - 6)
  const trimmed = query.trim()
  const offset = windowOffset(results.length, sel, VISIBLE)
  const visibleRows = results.slice(offset, offset + VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <TextInput
        color={t.color.text}
        columns={Math.max(20, width - 2)}
        ignoreVerticalArrows
        onChange={onQueryChange}
        placeholder={T.placeholder}
        value={query}
      />

      {!trimmed && <Text color={t.color.muted}>{T.noQueryHint}</Text>}
      {/* Pending covers both the in-flight RPC and the debounce wait before it. */}
      {trimmed && (searching || completedQuery !== trimmed) && (
        <Text color={t.color.muted}>{T.searching}</Text>
      )}
      {trimmed && !searching && completedQuery === trimmed && !err && !results.length && (
        <Text color={t.color.muted}>{T.noResults(trimmed)}</Text>
      )}
      {err && <Text color={t.color.label}>{err}</Text>}

      {offset > 0 && <Text color={t.color.muted}>{C.moreAbove(offset)}</Text>}

      {visibleRows.map((row, i) => {
        const selected = offset + i === sel
        const selectedStyle = selected ? selectedSessionRowStyle(t) : null
        const rowTextColor = selectedStyle?.color
        const snippet = cleanSnippet(row.snippet ?? '')

        const head = [row.title || row.preview || '', relativeSessionAge(row.started_at), row.source ?? '']
          .filter(Boolean)
          .join(' · ')

        return (
          <Box backgroundColor={selectedStyle?.backgroundColor} flexDirection="column" key={row.id}>
            <Text bold={selected} color={rowTextColor ?? t.color.text} wrap="truncate-end">
              {selected ? '▸ ' : '  '}
              {head}
            </Text>
            {snippet && (
              <Text color={rowTextColor ?? t.color.muted} wrap="truncate-end">
                {'  '}
                {snippet.length > snippetWidth ? `${snippet.slice(0, snippetWidth - 1)}…` : snippet}
              </Text>
            )}
          </Box>
        )
      })}

      {offset + VISIBLE < results.length && (
        <Text color={t.color.muted}>{C.moreBelow(results.length - offset - VISIBLE)}</Text>
      )}

      <Text color={t.color.muted} marginTop={1}>
        {T.footerHint}
      </Text>
    </Box>
  )
}

interface SessionSearchOverlayProps {
  gw: GatewayClient
  initialQuery?: string
  maxWidth?: number
  onCancel: () => void
  onResume: (id: string) => void
  t: Theme
}
