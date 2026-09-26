import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 10
const MIN_WIDTH = 44
const MAX_WIDTH = 90

export interface WorkflowTrigger {
  phrase: string
  description: string
}

interface TriggerList {
  triggers: WorkflowTrigger[]
}

/**
 * Interactive workflow-trigger picker overlay. Pulls the trigger list via
 * `trigger.list` (the same configured source the CLI `/trigger` chooser uses),
 * filters as you type, and dispatches the chosen phrase as a prompt. This is the
 * TUI sibling of the classic-CLI curses chooser — the renderer owns the terminal,
 * so it works where the slash worker's piped stdin cannot.
 */
export function TriggerPicker({
  gw,
  maxWidth,
  onClose,
  onPick,
  t
}: TriggerPickerProps) {
  const [list, setList] = useState<WorkflowTrigger[] | null>(null)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    gw.request<TriggerList>('trigger.list')
      .then(r => {
        setList(r?.triggers ?? [])
        setErr('')
      })
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setLoading(false))
  }, [gw])

  const view = useMemo(() => {
    const triggers = list ?? []
    const needle = query.trim().toLowerCase()

    if (!needle) {
      return triggers
    }

    return triggers.filter(
      tr => tr.phrase.toLowerCase().includes(needle) || tr.description.toLowerCase().includes(needle)
    )
  }, [list, query])

  const pick = (phrase: string) => {
    onPick(phrase)
    onClose()
  }

  useInput((input, key) => {
    if (key.escape) {
      return onClose()
    }

    if (key.upArrow || input.toLowerCase() === 'k') {
      return setIdx(i => Math.max(0, i - 1))
    }

    if (key.downArrow || input.toLowerCase() === 'j') {
      return setIdx(i => Math.min(view.length - 1, i + 1))
    }

    if (key.return) {
      const tr = view[idx]

      return tr ? pick(tr.phrase) : undefined
    }

    if (key.backspace || key.delete) {
      setQuery(q => q.slice(0, -1))

      return setIdx(0)
    }

    if (input && input.length === 1 && input >= ' ' && !key.ctrl && !key.meta) {
      setQuery(q => q + input)
      setIdx(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading triggers…</Text>
  }

  if (err && !list) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc cancel</OverlayHint>
      </Box>
    )
  }

  const { items, offset } = windowItems(view, idx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Workflow trigger
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {query ? `filter: ${query}` : 'type to filter'} · {view.length} trigger{view.length === 1 ? '' : 's'}
      </Text>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {view.length === 0 ? (
        <Text color={t.color.muted}>{query ? `no triggers match "${query}"` : 'no triggers available'}</Text>
      ) : (
        items.map((tr, i) => {
          const at = offset + i === idx

          return (
            <Text color={t.color.muted} {...chipRowProps(t, at)} key={tr.phrase} wrap="truncate-end">
              {at ? '▸ ' : '  '}
              {tr.phrase}
              {tr.description ? (
                <Text color={at ? t.color.accent : t.color.muted}>
                  {' '}
                  — {tr.description}
                </Text>
              ) : null}
            </Text>
          )
        })
      )}

      {offset + VISIBLE < view.length && <Text color={t.color.muted}> ↓ {view.length - offset - VISIBLE} more</Text>}

      {err ? <Text color={t.color.label}>error: {err}</Text> : null}

      <OverlayHint t={t}>↑/↓ select · Enter dispatch · type to filter · Esc cancel</OverlayHint>
    </Box>
  )
}

interface TriggerPickerProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  onPick: (phrase: string) => void
  t: Theme
}