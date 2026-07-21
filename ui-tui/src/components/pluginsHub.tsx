import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { clampIndex, OverlayHint, useOverlayKeys, windowItems, windowOffset } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 44
const MAX_WIDTH = 96

export interface PluginRow {
  description?: string
  name: string
  source?: string
  status?: string
  version?: string
}

interface PluginsListResponse {
  bundled_count?: number
  plugins?: PluginRow[]
  user_count?: number
}

interface PluginsToggleResponse {
  name?: string
  ok?: boolean
  plugin?: PluginRow
  unchanged?: boolean
}

type Scope = 'all' | 'user'

const GLYPH: Record<string, string> = {
  disabled: '✗',
  enabled: '✓'
}

const ctrlChar = (letter: string) => String.fromCharCode(letter.charCodeAt(0) - 96)

export const rankPluginRows = (rows: readonly PluginRow[], query: string): PluginRow[] =>
  query.trim()
    ? fuzzyRank(rows, query, row =>
        [row.name, row.description, row.source, row.version, row.status].filter(Boolean).join(' ')
      ).map(result => result.item)
    : [...rows]

export interface PluginsListKeyFlags {
  backspace?: boolean
  ctrl?: boolean
  delete?: boolean
  downArrow?: boolean
  escape?: boolean
  meta?: boolean
  return?: boolean
  tab?: boolean
  upArrow?: boolean
}

export type PluginsListKeyAction =
  | { ch: string; kind: 'append' }
  | { kind: 'backspace' }
  | { kind: 'clearFilter' }
  | { kind: 'close' }
  | { kind: 'down' }
  | { kind: 'ignore' }
  | { kind: 'quick'; n: number }
  | { kind: 'select' }
  | { kind: 'toggleScope' }
  | { kind: 'up' }

export function classifyPluginsListKey(
  ch: string,
  key: PluginsListKeyFlags,
  filterActive: boolean
): PluginsListKeyAction {
  if (key.escape) {
    return filterActive ? { kind: 'clearFilter' } : { kind: 'close' }
  }

  if (key.upArrow) {
    return { kind: 'up' }
  }

  if (key.downArrow) {
    return { kind: 'down' }
  }

  if (key.return) {
    return { kind: 'select' }
  }

  if (key.tab) {
    return { kind: 'toggleScope' }
  }

  if ((key.backspace || key.delete) && filterActive) {
    return { kind: 'backspace' }
  }

  if (key.ctrl && (ch.toLowerCase() === 'u' || ch === ctrlChar('u')) && filterActive) {
    return { kind: 'clearFilter' }
  }

  if (!filterActive && ch === 'q') {
    return { kind: 'close' }
  }

  // Preserve Space-to-toggle when no query is active; once filtering has
  // started, spaces belong to multi-token search text.
  if (!filterActive && ch === ' ') {
    return { kind: 'select' }
  }

  if (!filterActive) {
    const n = ch === '0' ? 10 : parseInt(ch, 10)

    if (!Number.isNaN(n) && n >= 1 && n <= 10) {
      return { kind: 'quick', n }
    }
  }

  if (ch && ch.length === 1 && ch >= ' ' && !key.ctrl && !key.meta) {
    return { ch, kind: 'append' }
  }

  return { kind: 'ignore' }
}

export function PluginsHub({ gw, onClose, t }: PluginsHubProps) {
  const [rows, setRows] = useState<PluginRow[]>([])
  const [bundledCount, setBundledCount] = useState(0)
  const [userCount, setUserCount] = useState(0)
  const [idx, setIdx] = useState(0)
  const [scope, setScope] = useState<Scope>('user')
  const [filter, setFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  const load = () => {
    gw.request<PluginsListResponse>('plugins.manage', { action: 'list' })
      .then(r => {
        setRows(r?.plugins ?? [])
        setUserCount(Number(r?.user_count ?? 0))
        setBundledCount(Number(r?.bundled_count ?? 0))
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }

  useEffect(load, [gw])

  // Default to user plugins; fall back to all when there are none so the
  // overlay is never empty when bundled plugins exist.
  const { effectiveRows, effectiveScope } = useMemo(() => {
    const visibleRows = scope === 'user' ? rows.filter(r => r.source !== 'bundled') : rows
    const fellBackToAll = scope === 'user' && !visibleRows.length && rows.length > 0
    const scopedRows = fellBackToAll ? rows : visibleRows

    return {
      effectiveRows: rankPluginRows(scopedRows, filter),
      effectiveScope: (fellBackToAll ? 'all' : scope) as Scope
    }
  }, [filter, rows, scope])

  const clampedIdx = clampIndex(idx, effectiveRows.length)

  // On the populated list, q/Esc are filter-aware and handled locally. Keep
  // the shared close behavior for loading, empty, and error states.
  useOverlayKeys({ disabled: busy || rows.length > 0, onClose })

  const toggle = (row: PluginRow) => {
    if (busy || !row) {
      return
    }

    const enable = row.status !== 'enabled'
    setBusy(true)
    setErr('')

    gw.request<PluginsToggleResponse>('plugins.manage', { action: 'toggle', enable, name: row.name })
      .then(r => {
        if (r?.plugin) {
          setRows(prev => prev.map(p => (p.name === r.plugin!.name ? r.plugin! : p)))
        } else {
          load()
        }
      })
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setBusy(false))
  }

  useInput((ch, key) => {
    if (busy || !rows.length) {
      return
    }

    const count = effectiveRows.length
    const action = classifyPluginsListKey(ch, key, filter.trim() !== '')

    switch (action.kind) {
      case 'close':
        return onClose()

      case 'clearFilter':
        setFilter('')
        setIdx(0)

        return

      case 'backspace':
        setFilter(value => value.slice(0, -1))
        setIdx(0)

        return

      case 'append':
        setFilter(value => value + action.ch)
        setIdx(0)

        return

      case 'up':
        if (clampedIdx > 0) {
          setIdx(clampedIdx - 1)
        }

        return

      case 'down':
        if (clampedIdx < count - 1) {
          setIdx(clampedIdx + 1)
        }

        return

      case 'toggleScope':
        setScope(value => (value === 'user' ? 'all' : 'user'))
        setIdx(0)

        return
      case 'select': {
        const row = effectiveRows[clampedIdx]

        if (row) {
          toggle(row)
        }

        return
      }

      case 'quick': {
        if (action.n > count) {
          return
        }

        const next = windowOffset(count, clampedIdx, VISIBLE) + action.n - 1
        const row = effectiveRows[next]

        if (row) {
          setIdx(next)
          toggle(row)
        }

        return
      }

      default:
        return
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading plugins…</Text>
  }

  if (err && !rows.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q close</OverlayHint>
      </Box>
    )
  }

  if (!rows.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          Plugins Hub
        </Text>
        <Text color={t.color.muted}>no plugins installed</Text>
        <Text color={t.color.muted}>install: hermes plugins install owner/repo</Text>
        <OverlayHint t={t}>Esc/q close</OverlayHint>
      </Box>
    )
  }

  const labels = effectiveRows.map(r => {
    const status = r.status ?? 'not enabled'
    const glyph = GLYPH[status] ?? '○'
    const ver = r.version ? ` v${r.version}` : ''
    const src = effectiveScope === 'all' && r.source === 'bundled' ? ' [bundled]' : ''
    const state = status === 'enabled' ? '' : ` (${status})`

    return `${glyph} ${r.name}${ver}${src}${state}`
  })

  const { items, offset } = windowItems(labels, clampedIdx, VISIBLE)

  const scopeLabel =
    effectiveScope === 'user'
      ? `${userCount} user plugin(s)${bundledCount ? ` · +${bundledCount} bundled (Tab)` : ''}`
      : `all ${rows.length} plugins`

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Plugins Hub
      </Text>

      <Text color={t.color.muted}>{scopeLabel}</Text>
      <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
        {filter
          ? `filter: ${filter}▎ · ${effectiveRows.length} match${effectiveRows.length === 1 ? '' : 'es'}`
          : 'type to filter'}
      </Text>
      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {effectiveRows.length === 0 && filter ? (
        <Text color={t.color.muted}>no plugins match the filter</Text>
      ) : (
        items.map((row, i) => {
          const lineIdx = offset + i
          const active = clampedIdx === lineIdx

          return (
            <Text
              bold={active}
              color={active ? t.color.accent : t.color.muted}
              inverse={active}
              key={effectiveRows[lineIdx]?.name ?? row}
              wrap="truncate-end"
            >
              {active ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })
      )}

      {offset + VISIBLE < labels.length && (
        <Text color={t.color.muted}> ↓ {labels.length - offset - VISIBLE} more</Text>
      )}

      {err ? <Text color={t.color.label}>error: {err}</Text> : null}
      {busy ? <Text color={t.color.accent}>updating…</Text> : null}

      <OverlayHint t={t}>
        {filter
          ? effectiveRows.length
            ? '↑/↓ select · Enter toggle · Tab user/all · Backspace edit · Esc clear filter'
            : 'Tab user/all · Backspace edit · Esc clear filter'
          : '↑/↓ select · Enter/Space toggle · Tab user/all · 1-9,0 quick · type to filter · Esc/q close'}
      </OverlayHint>
    </Box>
  )
}

interface PluginsHubProps {
  gw: GatewayClient
  onClose: () => void
  t: Theme
}
