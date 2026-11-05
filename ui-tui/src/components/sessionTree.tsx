import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { SessionTreeNode, SessionTreeOverlay, SessionTreeView } from '../app/interfaces.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 10

/** The rows a query leaves: a case-insensitive title match over the owner's own row order. */
export function treeRows(nodes: SessionTreeNode[], query: string): SessionTreeNode[] {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return nodes
  }

  return nodes.filter(node => (node.title || node.session_id).toLowerCase().includes(needle))
}

/** `dd/mm hh:mm` in the reader's own timezone, the same shape the classic CLI prints. */
function when(createdAt: number): string {
  if (!createdAt) {
    return ''
  }

  const at = new Date(createdAt * 1000)
  const pad = (value: number) => String(value).padStart(2, '0')

  return `${pad(at.getDate())}/${pad(at.getMonth() + 1)} ${pad(at.getHours())}:${pad(at.getMinutes())}`
}

/**
 * `/tree`: the session family around this session — root, branches, compression continuations and
 * subagent runs — with the current session marked. ↑/↓ move, ←/→ fold and unfold, `/` filters
 * titles, `c` and `s` show the continuations and subagent runs the default view folds away, and
 * Enter switches through the ordinary `session.resume` path. Esc closes.
 *
 * The rows come from `session.family`, so the fold and filter rules are the shared Python
 * builder's, not a second implementation: every view change asks the owner for the same family
 * under the new view.
 */
export function SessionTree({ cols = 80, onClose, overlay, t }: SessionTreeProps) {
  const T = useT()
  const R = T.prompt.tree
  const [idx, setIdx] = useState(() => Math.max(0, overlay.nodes.findIndex(node => node.is_current)))
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)

  const title = (node: SessionTreeNode) => node.title || R.untitled(node.session_id.slice(0, 8))
  const rows = treeRows(overlay.nodes, query)
  const at = Math.min(idx, Math.max(0, rows.length - 1))
  const selected = rows[at]

  const setView = (patch: Partial<SessionTreeView>) => overlay.onView({ ...overlay.view, ...patch })

  const fold = (node: SessionTreeNode | undefined, folded: boolean) => {
    if (!node?.has_children) {
      return
    }

    const ids = new Set(overlay.view.folded)

    if (folded) {
      ids.add(node.session_id)
    } else {
      ids.delete(node.session_id)
    }

    setView({ folded: [...ids] })
  }

  useInput((ch, key) => {
    if (searching) {
      if (key.escape) {
        setQuery('')
        setSearching(false)
      }

      return
    }

    if (key.escape) {
      return onClose()
    }

    if (key.upArrow) {
      return setIdx(Math.max(0, at - 1))
    }

    if (key.downArrow) {
      return setIdx(Math.min(Math.max(0, rows.length - 1), at + 1))
    }

    if (key.leftArrow) {
      return fold(selected, true)
    }

    if (key.rightArrow) {
      return fold(selected, false)
    }

    if (ch === '/') {
      return setSearching(true)
    }

    if (ch === 'c' || ch === 'C') {
      return setView({ show_compressions: !overlay.view.show_compressions })
    }

    if (ch === 's' || ch === 'S') {
      return setView({ show_subagents: !overlay.view.show_subagents })
    }

    if (key.return && selected && !selected.is_current) {
      return overlay.onSelect(selected.session_id)
    }
  })

  const width = Math.max(20, cols - 8)
  const { items, offset } = windowItems(rows, at, VISIBLE)

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      <Text bold color={t.color.accent}>
        {R.title}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {R.detail}
      </Text>

      {searching ? (
        <Box paddingLeft={1}>
          <Text color={t.color.label}>{'filter: '}</Text>
          <TextInput
            color={t.color.text}
            columns={width - 10}
            onChange={setQuery}
            onSubmit={() => setSearching(false)}
            value={query}
          />
        </Box>
      ) : null}

      <Text />

      {items.length ? (
        items.map((node, i) => {
          const active = offset + i === at
          const folded = overlay.view.folded.includes(node.session_id)
          const glyph = node.has_children ? (folded ? '▸ ' : '▾ ') : '· '

          return (
            <Text key={node.session_id} wrap="truncate-end">
              <Text color={t.color.muted} {...chipRowProps(t, active)}>
                {active ? '▸ ' : '  '}
                {'  '.repeat(node.depth)}
                {glyph}
                {title(node)}
                {' · '}
                {kindLabel(R, node.kind)}
                {' · '}
                {node.message_count === 1 ? R.msg(String(node.message_count)) : R.msgs(String(node.message_count))}
                {when(node.created_at) ? ` · ${when(node.created_at)}` : ''}
                {node.branch_from === null ? '' : ` · ${R.afterMsg(String(node.branch_from))}`}
                {node.folded_segments ? ` · ${R.compressed(String(node.folded_segments))}` : ''}
                {node.is_current ? ` · ${R.current}` : ''}
              </Text>
            </Text>
          )
        })
      ) : (
        <Text color={t.color.muted}>{R.empty}</Text>
      )}

      {overlay.truncated ? <Text color={t.color.muted}>{R.truncated}</Text> : null}

      <Text color={t.color.muted}>{searching ? R.searchHint : R.listHint}</Text>
    </Box>
  )
}

/** Kind labels are a table, not a ladder: a new kind in the builder gets a label or reads unknown. */
const KIND_LABELS = {
  branch: 'kindBranch',
  compression: 'kindCompression',
  root: 'kindRoot',
  subagent: 'kindSubagent',
  unknown: 'kindUnknown'
} as const

type TreeStrings = ReturnType<typeof useT>['prompt']['tree']
type KindLabels = Record<(typeof KIND_LABELS)[keyof typeof KIND_LABELS], string>

function kindLabel(R: KindLabels, kind: SessionTreeNode['kind']) {
  return R[KIND_LABELS[kind] ?? 'kindUnknown']
}

interface SessionTreeProps {
  cols?: number
  onClose: () => void
  overlay: SessionTreeOverlay
  t: Theme
}
