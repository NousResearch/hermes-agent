import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useState } from 'react'

import type { SessionTreeNode, SessionTreeOverlay, SessionTreeView } from '../app/interfaces.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

/** The same marks the classic CLI's browser draws (hermes_cli/session_tree.py KIND_GLYPHS). */
const KIND_GLYPHS: Record<SessionTreeNode['kind'], string> = {
  branch: '◇',
  compression: '≋',
  root: '◆',
  subagent: '✦',
  unknown: '·'
}

const HERE = '☤'
const RULE = '┄'
const WEIGHT_BARS = '▁▂▃▄▅▆▇█'
const CHROME_ROWS = 16

/** The rows a query leaves: a case-insensitive title match over the owner's own row order. */
export function treeRows(nodes: SessionTreeNode[], query: string): SessionTreeNode[] {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return nodes
  }

  return nodes.filter(node => (node.title || node.session_id).toLowerCase().includes(needle))
}

/** One cell for a row's message count, log-scaled against the family's largest (as the CLI). */
export function weightBar(count: number, heaviest: number): string {
  if (count <= 0 || heaviest <= 0) {
    return ' '
  }

  const level = Math.round(((WEIGHT_BARS.length - 1) * Math.log1p(count)) / Math.log1p(heaviest))

  return WEIGHT_BARS[Math.min(level, WEIGHT_BARS.length - 1)]!
}

const clip = (text: string, width: number) =>
  text.length <= width ? text : `${text.slice(0, Math.max(1, width - 1))}…`

/**
 * `/tree`: the session family around this session — the original, its branches, compressed
 * segments and subagent runs — as a tree with a preview of the highlighted chat. ↑/↓ move, ←/→
 * fold and unfold (← on a leaf jumps to its parent), `/` filters titles, `c` and `s` show the
 * segments and runs the default view folds away, Enter switches through the ordinary
 * `session.resume` path, Esc closes.
 *
 * Rows, guides, ages and previews come from `session.family` (the shared Python builder), so the
 * classic CLI's browser shows the same cells; every view change asks the owner again.
 */
export function SessionTree({ cols = 80, onClose, overlay, t }: SessionTreeProps) {
  const T = useT()
  const R = T.prompt.tree
  const { stdout } = useStdout()

  const [idx, setIdx] = useState(() =>
    Math.max(
      0,
      overlay.nodes.findIndex(node => node.is_current)
    )
  )

  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)

  const title = (node: SessionTreeNode) => node.title || R.untitled(node.session_id.slice(0, 8))
  const rows = treeRows(overlay.nodes, query)
  const at = Math.min(idx, Math.max(0, rows.length - 1))
  const selected = rows[at]

  const setView = (patch: Partial<SessionTreeView>) => overlay.onView({ ...overlay.view, ...patch })

  const fold = (node: SessionTreeNode | undefined, close: boolean) => {
    if (!node) {
      return
    }

    const ids = new Set(overlay.view.folded)

    if (close && node.has_children && !ids.has(node.session_id)) {
      ids.add(node.session_id)
    } else if (close) {
      const parent = rows.findIndex(row => row.session_id === node.parent_session_id)

      return parent >= 0 ? setIdx(parent) : undefined
    } else if (ids.has(node.session_id)) {
      ids.delete(node.session_id)
    } else {
      return
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

    if (key.leftArrow || key.rightArrow) {
      return fold(selected, key.leftArrow)
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

  const width = Math.max(40, cols - 6)
  const visible = Math.max(3, Math.min(rows.length || 1, (stdout?.rows ?? 30) - CHROME_ROWS))
  const { items, offset } = windowItems(rows, at, visible)
  const heaviest = Math.max(0, ...overlay.nodes.map(node => node.message_count))
  const widest = Math.max(0, ...rows.map(node => (node.guide ?? '').length + title(node).length))
  const titleWidth = Math.max(24, Math.min(widest + 4, width - 50))
  const root = overlay.nodes.find(node => node.session_id === overlay.rootId)
  const rule = <Text color={t.color.muted}>{RULE.repeat(width)}</Text>

  const kindColor = (kind: SessionTreeNode['kind']) =>
    ({
      branch: t.color.label,
      compression: t.color.syntaxKeyword,
      root: t.color.accent,
      subagent: t.color.ok,
      unknown: t.color.muted
    })[kind]

  const notes = (node: SessionTreeNode) =>
    [
      node.branch_from === null ? '' : R.afterMsg(String(node.branch_from)),
      node.folded_segments ? R.compressed(String(node.folded_segments)) : '',
      node.has_children && overlay.view.folded.includes(node.session_id) ? R.folded : '',
      node.is_current ? R.here : ''
    ].filter(Boolean)

  const messages = (count: number) => (count === 1 ? R.msg(String(count)) : R.msgs(String(count)))

  const hints: [string, string][] = [
    ['↑↓', R.hintMove],
    ['←→', R.hintFold],
    ['⏎', R.hintSwitch],
    ['/', R.hintSearch],
    ['c', R.hintCompressed],
    ['s', R.hintSubagents],
    ['esc', R.hintClose]
  ]

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box justifyContent="space-between" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {`${HERE} ${R.header(root ? title(root) : overlay.rootId)}`}
        </Text>
        <Text color={t.color.muted}>{R.count(String(overlay.nodes.length))}</Text>
      </Box>
      {rule}
      <Text />

      {items.length ? (
        items.map((node, i) => {
          const active = offset + i === at
          const chip = chipRowProps(t, active)
          const guide = node.guide ?? ''
          const name = clip(title(node), Math.max(8, titleWidth - guide.length))
          const glyph = node.is_current ? HERE : (KIND_GLYPHS[node.kind] ?? '·')
          const kind = `   ${kindLabel(R, node.kind).padEnd(12)}`
          const size = `${weightBar(node.message_count, heaviest)} ${messages(node.message_count).padStart(8)}`
          const age = `    ${(node.age ?? '').padEnd(11)}`
          const note = notes(node).join('  ')
          // Pad to the panel width so the selection chip runs edge to edge, not just under the text.
          const used = 3 + titleWidth + kind.length + size.length + age.length + note.length
          const fill = ' '.repeat(Math.max(0, width - used))
          const quiet = active ? (chip.color ?? t.color.text) : t.color.muted

          return (
            <Text backgroundColor={chip.backgroundColor} key={node.session_id} wrap="truncate-end">
              <Text color={quiet}>{` ${guide}`}</Text>
              <Text bold color={kindColor(node.kind)}>{`${glyph} `}</Text>
              <Text bold={active || node.is_current} color={chip.color ?? t.color.text}>
                {name.padEnd(Math.max(0, titleWidth - guide.length))}
              </Text>
              <Text color={kindColor(node.kind)}>{kind}</Text>
              <Text color={t.color.label}>{size}</Text>
              <Text color={quiet}>{age}</Text>
              <Text bold color={t.color.accent}>
                {note}
              </Text>
              <Text>{fill}</Text>
            </Text>
          )
        })
      ) : (
        <Text color={t.color.muted}>{query ? `  ${R.noMatches}` : `  ${R.empty}`}</Text>
      )}

      {overlay.truncated ? <Text color={t.color.muted}>{`  ${R.truncated}`}</Text> : null}
      <Text />
      {rule}

      {selected ? <Preview messages={messages} node={selected} R={R} t={t} title={title} width={width} /> : null}

      {rule}
      {searching ? (
        <Box paddingLeft={1}>
          <Text bold color={t.color.accent}>
            {'/ '}
          </Text>
          <TextInput
            color={t.color.text}
            columns={width - 6}
            onChange={setQuery}
            onSubmit={() => setSearching(false)}
            value={query}
          />
          <Text color={t.color.muted}>{`  ${R.searchHint}`}</Text>
        </Box>
      ) : (
        <Text wrap="truncate-end">
          {hints.map(([keyName, action]) => (
            <Text key={keyName}>
              <Text bold color={t.color.accent}>{` ${keyName}`}</Text>
              <Text color={t.color.muted}>{` ${action}  `}</Text>
            </Text>
          ))}
        </Text>
      )}
    </Box>
  )
}

/** The highlighted chat: where it came from, how big and how recent, then its last exchange. */
function Preview({ messages, node, R, t, title, width }: PreviewProps) {
  const facts = [
    node.parent_session_id
      ? R.fromParent(kindLabel(R, node.kind), node.parent_title || R.untitled(node.parent_session_id.slice(0, 8)))
      : kindLabel(R, node.kind),
    messages(node.message_count),
    node.age ? R.lastUsed(node.age) : ''
  ].filter(Boolean)

  const exchange: [string, string, string][] = [
    [R.you, '›', node.last_prompt ?? ''],
    [R.hermes, '‹', node.last_reply ?? '']
  ]

  const said = exchange.filter(([, , text]) => text)

  return (
    <Box flexDirection="column">
      <Text wrap="truncate-end">
        <Text
          bold
          color={t.color.accent}
        >{`  ${node.is_current ? HERE : (KIND_GLYPHS[node.kind] ?? '·')} ${title(node)}`}</Text>
        <Text color={t.color.muted}>{`   ${facts.join(' · ')}`}</Text>
      </Text>
      {said.length ? (
        said.map(([who, mark, text]) => (
          <Text key={who} wrap="truncate-end">
            <Text bold color={mark === '›' ? t.color.label : t.color.accent}>{`${who.padStart(10)} ${mark}  `}</Text>
            <Text color={t.color.text}>{clip(text, width - 16)}</Text>
          </Text>
        ))
      ) : (
        <Text color={t.color.muted}>{`${' '.repeat(14)}${R.noMessages}`}</Text>
      )}
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

function kindLabel(R: TreeStrings, kind: SessionTreeNode['kind']) {
  return R[KIND_LABELS[kind] ?? 'kindUnknown']
}

interface PreviewProps {
  messages: (count: number) => string
  node: SessionTreeNode
  R: TreeStrings
  t: Theme
  title: (node: SessionTreeNode) => string
  width: number
}

interface SessionTreeProps {
  cols?: number
  onClose: () => void
  overlay: SessionTreeOverlay
  t: Theme
}
