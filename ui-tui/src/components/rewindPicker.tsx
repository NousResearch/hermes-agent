import { Box, Text, useInput, useStdout, wrapAnsi } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { RewindOverlay, RewindTarget } from '../app/interfaces.js'
import { $rewindFocus } from '../app/rewindFocusStore.js'
import { NATIVE_MODE } from '../config/env.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 8
const PREVIEW_LINES = 3

const CONTEXT_MIN_ROWS = 6

const lines = (text: string) => text.trim().split('\n')
const firstLine = (text: string) => lines(text)[0] ?? ''

/** Rows of the selected exchange for the context view: the prompt in full, then the reply after it. */
export function rewindContextRows(target: RewindTarget, width: number, labels: RewindContextLabels): ContextRow[] {
  const wrap = (text: string) =>
    lines(text).flatMap(line => wrapAnsi(line, Math.max(8, width - 2), { hard: true, trim: false }).split('\n'))

  return [
    { kind: 'label', text: labels.prompt },
    ...wrap(target.text).map(text => ({ kind: 'text' as const, text: `  ${text}` })),
    { kind: 'text', text: '' },
    { kind: 'label', text: labels.reply },
    ...(target.reply
      ? wrap(target.reply).map(text => ({ kind: 'text' as const, text: `  ${text}` }))
      : [{ kind: 'muted' as const, text: `  ${labels.noReply}` }])
  ]
}

/** How many context rows fit: about half of what the list and the picker chrome leave. */
export const rewindContextWindow = (termRows: number, listRows: number) =>
  Math.max(CONTEXT_MIN_ROWS, Math.floor((termRows - 14 - Math.min(listRows, VISIBLE)) / 2))

/**
 * The rewind picker: earlier prompts newest first; Enter opens the picked one for editing, and
 * Enter again cuts the session back to just before it and sends the edit. `f` forks a new session
 * from just before the picked prompt (the original stays intact). Esc steps back (edit → list →
 * closed); Ctrl+C closes it from the global handler. Nothing is cut before that send. The selected
 * prompt is scrolled to and marked in the transcript while the list is up. In native-scrollback mode
 * the transcript is the terminal's own scrollback, which the app cannot scroll, so `contextView`
 * shows the selected prompt with the reply that followed it inside the picker (PgUp/PgDn page it).
 */
export function RewindPicker({
  cols = 80,
  contextView = NATIVE_MODE,
  onClose,
  onFork,
  onSubmit,
  req,
  t
}: RewindPickerProps) {
  const T = useT()
  const R = T.prompt.rewind
  const rows = [...req.targets].reverse()
  const [idx, setIdx] = useState(0)
  const [editing, setEditing] = useState<null | RewindTarget>(null)
  const [draft, setDraft] = useState('')
  const [contextOffset, setContextOffset] = useState(0)
  const { stdout } = useStdout()
  const focused = editing ?? rows[idx]
  const width = Math.max(20, cols - 8)

  const contextRows =
    contextView && rows[idx]
      ? rewindContextRows(rows[idx], width - 4, { noReply: R.noReply, prompt: R.promptLabel, reply: R.replyLabel })
      : []

  const contextWindow = rewindContextWindow(stdout?.rows ?? 24, rows.length)
  const maxContextOffset = Math.max(0, contextRows.length - contextWindow)
  const shownOffset = Math.min(contextOffset, maxContextOffset)

  useEffect(() => {
    $rewindFocus.set(focused ? { text: focused.text, turn: focused.turn } : null)
  }, [focused])

  useEffect(() => () => $rewindFocus.set(null), [])

  useInput((ch, key) => {
    if (editing) {
      if (key.escape) {
        setEditing(null)
      }

      return
    }

    if (key.escape) {
      return onClose()
    }

    if (key.upArrow || key.downArrow) {
      setContextOffset(0)

      return setIdx(i => (key.upArrow ? Math.max(0, i - 1) : Math.min(rows.length - 1, i + 1)))
    }

    if (contextView && (key.pageUp || key.pageDown)) {
      const step = Math.max(1, contextWindow - 1)

      return setContextOffset(Math.max(0, Math.min(maxContextOffset, shownOffset + (key.pageUp ? -step : step))))
    }

    if ((ch === 'f' || ch === 'F') && rows[idx]) {
      return onFork(rows[idx])
    }

    if (key.return && rows[idx]) {
      setEditing(rows[idx])
      setDraft(rows[idx].text)
    }
  })

  const { items, offset } = windowItems(rows, idx, VISIBLE)

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      <Text bold color={t.color.accent}>
        {R.title}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {R.detail}
      </Text>

      <Text />

      {editing ? (
        <Box flexDirection="column">
          <Text color={t.color.label}>{R.editing(String(editing.turn))}</Text>

          <Box paddingLeft={1}>
            <Text color={t.color.label}>{'> '}</Text>
            <TextInput
              color={t.color.text}
              columns={width - 2}
              onChange={setDraft}
              onSubmit={value => (value.trim() ? onSubmit(editing, value) : setEditing(null))}
              value={draft}
            />
          </Box>

          <Text color={t.color.muted}>{R.editHint}</Text>
        </Box>
      ) : (
        <Box flexDirection="column">
          {items.map((row, i) => {
            const selected = offset + i === idx

            return (
              <Text key={row.rowId} wrap="truncate-end">
                <Text color={t.color.muted} {...chipRowProps(t, selected)}>
                  {selected ? '▸ ' : '  '}
                  {row.turn}. {firstLine(row.text)}
                </Text>
              </Text>
            )
          })}

          {rows[idx] && contextView ? (
            <Box borderColor={t.color.muted} borderStyle="single" flexDirection="column" marginTop={1} paddingX={1}>
              <Text color={t.color.label} wrap="truncate-end">
                {R.preview(String(rows[idx].turn))}
                {contextRows.length > contextWindow
                  ? `  ${R.contextScroll(
                      String(shownOffset + 1),
                      String(Math.min(contextRows.length, shownOffset + contextWindow)),
                      String(contextRows.length)
                    )}`
                  : ''}
              </Text>
              {contextRows.slice(shownOffset, shownOffset + contextWindow).map((row, i) => (
                <Text
                  bold={row.kind === 'label'}
                  color={row.kind === 'label' ? t.color.label : row.kind === 'muted' ? t.color.muted : t.color.text}
                  key={shownOffset + i}
                  wrap="truncate-end"
                >
                  {row.text || ' '}
                </Text>
              ))}
            </Box>
          ) : rows[idx] ? (
            <Box borderColor={t.color.muted} borderStyle="single" flexDirection="column" marginTop={1} paddingX={1}>
              <Text color={t.color.label}>{R.preview(String(rows[idx].turn))}</Text>
              {lines(rows[idx].text)
                .slice(0, PREVIEW_LINES)
                .map((line, i) => (
                  <Text color={t.color.text} key={i} wrap="truncate-end">
                    {line || ' '}
                  </Text>
                ))}
              {lines(rows[idx].text).length > PREVIEW_LINES ? <Text color={t.color.muted}>…</Text> : null}
            </Box>
          ) : null}

          <Text color={t.color.muted}>{contextView ? R.listHintContext : R.listHint}</Text>
        </Box>
      )}
    </Box>
  )
}

interface ContextRow {
  kind: 'label' | 'muted' | 'text'
  text: string
}

interface RewindContextLabels {
  noReply: string
  prompt: string
  reply: string
}

interface RewindPickerProps {
  cols?: number
  /** Show the selected exchange inside the picker (native-scrollback mode, where the transcript cannot scroll). */
  contextView?: boolean
  onClose: () => void
  onFork: (target: RewindTarget) => void
  onSubmit: (target: RewindTarget, text: string) => void
  req: RewindOverlay
  t: Theme
}
