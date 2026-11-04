import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { RewindOverlay, RewindTarget } from '../app/interfaces.js'
import { $rewindFocus } from '../app/rewindFocusStore.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 8
const PREVIEW_LINES = 3

const lines = (text: string) => text.trim().split('\n')
const firstLine = (text: string) => lines(text)[0] ?? ''

/**
 * The rewind picker: earlier prompts newest first; Enter opens the picked one for editing, and
 * Enter again cuts the session back to just before it and sends the edit. `f` forks a new session
 * from just before the picked prompt (the original stays intact). Esc steps back (edit → list →
 * closed); Ctrl+C closes it from the global handler. Nothing is cut before that send. The selected
 * prompt is scrolled to and marked in the transcript while the list is up.
 */
export function RewindPicker({ cols = 80, onClose, onFork, onSubmit, req, t }: RewindPickerProps) {
  const T = useT()
  const R = T.prompt.rewind
  const rows = [...req.targets].reverse()
  const [idx, setIdx] = useState(0)
  const [editing, setEditing] = useState<null | RewindTarget>(null)
  const [draft, setDraft] = useState('')
  const focused = editing ?? rows[idx]

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

    if (key.upArrow) {
      return setIdx(i => Math.max(0, i - 1))
    }

    if (key.downArrow) {
      return setIdx(i => Math.min(rows.length - 1, i + 1))
    }

    if ((ch === 'f' || ch === 'F') && rows[idx]) {
      return onFork(rows[idx])
    }

    if (key.return && rows[idx]) {
      setEditing(rows[idx])
      setDraft(rows[idx].text)
    }
  })

  const width = Math.max(20, cols - 8)
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

          {rows[idx] ? (
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

          <Text color={t.color.muted}>{R.listHint}</Text>
        </Box>
      )}
    </Box>
  )
}

interface RewindPickerProps {
  cols?: number
  onClose: () => void
  onFork: (target: RewindTarget) => void
  onSubmit: (target: RewindTarget, text: string) => void
  req: RewindOverlay
  t: Theme
}
