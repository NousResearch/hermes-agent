import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { RewindOverlay, RewindTarget } from '../app/interfaces.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { windowItems } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 8

const firstLine = (text: string) => text.trim().split('\n')[0] ?? ''

/**
 * The rewind picker: earlier prompts newest first; Enter opens the picked one for editing, and
 * Enter again cuts the session back to just before it and sends the edit. Esc steps back (edit →
 * list → closed); Ctrl+C closes it from the global handler. Nothing is cut before that send.
 */
export function RewindPicker({ cols = 80, onClose, onSubmit, req, t }: RewindPickerProps) {
  const T = useT()
  const R = T.prompt.rewind
  const rows = [...req.targets].reverse()
  const [idx, setIdx] = useState(0)
  const [editing, setEditing] = useState<null | RewindTarget>(null)
  const [draft, setDraft] = useState('')

  useInput((_ch, key) => {
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

          <Text color={t.color.muted}>{R.listHint}</Text>
        </Box>
      )}
    </Box>
  )
}

interface RewindPickerProps {
  cols?: number
  onClose: () => void
  onSubmit: (target: RewindTarget, text: string) => void
  req: RewindOverlay
  t: Theme
}
