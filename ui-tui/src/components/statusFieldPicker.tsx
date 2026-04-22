import { Box, Text, useInput } from '@hermes/ink'
import type { ReactNode } from 'react'
import { useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { ConfigSetResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'

const ALL_FIELDS = [
  { id: 'status', label: 'status', desc: 'ready / FaceTicker' },
  { id: 'model', label: 'model', desc: 'model name' },
  { id: 'context', label: 'context', desc: '45k/200k' },
  { id: 'context_bar', label: 'ctx_bar', desc: 'progress bar + %' },
  { id: 'duration', label: 'duration', desc: 'elapsed time' },
  { id: 'voice', label: 'voice', desc: 'voice label' },
  { id: 'bg', label: 'bg', desc: 'background tasks' },
  { id: 'cost', label: 'cost', desc: '$ cost' },
  { id: 'cwd', label: 'cwd', desc: 'dir + branch' }
] as const

type FieldId = (typeof ALL_FIELDS)[number]['id']

type Side = 'left' | 'right'

const DEFAULT_LEFT: FieldId[] = ['status', 'model', 'context', 'context_bar', 'duration', 'voice', 'bg', 'cost']
const DEFAULT_RIGHT: FieldId[] = ['cwd']

const FIELD_BY_ID = new Map<string, (typeof ALL_FIELDS)[number]>(ALL_FIELDS.map(f => [f.id, f]))

export function StatusFieldPicker({
  currentFieldsLeft,
  currentFieldsRight,
  gw,
  onCancel,
  onSelect,
  t
}: StatusFieldPickerProps) {
  const [left, setLeft] = useState<FieldId[]>(() =>
    currentFieldsLeft?.length ? (currentFieldsLeft as FieldId[]) : [...DEFAULT_LEFT]
  )

  const [right, setRight] = useState<FieldId[]>(() =>
    currentFieldsRight?.length ? (currentFieldsRight as FieldId[]) : [...DEFAULT_RIGHT]
  )

  const [side, setSide] = useState<Side>('left')
  const [sel, setSel] = useState(0)

  // 'add' mode: picking a field to add from available pool
  const [addMode, setAddMode] = useState(false)
  const [addSel, setAddSel] = useState(0)

  const enabledList = side === 'left' ? left : right
  const setEnabledList = side === 'left' ? setLeft : setRight

  const maxSel = enabledList.length - 1

  // Available fields = not in either side
  const usedSet = new Set<string>([...left, ...right])
  const available = ALL_FIELDS.filter(f => !usedSet.has(f.id))

  useInput((ch, key) => {
    if (key.escape) {
      if (addMode) {
        setAddMode(false)

        return
      }

      onCancel()

      return
    }

    // Add mode: pick a field to add
    if (addMode) {
      if (key.upArrow) {
        setAddSel(v => Math.max(0, v - 1))

        return
      }

      if (key.downArrow) {
        setAddSel(v => Math.min(available.length - 1, v + 1))

        return
      }

      if (key.return && available.length > 0) {
        const fieldId = available[addSel]!.id
        setEnabledList(prev => [...prev, fieldId])
        setAddMode(false)
        setSel(enabledList.length) // select the newly added item

        return
      }

      // Any other key cancels add mode
      if (ch && !key.return) {
        setAddMode(false)
      }

      return
    }

    // ←/→: switch side
    if (key.leftArrow) {
      setSide('left')
      setSel(0)

      return
    }

    if (key.rightArrow) {
      setSide('right')
      setSel(0)

      return
    }

    // ↑/↓: move selection
    if (key.upArrow && !key.shift && maxSel >= 0) {
      setSel(v => Math.max(0, v - 1))

      return
    }

    if (key.downArrow && !key.shift && maxSel >= 0) {
      setSel(v => Math.min(maxSel, v + 1))

      return
    }

    // Shift+↑/↓: reorder
    if (key.shift && key.upArrow && sel > 0) {
      setEnabledList(prev => {
        const next = [...prev]

        ;[next[sel - 1]!, next[sel]!] = [next[sel]!, next[sel - 1]!]

        return next
      })
      setSel(v => v - 1)

      return
    }

    if (key.shift && key.downArrow && sel < maxSel) {
      setEnabledList(prev => {
        const next = [...prev]

        ;[next[sel]!, next[sel + 1]!] = [next[sel + 1]!, next[sel]!]

        return next
      })
      setSel(v => v + 1)

      return
    }

    // a: add field (enter add mode)
    if (ch === 'a') {
      if (available.length === 0) {return}
      setAddMode(true)
      setAddSel(0)

      return
    }

    // d: delete selected field
    if (ch === 'd' && maxSel >= 0) {
      setEnabledList(prev => prev.filter((_, i) => i !== sel))
      setSel(v => Math.min(v, Math.max(0, maxSel - 1)))

      return
    }

    // Enter: confirm
    if (key.return) {
      onSelect(left, right)
      gw.request<ConfigSetResponse>('config.set', { key: 'statusbar_fields_left', value: left }).catch(() => {})
      gw.request<ConfigSetResponse>('config.set', { key: 'statusbar_fields_right', value: right }).catch(() => {})

      return
    }

    // r: reset
    if (ch === 'r') {
      setLeft([...DEFAULT_LEFT])
      setRight([...DEFAULT_RIGHT])
      setSide('left')
      setSel(0)

      return
    }
  })

  const renderColumn = (
    title: string,
    isActive: boolean,
    list: FieldId[]
  ): ReactNode => (
    <Box flexDirection="column">
      <Text bold color={isActive ? t.color.cornsilk : t.color.dim}>
        {isActive ? '▸ ' : '  '}{title}
      </Text>
      <Text color={t.color.dim}>{'─'.repeat(24)}</Text>
      {list.length === 0 ? (
        <Text color={t.color.dim} italic>  (empty)</Text>
      ) : (
        list.map((id, i) => {
          const f = FIELD_BY_ID.get(id)
          const isCur = isActive && !addMode && sel === i

          return (
            <Box key={`${id}:${i}`}>
              <Text bold={isCur} color={isCur ? t.color.cornsilk : undefined}>
                {isCur ? '▸ ' : '  '}{f?.label ?? id}: {f?.desc ?? ''}
              </Text>
            </Box>
          )
        })
      )}
    </Box>
  )

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>Status Bar Fields</Text>
      {!addMode && (
        <Text color={t.color.dim}>
          ↑/↓ select · ←/→ switch side · a add · d delete · Shift+↑/↓ reorder · Enter confirm · r reset · Esc cancel
        </Text>
      )}
      {addMode && (
        <Text color={t.color.cornsilk}>
          Pick a field to add · ↑/↓ select · Enter confirm · Esc cancel
        </Text>
      )}

      <Box gap={4} marginTop={1}>
        {renderColumn('Left (flex)', side === 'left', left)}
        {renderColumn('Right (pinned)', side === 'right', right)}
      </Box>

      {/* Add mode overlay: show available fields */}
      {addMode && available.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color={t.color.amber}>Add field to {side}:</Text>
          {available.map((f, i) => (
            <Box key={f.id}>
              <Text bold={addSel === i} color={addSel === i ? t.color.cornsilk : undefined}>
                {addSel === i ? '▸ ' : '  '}{f.label}: {f.desc}
              </Text>
            </Box>
          ))}
        </Box>
      )}

      <Box marginTop={1}>
        <Text color={t.color.dim} italic>
          Left: {left.length} · Right: {right.length}
        </Text>
      </Box>
    </Box>
  )
}

interface StatusFieldPickerProps {
  currentFieldsLeft?: string[]
  currentFieldsRight?: string[]
  gw: GatewayClient
  onCancel: () => void
  onSelect: (fieldsLeft: string[], fieldsRight: string[]) => void
  t: Theme
}
