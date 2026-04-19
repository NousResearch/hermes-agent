import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq, ConfirmReq } from '../types.js'

import { TextInput } from './textInput.js'
import { isMac } from '../lib/platform.js'

const OPTS = ['once', 'session', 'always', 'deny'] as const
const LABELS = { always: 'Always allow', deny: 'Deny', once: 'Allow once', session: 'Allow this session' } as const
const CMD_PREVIEW_LINES = 10

export function ApprovalPrompt({ onChoice, req, t }: ApprovalPromptProps) {
  const [sel, setSel] = useState(0)

  useInput((ch, key) => {
    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < OPTS.length - 1) {
      setSel(s => s + 1)
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= OPTS.length) {
      onChoice(OPTS[n - 1]!)

      return
    }

    if (key.return) {
      onChoice(OPTS[sel]!)
    }
  })

  const rawLines = req.command.split('\n')
  const shown = rawLines.slice(0, CMD_PREVIEW_LINES)
  const overflow = rawLines.length - shown.length

  return (
    <Box borderColor={t.color.warn} borderStyle="double" flexDirection="column" paddingX={1}>
      <Text bold color={t.color.warn}>
        ⚠ approval required · {req.description}
      </Text>

      <Box flexDirection="column" paddingLeft={1}>
        {shown.map((line, i) => (
          <Text color={t.color.cornsilk} key={i} wrap="truncate-end">
            {line || ' '}
          </Text>
        ))}

        {overflow > 0 ? (
          <Text color={t.color.dim}>
            … +{overflow} more line{overflow === 1 ? '' : 's'} (full text above)
          </Text>
        ) : null}
      </Box>

      <Text />

      {OPTS.map((o, i) => (
        <Text key={o}>
          <Text color={sel === i ? t.color.warn : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            {i + 1}. {LABELS[o]}
          </Text>
        </Text>
      ))}

      <Text color={t.color.dim}>↑/↓ select · Enter confirm · 1-4 quick pick · Ctrl+C deny</Text>
    </Box>
  )
}

export function ClarifyPrompt({ cols = 80, onAnswer, onCancel, req, t }: ClarifyPromptProps) {
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []

  const heading = (
    <Text bold>
      <Text color={t.color.amber}>ask</Text>
      <Text color={t.color.cornsilk}> {req.question}</Text>
    </Text>
  )

  useInput((ch, key) => {
    if (key.escape) {
      typing && choices.length ? setTyping(false) : onCancel()

      return
    }

    if (typing || !choices.length) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < choices.length) {
      setSel(s => s + 1)
    }

    if (key.return) {
      sel === choices.length ? setTyping(true) : choices[sel] && onAnswer(choices[sel]!)
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= choices.length) {
      onAnswer(choices[n - 1]!)
    }
  })

  if (typing || !choices.length) {
    return (
      <Box flexDirection="column">
        {heading}

        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput columns={Math.max(20, cols - 6)} allowClipboardHotkeys={isMac} onChange={setCustom} onSubmit={onAnswer} value={custom} />
        </Box>

        <Text color={t.color.dim}>
          Enter send · Esc {choices.length ? 'back' : 'cancel'} · {isMac ? 'Cmd+C copy · Cmd+V paste · Ctrl+C cancel' : 'Ctrl+C cancel'}
        </Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      {heading}

      {[...choices, 'Other (type your answer)'].map((c, i) => (
        <Text key={i}>
          <Text color={sel === i ? t.color.label : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            {i + 1}. {c}
          </Text>
        </Text>
      ))}

      <Text color={t.color.dim}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick · Esc/Ctrl+C cancel</Text>
    </Box>
  )
}

export function ConfirmPrompt({ onCancel, onConfirm, req, t }: ConfirmPromptProps) {
  const [sel, setSel] = useState(0)

  useInput((ch, key) => {
    const lower = ch.toLowerCase()

    if (key.escape || (key.ctrl && lower === 'c') || lower === 'n') {
      return onCancel()
    }

    if (lower === 'y') {
      return onConfirm()
    }

    if (key.upArrow) {
      setSel(0)
    }

    if (key.downArrow) {
      setSel(1)
    }

    if (key.return) {
      sel === 0 ? onCancel() : onConfirm()
    }
  })

  const accent = req.danger ? t.color.error : t.color.warn

  const rows = [
    { color: t.color.cornsilk, label: req.cancelLabel ?? 'No' },
    { color: req.danger ? t.color.error : t.color.cornsilk, label: req.confirmLabel ?? 'Yes' }
  ]

  return (
    <Box borderColor={accent} borderStyle="double" flexDirection="column" paddingX={1}>
      <Text bold color={accent}>
        {req.danger ? '⚠' : '?'} {req.title}
      </Text>

      {req.detail ? (
        <Box paddingLeft={1}>
          <Text color={t.color.cornsilk} wrap="truncate-end">
            {req.detail}
          </Text>
        </Box>
      ) : null}

      <Text />

      {rows.map((row, i) => (
        <Text key={row.label}>
          <Text color={sel === i ? accent : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? row.color : t.color.dim}>{row.label}</Text>
        </Text>
      ))}

      <Text color={t.color.dim}>↑/↓ select · Enter confirm · Y/N quick · Esc cancel</Text>
    </Box>
  )
}

interface ApprovalPromptProps {
  onChoice: (s: string) => void
  req: ApprovalReq
  t: Theme
}

interface ClarifyPromptProps {
  cols?: number
  onAnswer: (s: string) => void
  onCancel: () => void
  req: ClarifyReq
  t: Theme
}

interface ConfirmPromptProps {
  onCancel: () => void
  onConfirm: () => void
  req: ConfirmReq
  t: Theme
}
