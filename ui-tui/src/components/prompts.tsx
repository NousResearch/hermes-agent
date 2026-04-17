import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq } from '../types.js'

import { TextInput } from './textInput.js'

const OPTS = ['once', 'session', 'always', 'deny'] as const
const LABELS = { always: 'Always allow', deny: 'Deny', once: 'Allow once', session: 'Allow this session' } as const

export function ApprovalPrompt({ onChoice, req, t }: ApprovalPromptProps) {
  const [sel, setSel] = useState(3)

  useInput((ch, key) => {
    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < 3) {
      setSel(s => s + 1)
    }

    if (key.return) {
      onChoice(OPTS[sel]!)
    }

    if (ch === 'o') {
      onChoice('once')
    }

    if (ch === 's') {
      onChoice('session')
    }

    if (ch === 'a') {
      onChoice('always')
    }

    if (ch === 'd' || key.escape) {
      onChoice('deny')
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        ! DANGEROUS COMMAND: {req.description}
      </Text>

      <Text color={t.color.dim}> {req.command}</Text>
      <Text />

      {OPTS.map((o, i) => (
        <Text key={o}>
          <Text color={sel === i ? t.color.warn : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            [{o[0]}] {LABELS[o]}
          </Text>
        </Text>
      ))}

      <Text color={t.color.dim}>↑/↓ select · Enter confirm · o/s/a/d quick pick</Text>
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
          <TextInput columns={Math.max(20, cols - 6)} onChange={setCustom} onSubmit={onAnswer} value={custom} />
        </Box>

        <Text color={t.color.dim}>Enter send · Esc {choices.length ? 'back' : 'cancel'} · Ctrl+C cancel</Text>
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
