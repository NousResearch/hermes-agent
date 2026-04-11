import { Box, Text, TextInput, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq } from '../types.js'

export function ApprovalPrompt({ onChoice, req, t }: { onChoice: (s: string) => void; req: ApprovalReq; t: Theme }) {
  const [sel, setSel] = useState(3)
  const opts = ['once', 'session', 'always', 'deny'] as const
  const labels = { always: 'Always allow', deny: 'Deny', once: 'Allow once', session: 'Allow this session' } as const

  useInput((ch, key) => {
    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < 3) {
      setSel(s => s + 1)
    }

    if (key.return) {
      onChoice(opts[sel]!)
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
      {opts.map((o, i) => (
        <Text key={o}>
          <Text color={sel === i ? t.color.warn : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            [{o[0]}] {labels[o]}
          </Text>
        </Text>
      ))}
      <Text color={t.color.dim}>↑/↓ select · Enter confirm · o/s/a/d quick pick</Text>
    </Box>
  )
}

export function ClarifyPrompt({ onAnswer, req, t }: { onAnswer: (s: string) => void; req: ClarifyReq; t: Theme }) {
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []

  useInput((ch, key) => {
    if (typing) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < choices.length) {
      setSel(s => s + 1)
    }

    if (key.return) {
      if (sel === choices.length) {
        setTyping(true)
      } else if (choices[sel]) {
        onAnswer(choices[sel]!)
      }
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= choices.length) {
      onAnswer(choices[n - 1]!)
    }
  })

  if (typing || !choices.length) {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>
          ❓ {req.question}
        </Text>
        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput onChange={setCustom} onSubmit={onAnswer} value={custom} />
        </Box>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>
        ❓ {req.question}
      </Text>
      {[...choices, 'Other (type your answer)'].map((c, i) => (
        <Text key={i}>
          <Text color={sel === i ? t.color.label : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            {i + 1}. {c}
          </Text>
        </Text>
      ))}
      <Text color={t.color.dim}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick</Text>
    </Box>
  )
}
