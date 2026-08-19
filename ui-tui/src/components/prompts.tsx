import { Box, Text, useInput, wrapAnsi } from '@hermes/ink'
import { useEffect, useState } from 'react'

import { isMac } from '../lib/platform.js'
import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq, ConfirmReq } from '../types.js'

import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const APPROVAL_OPTS = ['once', 'session', 'always', 'deny'] as const
// tirith warning present → backend downgrades "always" to session scope, so drop it.
const APPROVAL_OPTS_NO_ALWAYS = APPROVAL_OPTS.filter(o => o !== 'always')
const APPROVAL_OPTS_SMART_DENY = ['once', 'deny'] as const
const LABELS = { always: 'Always allow', deny: 'Deny', once: 'Allow once', session: 'Allow this session' } as const
const CMD_PREVIEW_LINES = 10

type ApprovalChoice = 'always' | 'deny' | 'once' | 'session'

export function approvalOptions(req: ApprovalReq): readonly ApprovalChoice[] {
  if (req.choices) {
    return req.choices.filter((choice): choice is ApprovalChoice => APPROVAL_OPTS.includes(choice as ApprovalChoice))
  }

  if (req.smartDenied) {
    return APPROVAL_OPTS_SMART_DENY
  }

  return req.allowPermanent === false ? APPROVAL_OPTS_NO_ALWAYS : APPROVAL_OPTS
}

type ApprovalKey = {
  downArrow?: boolean
  escape?: boolean
  return?: boolean
  upArrow?: boolean
}

type ApprovalAction = { kind: 'choose'; choice: ApprovalChoice } | { kind: 'move'; delta: -1 | 1 } | { kind: 'noop' }

/**
 * Pure key-dispatch for the approval prompt — exported so the regression
 * matrix (Esc, Ctrl+C-equivalent, number keys, Enter, ↑↓) is testable
 * without mounting React + Ink + a fake stdin.  The component just maps the
 * action onto its own state setters.
 *
 * Esc and number keys both terminate the prompt; Esc maps to deny (parity
 * with the global Ctrl+C handler that already calls cancelOverlayFromCtrlC
 * for approvals).  Numbers 1..opts.length pick the labelled choice.  Enter
 * confirms the current selection.  ↑/↓ moves the selection within bounds.
 */
export function approvalAction(
  ch: string,
  key: ApprovalKey,
  sel: number,
  opts: readonly ApprovalChoice[] = APPROVAL_OPTS
): ApprovalAction {
  if (key.escape) {
    return { kind: 'choose', choice: 'deny' }
  }

  const n = parseInt(ch, 10)

  if (n >= 1 && n <= opts.length) {
    return { kind: 'choose', choice: opts[n - 1]! }
  }

  if (key.return) {
    return { kind: 'choose', choice: opts[sel]! }
  }

  if (key.upArrow && sel > 0) {
    return { kind: 'move', delta: -1 }
  }

  if (key.downArrow && sel < opts.length - 1) {
    return { kind: 'move', delta: 1 }
  }

  return { kind: 'noop' }
}

export function ApprovalPrompt({ cols = 80, onChoice, req, t }: ApprovalPromptProps) {
  const [sel, setSel] = useState(0)
  const opts = approvalOptions(req)

  useInput((ch, key) => {
    const action = approvalAction(ch, key, sel, opts)

    if (action.kind === 'choose') {
      onChoice(action.choice)
    } else if (action.kind === 'move') {
      setSel(s => s + action.delta)
    }
  })

  // Wrap long single-line commands to the panel width instead of clipping the
  // tail (mirrors the CLI approval panel fix — the full command must be
  // reviewable before approving). Border + paddingX + inner padding ≈ 8 cols.
  const innerWidth = Math.max(20, cols - 8)

  const rawLines = req.command
    .split('\n')
    .flatMap(line => wrapAnsi(line, innerWidth, { hard: true, trim: false }).split('\n'))

  const shown = rawLines.slice(0, CMD_PREVIEW_LINES)
  const overflow = rawLines.length - shown.length

  return (
    <Box borderColor={t.color.warn} borderStyle="double" flexDirection="column" paddingX={1}>
      <Text bold color={t.color.warn}>
        ⚠ approval required · {req.description}
      </Text>

      <Box flexDirection="column" paddingLeft={1}>
        {shown.map((line, i) => (
          <Text color={t.color.text} key={i} wrap="truncate-end">
            {line || ' '}
          </Text>
        ))}

        {overflow > 0 ? (
          <Text color={t.color.muted}>
            … +{overflow} more line{overflow === 1 ? '' : 's'} (full text above)
          </Text>
        ) : null}
      </Box>

      <Text />

      {opts.map((o, i) => (
        <Text key={o}>
          <Text color={t.color.muted} {...chipRowProps(t, sel === i)}>
            {sel === i ? '▸ ' : '  '}
            {i + 1}. {LABELS[o]}
          </Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · 1-{opts.length} quick pick · Esc/Ctrl+C deny</Text>
    </Box>
  )
}

export function ClarifyPrompt({ cols = 80, onAnswer, onCancel, onQuestionAnswer, req, t }: ClarifyPromptProps) {
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []
  const batch = req.questions ?? []
  const isBatch = batch.length > 0

  // ── Batch (A-compact) state: status list + one expanded active question.
  // `active` walks the QUESTION list (Tab/Shift-Tab cycle it, any order);
  // `sel` is reused as the cursor within the active question's choice rows.
  const answers = req.answers ?? {}
  const firstUnanswered = batch.findIndex(q => answers[q.qid] === undefined)
  const [active, setActive] = useState(Math.max(0, firstUnanswered))

  const moveActive = (delta: number) => {
    setActive(a => (a + delta + batch.length) % batch.length)
    setSel(0)
    setCustom('')
    setTyping(false)
  }

  // After a lock the overlay is re-patched with the new answers map — jump
  // the cursor to the next unanswered question (stay put when editing).
  useEffect(() => {
    if (!isBatch) {
      return
    }

    const current = batch[active]

    if (current && answers[current.qid] === undefined) {
      return
    }

    const next = batch.findIndex(q => answers[q.qid] === undefined)

    if (next >= 0) {
      setActive(next)
      setSel(0)
      setCustom('')
      setTyping(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the answers map only
  }, [req.answers])

  const heading = (
    <Text bold>
      <Text color={t.color.accent}>ask</Text>
      <Text color={t.color.text}> {isBatch ? `${batch.length} questions` : req.question}</Text>
    </Text>
  )

  const activeQuestion = isBatch ? batch[active] : undefined
  const activeChoices = activeQuestion ? (activeQuestion.choices ?? []) : choices
  const answeredCount = isBatch ? batch.filter(q => answers[q.qid] !== undefined).length : 0
  const remainingCount = isBatch ? batch.length - answeredCount : 0

  const lockActive = (value: string) => {
    if (activeQuestion) {
      onQuestionAnswer?.(activeQuestion.qid, value)
      setSel(0)
      setCustom('')
      setTyping(false)
    }
  }

  useInput((ch, key) => {
    if (key.escape) {
      if (typing) {
        setTyping(false)

        return
      }

      onCancel()

      return
    }

    if (typing) {
      return
    }

    if (isBatch) {
      // Tab / Shift-Tab cycle the active question (with wrap) — the
      // selected question is always the expanded one, like the CLI panel.
      if (key.tab) {
        moveActive(key.shift ? -1 : 1)

        return
      }

      if (!activeQuestion) {
        return
      }

      if (activeChoices.length === 0) {
        // Open-ended question: any keypress starts typing (TextInput below).
        setTyping(true)

        return
      }

      if (key.upArrow && sel > 0) {
        setSel(s => s - 1)
      }

      if (key.downArrow && sel < activeChoices.length) {
        setSel(s => s + 1)
      }

      if (key.return) {
        if (sel === activeChoices.length) {
          setTyping(true)
        } else if (activeChoices[sel]) {
          lockActive(activeChoices[sel]!)
        }

        return
      }

      const n = parseInt(ch)

      if (n >= 1 && n <= activeChoices.length) {
        lockActive(activeChoices[n - 1]!)
      }

      return
    }

    if (!choices.length) {
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

  if (isBatch) {
    const hint = typing
      ? `Enter ${remainingCount === 1 ? 'confirm and continue' : 'lock answer'} · Esc back`
      : `↑/↓ select · Enter ${remainingCount === 1 ? 'confirm and continue' : 'lock answer'} · Tab/Shift+Tab switch question · Esc/Ctrl+C cancel`

    return (
      <Box flexDirection="column">
        {heading}

        {batch.map((q, i) => {
          const answer = answers[q.qid]
          const isActive = i === active
          const marker = answer !== undefined ? '✓' : isActive ? '▸' : '·'

          return (
            <Box flexDirection="column" key={q.qid}>
              <Text>
                <Text bold={isActive} color={isActive ? t.color.text : t.color.muted}>
                  {marker} {q.question}
                </Text>
              </Text>

              {answer !== undefined ? (
                // The locked answer on its own line, in the ok color, so the
                // current answers stay readable while Tab walks the list.
                <Box paddingLeft={2}>
                  <Text color={answer ? t.color.ok : t.color.muted} italic={!answer}>
                    {answer || '(skipped)'}
                  </Text>
                </Box>
              ) : null}

              {isActive ? (
                typing || activeChoices.length === 0 ? (
                  <Box paddingLeft={2}>
                    <Text color={t.color.label}>{'> '}</Text>
                    <TextInput
                      color={t.color.text}
                      columns={Math.max(20, cols - 8)}
                      onChange={setCustom}
                      onSubmit={lockActive}
                      value={custom}
                    />
                  </Box>
                ) : (
                  <Box flexDirection="column" paddingLeft={2}>
                    {[...activeChoices, 'Other (type your answer)'].map((c, ci) => (
                      <Text key={ci}>
                        <Text color={t.color.muted} {...chipRowProps(t, sel === ci)}>
                          {sel === ci ? '▸ ' : '  '}
                          {ci + 1}. {c}
                        </Text>
                      </Text>
                    ))}
                  </Box>
                )
              ) : null}
            </Box>
          )
        })}

        <Text color={t.color.muted}>
          {answeredCount}/{batch.length} answered · {hint}
        </Text>
      </Box>
    )
  }

  if (typing || !choices.length) {
    return (
      <Box flexDirection="column">
        {heading}

        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput
            color={t.color.text}
            columns={Math.max(20, cols - 6)}
            onChange={setCustom}
            onSubmit={onAnswer}
            value={custom}
          />
        </Box>

        <Text color={t.color.muted}>
          Enter send · Esc {choices.length ? 'back' : 'cancel'} ·{' '}
          {isMac ? 'Cmd+C copy · Cmd+V paste · Ctrl+C cancel' : 'Ctrl+C cancel'}
        </Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      {heading}

      {[...choices, 'Other (type your answer)'].map((c, i) => (
        <Text key={i}>
          <Text color={t.color.muted} {...chipRowProps(t, sel === i)}>
            {sel === i ? '▸ ' : '  '}
            {i + 1}. {c}
          </Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick · Esc/Ctrl+C cancel</Text>
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
    { color: t.color.text, label: req.cancelLabel ?? 'No' },
    { color: req.danger ? t.color.error : t.color.text, label: req.confirmLabel ?? 'Yes' }
  ]

  return (
    <Box borderColor={accent} borderStyle="double" flexDirection="column" paddingX={1}>
      <Text bold color={accent}>
        {req.danger ? '⚠' : '?'} {req.title}
      </Text>

      {req.detail ? (
        <Box paddingLeft={1}>
          <Text color={t.color.text} wrap="truncate-end">
            {req.detail}
          </Text>
        </Box>
      ) : null}

      <Text />

      {rows.map((row, i) => (
        <Text key={row.label}>
          <Text color={sel === i ? accent : t.color.muted}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? row.color : t.color.muted}>{row.label}</Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · Y/N quick · Esc cancel</Text>
    </Box>
  )
}

interface ApprovalPromptProps {
  cols?: number
  onChoice: (s: string) => void
  req: ApprovalReq
  t: Theme
}

interface ClarifyPromptProps {
  cols?: number
  onAnswer: (s: string) => void
  onCancel: () => void
  /** Batch mode: lock one question's answer (clarify.respond + question_id). */
  onQuestionAnswer?: (qid: string, s: string) => void
  req: ClarifyReq
  t: Theme
}

interface ConfirmPromptProps {
  onCancel: () => void
  onConfirm: () => void
  req: ConfirmReq
  t: Theme
}
