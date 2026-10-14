import { Box, Text, useInput, wrapAnsi } from '@hermes/ink'
import { useEffect, useState } from 'react'

import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { clarifyAnswerText, clarifyRevisitState } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq, ConfirmReq } from '../types.js'

import { chipRowProps } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const APPROVAL_OPTS = ['once', 'session', 'always', 'deny'] as const
// The backend forbids a permanent allow for this prompt, so drop "always".
const APPROVAL_OPTS_NO_ALWAYS = APPROVAL_OPTS.filter(o => o !== 'always')
const APPROVAL_OPTS_SMART_DENY = ['once', 'deny'] as const

const approvalLabels = (): Record<ApprovalChoice, string> => {
  const p = messages().prompt.approval

  return { always: p.always, deny: p.deny, once: p.once, session: p.session }
}

const CMD_PREVIEW_LINES = 10

type ApprovalChoice = 'always' | 'deny' | 'once' | 'session'

export function approvalOptions(req: ApprovalReq): readonly ApprovalChoice[] {
  if (req.choices) {
    const choices = req.choices.filter((choice): choice is ApprovalChoice =>
      APPROVAL_OPTS.includes(choice as ApprovalChoice)
    )

    if (choices.length) {
      return choices
    }
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

  if (key.return && opts[sel]) {
    return { kind: 'choose', choice: opts[sel] }
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
  const T = useT()
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
        {T.prompt.approval.title} · {req.description}
      </Text>

      <Box flexDirection="column" paddingLeft={1}>
        {shown.map((line, i) => (
          <Text color={t.color.text} key={i} wrap="truncate-end">
            {line || ' '}
          </Text>
        ))}

        {overflow > 0 ? <Text color={t.color.muted}>{T.prompt.approval.moreLines(overflow)}</Text> : null}
      </Box>

      <Text />

      {opts.map((o, i) => (
        <Text key={o}>
          <Text color={t.color.muted} {...chipRowProps(t, sel === i)}>
            {sel === i ? '▸ ' : '  '}
            {i + 1}. {approvalLabels()[o]}
          </Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · 1-{opts.length} quick pick · Esc/Ctrl+C deny</Text>
    </Box>
  )
}

export function ClarifyPrompt({ cols = 80, onCancel, onQuestionAnswer, req, t }: ClarifyPromptProps) {
  const T = useT()
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const [picked, setPicked] = useState<string[]>([])
  const questions = req.questions

  // ── Batch (A-compact) state: status list + one expanded active question.
  // `active` walks the QUESTION list (Tab/Shift-Tab cycle it, any order);
  // `sel` is reused as the cursor within the active question's choice rows.
  const answers = req.answers ?? {}
  const firstUnanswered = questions.findIndex(q => answers[q.qid] === undefined)
  const [active, setActive] = useState(Math.max(0, firstUnanswered))

  const moveActive = (delta: number) => {
    const next = (active + delta + questions.length) % questions.length
    const question = questions[next]

    // Re-visit restore, same model as the CLI panel: a choice answer puts
    // the cursor back on its row; a typed answer lands on Other with the
    // text staged so Enter edits it instead of retyping.
    const restored = clarifyRevisitState(
      question?.choices ?? [],
      question ? answers[question.qid] : undefined,
      question?.multiSelect
    )

    setActive(next)
    setSel(restored.sel)
    setCustom(restored.custom)
    setTyping(false)
    setPicked(restored.picked)
  }

  // After a lock the overlay is re-patched with the new answers map — jump
  // the cursor to the next unanswered question (stay put when editing).
  useEffect(() => {
    const current = questions[active]

    if (current && answers[current.qid] === undefined) {
      return
    }

    const next = questions.findIndex(q => answers[q.qid] === undefined)

    if (next >= 0) {
      setActive(next)
      setSel(0)
      setCustom('')
      setTyping(false)
      setPicked([])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the answers map only
  }, [req.answers])

  const activeQuestion = questions[active]
  const activeChoices = activeQuestion?.choices ?? []
  const multi = activeQuestion?.multiSelect ?? false
  const answeredCount = questions.filter(q => answers[q.qid] !== undefined).length
  const remainingCount = questions.length - answeredCount

  const lockActive = (value: string) => {
    if (activeQuestion) {
      onQuestionAnswer(activeQuestion.qid, value)
      setSel(0)
      setCustom('')
      setTyping(false)
      setPicked([])
    }
  }

  const lockPicked = (extra: string[]) => {
    const values = [...picked, ...extra.map(v => v.trim()).filter(Boolean)]

    lockActive(values.length ? JSON.stringify(values) : '')
  }

  const togglePick = (choice: string) =>
    setPicked(current => (current.includes(choice) ? current.filter(v => v !== choice) : [...current, choice]))

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
      } else if (multi) {
        lockPicked(picked.length ? [] : [activeChoices[sel]!])
      } else if (activeChoices[sel]) {
        lockActive(activeChoices[sel]!)
      }

      return
    }

    if (multi && ch === ' ' && sel < activeChoices.length) {
      togglePick(activeChoices[sel]!)

      return
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= activeChoices.length) {
      if (multi) {
        togglePick(activeChoices[n - 1]!)
      } else {
        lockActive(activeChoices[n - 1]!)
      }
    }
  })

  const enterAction = remainingCount === 1 ? T.prompt.clarify.confirmAndContinue : T.prompt.clarify.lockAnswer

  const hint = typing
    ? T.prompt.clarify.typingHint(enterAction)
    : multi
      ? `${T.prompt.clarify.toggle} · ${T.prompt.clarify.hint(enterAction)}`
      : T.prompt.clarify.hint(enterAction)

  return (
    <Box flexDirection="column">
      <Text bold>
        <Text color={t.color.accent}>ask</Text>
        <Text color={t.color.text}>
          {' '}
          {questions.length === 1
            ? T.session.main.clarifyQuestionsOne('1')
            : T.session.main.clarifyQuestionsOther(String(questions.length))}
        </Text>
      </Text>

      {questions.map((q, i) => {
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
                  {answer ? clarifyAnswerText(answer, q.multiSelect) : T.prompt.clarify.skipped}
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
                    onSubmit={value => (multi ? lockPicked([value]) : lockActive(value))}
                    value={custom}
                  />
                </Box>
              ) : (
                <Box flexDirection="column" paddingLeft={2}>
                  {[...activeChoices, T.prompt.clarify.other].map((c, ci) => (
                    <Text key={ci}>
                      <Text color={t.color.muted} {...chipRowProps(t, sel === ci)}>
                        {sel === ci ? '▸ ' : '  '}
                        {multi && ci < activeChoices.length ? (picked.includes(c) ? '[x] ' : '[ ] ') : ''}
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
        {answeredCount}/{questions.length} answered · {hint}
      </Text>
    </Box>
  )
}

export function ConfirmPrompt({ onCancel, onConfirm, req, t }: ConfirmPromptProps) {
  const T = useT()
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
    { color: t.color.text, label: req.cancelLabel ?? T.prompt.confirm.cancel },
    { color: req.danger ? t.color.error : t.color.text, label: req.confirmLabel ?? T.prompt.confirm.confirm }
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

      <Text color={t.color.muted}>{T.prompt.confirm.hint}</Text>
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
  onCancel: () => void
  onQuestionAnswer: (qid: string, s: string) => void
  req: ClarifyReq
  t: Theme
}

interface ConfirmPromptProps {
  onCancel: () => void
  onConfirm: () => void
  req: ConfirmReq
  t: Theme
}
