// ── KENSEI CUSTOM: AskUserQuestionsTool ────────────────────────────────
// Per spec (2026-06-04): the new mode system (plan / UltraPlan / recon)
// uses a multi-question batched prompt instead of single ClarifyPrompt.
// Mirrors Claude Code's AskUserQuestion tool: boxed design, numbered
// selectable options, "Recommended" label highlighted, multi-question
// batching for UltraPlan, 4-question upfront for Recon, 0-3 ad-hoc for
// Plan.  Must survive upstream merges — leave existing ClarifyPrompt
// untouched per item 4A.
import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { Theme } from '../theme.js'
import type { AskUserQuestionsReq, AskUserQuestion as Question } from '../types.js'

const ARROW = '▸ '
const SPACER = '  '

/** Index in the choices list that means "Other (free-form follow-up)". */
const OTHER_INDEX_OFFSET = 1 // +1 for the synthetic "Other" option appended below

function initialSelections(qs: Question[]): number[] {
  // Pre-select the recommended option (or 0 if none marked).  The UI
  // always opens with a sensible default so Enter confirms something.
  return qs.map(q => {
    if (!q.options.length) {return OTHER_INDEX_OFFSET} // "Other" by default when no options
    const recIdx = q.options.findIndex(o => o.recommended)

    return recIdx >= 0 ? recIdx : 0
  })
}

function selectionLabel(q: Question, sel: number): string {
  if (sel === q.options.length) {return '__other__'} // sentinel for "Other" handoff
  const opt = q.options[sel]

  if (!opt) {return '__other__'}

  return opt.label
}

function QuestionPanel({
  active,
  idx,
  q,
  selected,
  t,
  total
}: {
  active: boolean
  idx: number
  q: Question
  selected: number
  t: Theme
  total: number
}) {
  const opts = q.options
  // Last row is the synthetic "Other (free-form)" option.  When
  // selected, the model is told to fall back to the legacy `clarify`
  // tool for free-form text (the TUI has no inline text capture).
  const totalChoices = opts.length + OTHER_INDEX_OFFSET

  const borderColor = active ? t.color.accent : t.color.border
  const header = q.header ? q.header.toUpperCase().slice(0, 12) : `Q${idx + 1}`

  return (
    <Box
      borderColor={borderColor}
      borderStyle={active ? 'round' : 'single'}
      flexDirection="column"
      paddingX={1}
    >
      <Box>
        <Text bold color={active ? t.color.accent : t.color.muted}>
          {header}  ·  Question {idx + 1}/{total}
        </Text>
        {q.multiSelect ? (
          <Text color={t.color.muted}>  (multi-select)</Text>
        ) : null}
      </Box>

      <Text bold color={t.color.text}>
        {q.question}
      </Text>

      <Box flexDirection="column" paddingLeft={1} paddingTop={1}>
        {opts.map((opt, i) => {
          const isSel = selected === i
          const isRec = opt.recommended

          return (
            <Text key={i}>
              <Text bold={isSel} color={isSel ? t.color.accent : t.color.text} inverse={isSel}>
                {isSel ? ARROW : SPACER}
                {i + 1}. {opt.label}
              </Text>
              {isRec ? <Text color={t.color.ok}>  (Recommended)</Text> : null}
              {opt.description ? (
                <Text color={t.color.muted}>  {opt.description}</Text>
              ) : null}
            </Text>
          )
        })}
        <Text>
          <Text
            bold={selected === opts.length}
            color={selected === opts.length ? t.color.accent : t.color.muted}
            inverse={selected === opts.length}
          >
            {selected === opts.length ? ARROW : SPACER}
            {opts.length + 1}. Other (free-form follow-up)
          </Text>
        </Text>
      </Box>

      {active ? (
        <Text color={t.color.muted}>
          ↑/↓ select · 1-{totalChoices} quick pick · Enter/Tab confirm · Esc cancel
        </Text>
      ) : null}
    </Box>
  )
}

export function AskUserQuestionsTool({
  onAnswer,
  onCancel,
  req,
  t
}: {
  onAnswer: (answers: Record<number, string>, requestId: string) => void
  onCancel: () => void
  req: AskUserQuestionsReq
  t: Theme
}) {
  const [activeIdx, setActiveIdx] = useState(0)
  // `selections[i]` is the current cursor position within question i.
  // Lifted to the parent so keyboard navigation is consistent across the
  // whole batch and the final `answers` map is built from a single source.
  const [selections, setSelections] = useState<number[]>(() => initialSelections(req.questions))
  const qs = req.questions
  const total = qs.length

  useInput((ch, key) => {
    if (key.escape) {
      return onCancel()
    }

    if (!total) {return}
    const currentQ = qs[activeIdx]

    if (!currentQ) {return}

    const choicesLen = currentQ.options.length + OTHER_INDEX_OFFSET

    if (key.upArrow) {
      setSelections(prev => {
        const next = prev.slice()
        next[activeIdx] = Math.max(0, (next[activeIdx] ?? 0) - 1)

        return next
      })

      return
    }

    if (key.downArrow) {
      setSelections(prev => {
        const next = prev.slice()
        next[activeIdx] = Math.min(choicesLen - 1, (next[activeIdx] ?? 0) + 1)

        return next
      })

      return
    }

    if (key.tab) {
      if (key.shift) {
        if (activeIdx > 0) {setActiveIdx(i => i - 1)}
      } else if (activeIdx === total - 1) {
        // Tab on the last question submits.
        finalize()
      } else {
        setActiveIdx(i => i + 1)
      }

      return
    }

    // Number keys 1-N jump directly to that option (or "Other" if N is
    // past the last real option).  This commits the current question
    // and either advances to the next or finalises on the last one.
    const n = parseInt(ch, 10)

    if (n >= 1 && n <= choicesLen) {
      const idx = n - 1
      setSelections(prev => {
        const next = prev.slice()
        next[activeIdx] = idx

        return next
      })

      if (activeIdx === total - 1) {
        finalize()
      } else {
        setActiveIdx(i => i + 1)
      }

      return
    }

    if (key.return) {
      if (activeIdx === total - 1) {
        finalize()
      } else {
        setActiveIdx(i => i + 1)
      }
    }
  })

  function finalize() {
    const answers: Record<number, string> = {}
    qs.forEach((q, i) => {
      const sel = selections[i] ?? 0
      answers[i] = selectionLabel(q, sel)
    })
    onAnswer(answers, req.requestId)
  }

  if (!total) {
    return (
      <Box borderColor={t.color.warn} borderStyle="round" flexDirection="column" paddingX={1}>
        <Text color={t.color.warn}>AskUserQuestionsTool: no questions provided</Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
      <Text bold color={t.color.accent}>
        {total} question{total === 1 ? '' : 's'} · ↑/↓/Tab to navigate · Enter to confirm · Esc to cancel
      </Text>
      {qs.map((q, i) => (
        <Box flexDirection="column" key={i} marginTop={i === 0 ? 1 : 1}>
          <QuestionPanel
            active={i === activeIdx}
            idx={i}
            q={q}
            selected={selections[i] ?? 0}
            t={t}
            total={total}
          />
        </Box>
      ))}
    </Box>
  )
}
