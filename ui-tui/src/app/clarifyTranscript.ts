import { t } from '../i18n/runtime.js'
import { buildToolTrailLine, clarifyAnswerText, formatAbandonedClarify } from '../lib/text.js'

export interface ClarifyTranscriptQuestion {
  multiSelect?: boolean
  qid: string
  question: string
}

export const clarifyQuestionCountLabel = (count: number) =>
  t(count === 1 ? 'session.main.clarifyQuestionsOne' : 'session.main.clarifyQuestionsOther', String(count))

/** The transcript a settled clarify leaves behind, as main's TUI wrote it: the collapsed tool row
 * `Clarify("N question(s)")` and one user bubble of `question → answer` lines (a blank answer reads
 * as skipped). The canonical one-card path and the batch path both settle through this. */
export function clarifySettledRows(
  questions: ClarifyTranscriptQuestion[],
  answers: Record<string, string | undefined>
) {
  return {
    trail: buildToolTrailLine('clarify', clarifyQuestionCountLabel(questions.length)),
    userText: questions
      .map(
        q =>
          `${q.question} → ${answers[q.qid]?.trim() ? clarifyAnswerText(answers[q.qid]!, q.multiSelect) : t('session.main.skipped')}`
      )
      .join('\n')
  }
}

/** The clarify tool's stored result (`tools/clarify_tool.py::_result`), as much of it as the transcript reads. */
interface StoredClarifyResult {
  notice?: string
  outcome?: string
  responses?: { question?: string; user_response?: null | string | string[] }[]
}

/** The reason line the live turn wrote when a form ended without being submitted. */
const abandonedReason = (outcome: string, notice?: string) =>
  outcome === 'cancelled'
    ? 'cancelled'
    : outcome === 'timed_out'
      ? t('gatewayMsg.clarify.timedOut')
      : (notice ?? outcome)

/** A stored clarify result on resume, shown as the live turn showed it: a submitted form as the settled trail and
 * `question → answer` bubble (`abandoned: false`), a cancelled, timed-out or undelivered one as its system line.
 * Null when the result has no responses. */
export function storedClarifyRows(content: string) {
  let result: StoredClarifyResult | null

  try {
    // SAFETY: `content` is the clarify tool's own JSON result; a row of another shape only loses its bubble.
    result = JSON.parse(content) as StoredClarifyResult | null
  } catch {
    return null
  }

  const responses = result?.responses

  if (!Array.isArray(responses) || !responses.length) {
    return null
  }

  const rows = responses.map((row, index) => ({
    answer: Array.isArray(row?.user_response) ? row.user_response.join(', ') : (row?.user_response ?? undefined),
    qid: `q${index}`,
    question: row?.question ?? ''
  }))

  const answers = Object.fromEntries(rows.map(row => [row.qid, row.answer]))
  const settled = clarifySettledRows(rows, answers)
  const outcome = result?.outcome ?? 'submitted'

  if (outcome === 'submitted') {
    return { abandoned: false, text: settled.userText, trail: settled.trail }
  }

  return {
    abandoned: true,
    text: formatAbandonedClarify(
      rows,
      Object.fromEntries(rows.map(row => [row.qid, row.answer ?? ''])),
      abandonedReason(outcome, result?.notice)
    ),
    trail: settled.trail
  }
}
