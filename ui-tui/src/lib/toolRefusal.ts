const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}

const text = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

/**
 * The trail note for a call an approval gate or command guard refused, so it never ran (a denied
 * or timed-out approval, a hardline block, a supervised gateway's lifecycle guard), else ''.
 * terminal marks approval refusals `status: "blocked"`; the error opens with `BLOCKED` (approval)
 * or `Blocked:` (lifecycle and workdir guards); a human decision also carries the one-line
 * `user_summary` (tools/approval.py::_denied), which is what the user should read.
 */
export const toolRefusalNote = (result: unknown): string => {
  const record = asRecord(result)
  const summary = text(record.user_summary)
  const error = text(record.error)

  if (!summary && record.status !== 'blocked' && !/^(?:BLOCKED\b|Blocked:)/.test(error)) {
    return ''
  }

  return summary || error.split('\n', 1)[0] || 'blocked'
}
