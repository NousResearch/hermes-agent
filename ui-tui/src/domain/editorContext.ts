/**
 * Last-reply context block for the external-editor composer — the Ink composer's half of
 * `display.editor_include_last_reply`.
 *
 * Mirrors `hermes_cli/editor_context.py` (the classic CLI calls that module directly). Both
 * suites read `tests/fixtures/editor_context_vectors.json`, so the two interfaces cannot drift on
 * what counts as the block or on which lines survive the strip.
 *
 * Only the marked region is removed. A draft's own `#` lines — markdown headings, shell comments,
 * code fences — are the user's text and stay.
 */

export const MARKER_START = '# ── last reply (lines starting with # are removed) ──'
export const MARKER_END = '# ── end of last reply ──'

const TRUTHY = new Set(['1', 'yes', 'true', 'on'])

/** One reply line as a comment; markers inside a reply are defanged so they cannot close early. */
function commentLine(line: string): string {
  if (!line) {
    return '#'
  }

  const stripped = line.trim()

  if (stripped === MARKER_START || stripped === MARKER_END) {
    return `# ${stripped.replace(/──/g, '--')}`
  }

  return `# ${line}`
}

/** The marker block for a reply, or '' when there is nothing to show. */
export function buildLastReplyBlock(lastReply: string): string {
  const text = (lastReply ?? '').replace(/^\n+|\n+$/g, '')

  if (!text.trim()) {
    return ''
  }

  const body = text.split('\n').map(line => commentLine(line.replace(/\r$/, '')))

  return `${MARKER_START}\n${body.join('\n')}\n${MARKER_END}`
}

/** What the editor opens with: the draft, then the reply block (when there is one). */
export function buildEditorSeed(draft: string, lastReply = ''): string {
  const block = buildLastReplyBlock(lastReply)
  const body = draft ?? ''

  if (!block) {
    return body
  }

  if (!body.trim()) {
    return block
  }

  return `${body.replace(/\n+$/, '')}\n\n${block}`
}

/**
 * Remove the marker block (and the blank separator above it) from an editor buffer.
 *
 * A user who deleted the end marker still gets the block stripped — to the end of the buffer —
 * because leaving it in would send the agent's own reply back to it as a prompt.
 */
export function stripLastReplyBlock(text: string): string {
  const lines = (text ?? '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')

  const startIndex = lines.findIndex(line => line.trim() === MARKER_START)

  if (startIndex === -1) {
    return text ?? ''
  }

  let endIndex = -1

  for (let i = startIndex + 1; i < lines.length; i++) {
    if (lines[i]!.trim() === MARKER_END) {
      endIndex = i

      break
    }
  }

  if (endIndex === -1) {
    endIndex = lines.length - 1
  }

  // Drop the single blank separator line the builder inserted above the block — but only when the
  // block runs to the end of the buffer. Text the user typed below the block keeps its spacing.
  let start = startIndex

  if (endIndex === lines.length - 1 && start > 0 && !lines[start - 1]!.trim()) {
    start -= 1
  }

  return [...lines.slice(0, start), ...lines.slice(endIndex + 1)].join('\n')
}

/** Seed text for `$EDITOR` under the `display.editor_include_last_reply` setting. */
export function editorSeed(draft: string, lastReply: string, enabled: boolean): string {
  return enabled ? buildEditorSeed(draft, lastReply) : (draft ?? '')
}

/** Read the setting the way the CLI reads YAML booleans. */
export function editorReplyContextEnabled(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value
  }

  if (typeof value === 'number') {
    return value !== 0
  }

  return typeof value === 'string' && TRUTHY.has(value.trim().toLowerCase())
}
