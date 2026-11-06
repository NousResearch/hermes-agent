/**
 * Last-reply context block for the external-editor composer — the Ink composer's half of
 * `display.editor_include_last_reply`. Mirrors `hermes_cli/editor_context.py`; both suites read
 * `tests/fixtures/editor_context_vectors.json`, so the two interfaces cannot drift.
 *
 * Only the marked region is removed. A draft's own `#` lines — markdown headings, shell comments,
 * code fences — are the user's text and stay.
 */

export const MARKER_START = '# ── last reply (lines starting with # are removed) ──'
export const MARKER_END = '# ── end of last reply ──'

/** One reply line as a comment; a line spelling a marker is defanged so it cannot end the block. */
function commentLine(line: string): string {
  const text = [MARKER_START, MARKER_END].includes(line.trim()) ? line.trim().replace(/──/g, '--') : line

  return text ? `# ${text}` : '#'
}

/** What the editor opens with: the draft, a blank line, then the reply block (if any). */
export function buildEditorSeed(draft: string, lastReply: string): string {
  const reply = lastReply.replace(/^\n+|\n+$/g, '')

  if (!reply.trim()) {
    return draft
  }

  const body = reply.split('\n').map(line => commentLine(line.replace(/\r$/, '')))

  // Always a blank line above the block, even with no draft: editors open on line 1, and text
  // typed there must not land on the marker line (which would stop the block being found).
  return `${draft.replace(/\n+$/, '')}\n\n${MARKER_START}\n${body.join('\n')}\n${MARKER_END}`
}

/**
 * Remove the marker block from an editor buffer (to the end, if the end marker was deleted). The
 * blank separator above the block goes too, but only when the block runs to the end of the
 * buffer: text typed below it keeps its spacing.
 */
export function stripLastReplyBlock(text: string): string {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  let start = lines.findIndex(line => line.trim() === MARKER_START)

  if (start === -1) {
    return text
  }

  const found = lines.findIndex((line, i) => i > start && line.trim() === MARKER_END)
  const end = found === -1 ? lines.length - 1 : found

  if (end === lines.length - 1 && start > 0 && !lines[start - 1]!.trim()) {
    start -= 1
  }

  return [...lines.slice(0, start), ...lines.slice(end + 1)].join('\n')
}
