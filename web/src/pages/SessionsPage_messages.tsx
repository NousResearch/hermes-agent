import type { SessionMessage } from '@/lib/api'

/** Backend display typing the transcript view honours (``hermes_cli/web_routers/sessions.py``
 *  ``_project_for_display``): ``hidden`` rows are plumbing (a silent Kanban wake turn, a bare
 *  NO_REPLY); a self-injected wake or async-result card is a system notice, never the user. */
type DisplayTypedMessage = SessionMessage & { display_kind?: string }

const SYSTEM_NOTICE_KINDS = new Set(['internal_notification', 'async_delegation_complete', 'process_complete'])

export function displayedSessionMessage(message: SessionMessage): SessionMessage | null {
  const kind = (message as DisplayTypedMessage).display_kind
  if (kind === 'hidden') return null
  return kind && SYSTEM_NOTICE_KINDS.has(kind) ? { ...message, role: 'system' } : message
}

// Context-compaction handoff blocks are persisted as ``role="user"`` or
// ``role="assistant"`` with content starting with one of these prefixes —
// they're metadata inserted by ``agent/context_compressor.py``, NOT real
// turns the user typed or the model replied with. Rendering them with
// the same styling as regular messages confuses operators scrolling the
// session timeline (#29824 — "WebUI can show context compaction block
// instead of latest assistant response after compression"), so we
// detect them here and downgrade them to a muted, clearly-labelled
// "Context handoff" row.
//
// Keep these prefixes (and the END marker below) in sync with
// ``SUMMARY_PREFIX`` / ``LEGACY_SUMMARY_PREFIX`` and the
// merge-into-tail marker in ``agent/context_compressor.py``.
const COMPACTION_PREFIXES = [
  '[CONTEXT COMPACTION — REFERENCE ONLY]',
  '[CONTEXT COMPACTION - REFERENCE ONLY]',
  '[CONTEXT SUMMARY]:'
] as const

// Marker the compressor inserts between a merged summary and the
// original tail message content. When the summary role would collide
// with both head and tail roles (e.g. head ends with ``user`` and tail
// starts with ``assistant``), the compressor merges the summary as a
// prefix on the first tail message instead of inserting a standalone
// row. We split on this marker so the WebUI still shows the original
// assistant reply as its own readable bubble — otherwise the merged
// row reads as a single opaque "Context compaction" block and the
// user can't see the reply (#29824).
const COMPACTION_END_MARKER = '--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---'

interface CompactionSplit {
  /** Summary text (header + body, without the end marker). */
  summary: string
  /** Original message content that came after the end marker. */
  remainder: string
}

export function splitCompactionContent(content: string): CompactionSplit | null {
  const head = content.trimStart()
  if (!COMPACTION_PREFIXES.some(p => head.startsWith(p))) return null
  const markerIdx = content.indexOf(COMPACTION_END_MARKER)
  if (markerIdx < 0) {
    return { summary: content, remainder: '' }
  }
  return {
    summary: content.slice(0, markerIdx),
    remainder: content.slice(markerIdx + COMPACTION_END_MARKER.length).replace(/^\s+/, '')
  }
}
