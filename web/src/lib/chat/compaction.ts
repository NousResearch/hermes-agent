/**
 * Context-compaction handoff detection / splitting.
 *
 * `agent/context_compressor.py` persists compaction summaries as a
 * `role="user"` or `role="assistant"` row whose content starts with one of
 * `COMPACTION_PREFIXES`. They're metadata inserted by the compressor, NOT
 * real turns — so rendering them with normal message styling confuses
 * operators scrolling the timeline (#29824).
 *
 * Two shapes hit the WebUI:
 *
 * 1. Standalone summary row:   "[CONTEXT COMPACTION ...]\n<summary>"
 *    → render with the muted "Context handoff" style.
 *
 * 2. Summary merged as a prefix on the first tail message (double-collision
 *    path — `_merge_summary_into_tail`): "[CONTEXT COMPACTION ...]\n---\n
 *    END OF CONTEXT SUMMARY — respond to the message below, not the
 *    summary above ---\n<real reply>". We split back into two visual rows
 *    so the real reply survives as a readable bubble next to the labelled
 *    handoff.
 *
 * Keep `COMPACTION_PREFIXES` and `COMPACTION_END_MARKER` in sync with
 * `SUMMARY_PREFIX` / `LEGACY_SUMMARY_PREFIX` and the merge-into-tail marker
 * in `agent/context_compressor.py`.
 */

/** Prefix patterns the context compressor writes at the start of summary rows. */
export const COMPACTION_PREFIXES = [
  "[CONTEXT COMPACTION — REFERENCE ONLY]",
  "[CONTEXT COMPACTION - REFERENCE ONLY]",
  "[CONTEXT SUMMARY]:",
] as const;

/**
 * Marker the compressor inserts between a merged summary and the original
 * tail message content.
 */
export const COMPACTION_END_MARKER =
  "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---";

export interface CompactionSplit {
  /** Summary text (header + body, without the end marker). */
  summary: string;
  /** Original message content that came after the end marker (may be empty). */
  remainder: string;
}

/**
 * If `content` begins with a compaction prefix, split out the merged-tail
 * remainder (if any) so the WebUI can render the summary as a labelled
 * handoff and the original reply as a separate bubble.
 *
 * Returns `null` when the content is not a compaction row at all.
 */
export function splitCompactionContent(content: string): CompactionSplit | null {
  const head = content.trimStart();
  if (!COMPACTION_PREFIXES.some((p) => head.startsWith(p))) return null;

  const markerIdx = content.indexOf(COMPACTION_END_MARKER);
  if (markerIdx < 0) {
    return { summary: content, remainder: "" };
  }
  return {
    summary: content.slice(0, markerIdx),
    remainder: content
      .slice(markerIdx + COMPACTION_END_MARKER.length)
      .replace(/^\s+/, ""),
  };
}

/**
 * Cheap prefix check, useful when callers only need to know whether a row
 * is a compaction handoff (vs. actually splitting it). Pure prefix match
 * against `COMPACTION_PREFIXES` after `trimStart()`.
 */
export function isCompactionContent(content: string): boolean {
  const head = content.trimStart();
  return COMPACTION_PREFIXES.some((p) => head.startsWith(p));
}