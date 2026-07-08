import type { DetailsMode, Msg, SectionVisibility } from '../types.js'

import { sectionMode } from './details.js'

/**
 * Visual group a transcript block belongs to. Blocks in the same group render
 * flush; a single blank line opens at each group boundary. So a run of tool
 * trails (or model paragraphs) reads as one section and the eye only catches a
 * gap where the *kind* of content actually changes — the boundary between
 * reasoning/tool trails, model prose, and notes/errors.
 *
 *   user   — the human turn (owns its separator + margins in MessageLine)
 *   model  — assistant prose, the model's voice
 *   trail  — reasoning + tool-call trails (the agent's working area)
 *   note   — system notes and errors (a quieter band)
 *   diff   — inline patch segments (an island, owns its own margins)
 *   slash  — slash-command echoes (owns its margin)
 *   intro  — banner / panels (rendered out-of-band, never gapped here)
 */
export type BlockGroup = 'diff' | 'intro' | 'model' | 'note' | 'slash' | 'trail' | 'user'

export const messageGroup = (msg: Pick<Msg, 'kind' | 'role'>): BlockGroup => {
  switch (msg.kind) {
    case 'intro':
    case 'panel':
      return 'intro'
    case 'slash':
      return 'slash'
    case 'diff':
      return 'diff'
    case 'trail':
      return 'trail'
  }

  if (msg.role === 'user') {
    return 'user'
  }

  // Assistant prose is the model's voice; system notes/errors are their own
  // band. (No runtime block uses role 'tool' — tool *results* fold into
  // trails — so a stray 'tool' falls through to the note band harmlessly.)
  return msg.role === 'assistant' ? 'model' : 'note'
}

// Groups whose leading gap is already owned by their own chrome in
// MessageLine (the turn separator + top margin for user, the top margin for
// slash, the top+bottom margins for diff) or that are painted out-of-band
// (intro). The grouping primitive only spaces the model working area —
// model prose, reasoning/tool trails, and notes/errors.
const SELF_SPACED: ReadonlySet<BlockGroup> = new Set(['diff', 'intro', 'slash', 'user'])

// Groups that already paint a trailing blank line beneath themselves
// (marginBottom in MessageLine), so the block that follows must not add its
// own leading gap or the single boundary would become a double gap.
const PAINTS_TRAILING_GAP: ReadonlySet<BlockGroup> = new Set(['diff', 'user'])

/**
 * Whether `cur` renders one blank line above it, given the block rendered
 * directly above it (`prev`). True only where the visual group changes, and
 * only for the model-working-area bands (model / trail / note) — user, slash,
 * diff, and intro keep their existing spacing.
 *
 * Streaming-safe by construction: the result depends on the *predecessor's*
 * group, never on `cur`'s own (live, changing) content. The actively-streaming
 * assistant block therefore computes the same gap while it streams as the
 * settled segment does once it flushes, so the live area never jumps.
 */
export const hasLeadGap = (
  prev: Pick<Msg, 'kind' | 'role'> | undefined,
  cur: Pick<Msg, 'kind' | 'role'>
): boolean => {
  const group = messageGroup(cur)

  if (SELF_SPACED.has(group)) {
    return false
  }

  if (!prev) {
    return false
  }

  const prevGroup = messageGroup(prev)

  return prevGroup !== group && !PAINTS_TRAILING_GAP.has(prevGroup)
}

export interface DetailsCtx {
  commandOverride?: boolean
  detailsMode: DetailsMode
  sections?: SectionVisibility
}

const trailAllHidden = (ctx: DetailsCtx): boolean =>
  sectionMode('thinking', ctx.detailsMode, ctx.sections, ctx.commandOverride) === 'hidden' &&
  sectionMode('tools', ctx.detailsMode, ctx.sections, ctx.commandOverride) === 'hidden' &&
  sectionMode('activity', ctx.detailsMode, ctx.sections, ctx.commandOverride) === 'hidden'

/**
 * Whether a settled transcript block paints anything. A trail renders nothing
 * when it has no reasoning/tools/todos to show (e.g. the finalDetails segment
 * that carries only a token tally) or when every section it does have is hidden
 * (`/details hidden`); every other block draws at least one row. A block that
 * renders nothing is *transparent* to grouping: the block below it draws its
 * boundary against the nearest visible block instead (see prevRenderedMsg), so
 * a hidden or content-less trail never leaves a floating blank line, doubles
 * the gap after a user prompt, or pads the space above the final reply. In the
 * default/collapsed modes content-bearing trails always render, so this is a
 * no-op there.
 */
export const blockRenders = (msg: Pick<Msg, 'kind' | 'thinking' | 'todos' | 'tools'>, ctx: DetailsCtx): boolean => {
  if (msg.kind !== 'trail') {
    return true
  }

  if (msg.todos?.length) {
    return true
  }

  if (!(msg.tools?.length || msg.thinking?.trim())) {
    return false
  }

  return !trailAllHidden(ctx)
}

/**
 * The nearest block above `index` that actually renders, resolved through a
 * lazy accessor so it works over either the virtualized history rows or the
 * live block list. This is the grouping predecessor — using it (instead of the
 * literal previous row) keeps hidden trails from interrupting the rhythm.
 */
export const prevRenderedMsg = (
  msgAt: (i: number) => Msg | undefined,
  index: number,
  ctx: DetailsCtx
): Msg | undefined => {
  for (let i = index - 1; i >= 0; i--) {
    const candidate = msgAt(i)

    if (candidate && blockRenders(candidate, ctx)) {
      return candidate
    }
  }

  return undefined
}
