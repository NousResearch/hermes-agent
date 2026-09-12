import type { ToolTrailPosition } from '../types.js'

const POSITIONS = ['above', 'below'] as const

// Where a message's own reasoning/tool trail sits relative to the text it
// belongs to. Orthogonal to `display.sections` visibility: position answers
// "in what order", the section modes answer "shown at all, and how far open".
// A trail hidden by `/details tools hidden` stays hidden in both positions.
//
//   above — today's transcript shape: trail, "Response" separator, answer.
//           The turn reads chronologically, which is what you want while a
//           turn is in flight.
//   below — answer first, trail underneath. A long tool trail no longer
//           pushes the answer away from the prompt that asked for it, which
//           is what you want scrolling back through a finished session.
//
// `above` is the default; it is what every existing session already renders.
export const DEFAULT_TOOL_TRAIL_POSITION: ToolTrailPosition = 'above'

/**
 * Read `display.tool_trail_position` off a raw (hand-editable) config value.
 * Anything that is not exactly `above`/`below` — a typo, a boolean, a missing
 * key — falls back to the default rather than throwing, matching how
 * `parseDetailsMode` treats unknown modes.
 */
export const parseToolTrailPosition = (v: unknown): ToolTrailPosition =>
  POSITIONS.find(
    p =>
      p ===
      String(v ?? '')
        .trim()
        .toLowerCase()
  ) ?? DEFAULT_TOOL_TRAIL_POSITION

export const nextToolTrailPosition = (p: ToolTrailPosition): ToolTrailPosition => (p === 'above' ? 'below' : 'above')
