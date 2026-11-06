import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

/** A prompt the user typed (not a slash command echo, diff, or other kind). */
export const isUserPrompt = (msg: Msg): boolean => msg.role === 'user' && !msg.kind

/** Rows the shade bar adds around a user prompt: one shaded row above and below. */
export const USER_PROMPT_SHADE_ROWS = 2

/**
 * Box props that put the user's own prompts on a full-width shade bar (with
 * one shaded row above and below), so turns are easy to find when scrolling
 * back. Spread onto the message's outer Box; empty for every other row.
 */
export const userPromptShade = (msg: Msg, t: Theme): { backgroundColor?: string; paddingY?: number } =>
  isUserPrompt(msg) ? { backgroundColor: t.color.userMessageBg, paddingY: 1 } : {}
