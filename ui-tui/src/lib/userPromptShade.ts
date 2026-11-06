import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

/** A prompt the user typed (not a slash command echo, diff, or other kind). */
export const isUserPrompt = (msg: Msg): boolean => msg.role === 'user' && !msg.kind

/** Rows the shade bar adds around a user prompt: one shaded row above and below. */
export const USER_PROMPT_SHADE_ROWS = 2

/** Whether `msg` gets the shade bar (`display.tui_user_message_shade`, off by default). */
export const hasUserPromptShade = (msg: Msg, enabled: boolean | undefined): boolean =>
  enabled === true && isUserPrompt(msg)

/**
 * Box props that put the user's own prompts on a full-width shade bar (with
 * one shaded row above and below), so turns are easy to find when scrolling
 * back. Spread onto the message's outer Box; empty for every other row and
 * whenever the feature is off.
 */
export const userPromptShade = (
  msg: Msg,
  t: Theme,
  enabled: boolean | undefined
): { backgroundColor?: string; paddingY?: number } =>
  hasUserPromptShade(msg, enabled) ? { backgroundColor: t.color.userMessageBg, paddingY: 1 } : {}
