import { sessionTabShortcut } from '../domain/sessionTabs.js'
import { isVoiceToggleKey } from '../lib/platform.js'

import type { InputHandlerContext } from './interfaces.js'

/** Navigation stays ahead of editor key handling, without stealing voice chords. */
export function handleSessionNavigationInput(
  ch: string,
  key: Parameters<typeof isVoiceToggleKey>[0] & { upArrow?: boolean; downArrow?: boolean },
  event: object,
  ctx: InputHandlerContext
): boolean {
  const tabAction = sessionTabShortcut(ch, key)

  if (tabAction && ctx.actions.sessionTabAction && !isVoiceToggleKey(key, ch, ctx.voice.recordKey)) {
    // Runtime Ink events inherit Event; its public declaration omits this method.
    ;(event as { stopImmediatePropagation: () => void }).stopImmediatePropagation()
    ctx.actions.sessionTabAction(tabAction)

    return true
  }

  const { state, actions } = ctx.composer

  if (state.completions.length && state.input && state.historyIdx === null && (key.upArrow || key.downArrow)) {
    const len = state.completions.length
    actions.setCompIdx(i => (key.upArrow ? (i - 1 + len) % len : (i + 1) % len))

    return true
  }

  return false
}
