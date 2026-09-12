import { atom } from 'nanostores'

import { buildFreeTierLimitSections, FREE_TIER_LIMIT_KEY, FREE_TIER_LIMIT_TITLE } from '../content/setup.js'
import type { PanelSection } from '../types.js'

import { patchOverlayState } from './overlayStore.js'
import { getUiState } from './uiStore.js'

// Separate from the single status-line notice: sign-in progress must not release
// queued inference, and switching chats must not erase another chat's refusal.
export const $freeTierBlocks = atom<Record<string, string>>({})

export function freeTierBlockMessage(): string | undefined {
  const { sid, notice } = getUiState()
  return (sid ? $freeTierBlocks.get()[sid] : undefined) || (notice?.key === FREE_TIER_LIMIT_KEY ? notice.text : undefined)
}

export function setFreeTierBlock(message: string | null): void {
  const { sid } = getUiState()
  if (!sid) return
  const next = { ...$freeTierBlocks.get() }
  if (message) next[sid] = message
  else delete next[sid]
  $freeTierBlocks.set(next)
}

/** The backend owns the gate; this only presents its notice after the turn settles. */
export function createFreeTierGatePresenter(
  panel: (title: string, sections: PanelSection[]) => void,
  onSignIn: () => void,
  onSetup: () => void
) {
  let shownFor: string | null | undefined

  return {
    clear() {
      shownFor = undefined
    },
    show() {
      const { busy, sid } = getUiState()
      const message = freeTierBlockMessage()

      if (busy || !message || shownFor === sid) {
        return
      }

      shownFor = sid
      panel(FREE_TIER_LIMIT_TITLE, buildFreeTierLimitSections(message))
      patchOverlayState({ modelPicker: { continuationMessage: message, onSignIn, onSetup } })
    }
  }
}
