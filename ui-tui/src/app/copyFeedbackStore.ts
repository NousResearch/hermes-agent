import { atom } from 'nanostores'

import { getUiState } from './uiStore.js'

export type CopyDelivery = 'clipboard' | 'terminal'

export const $copyFeedback = atom<CopyDelivery | null>(null)

let requestId = 0
let timer: ReturnType<typeof setTimeout> | undefined

export function clearCopyFeedback() {
  requestId += 1
  clearTimeout(timer)
  timer = undefined
  $copyFeedback.set(null)
}

/** Confirm only the latest completed copy in this session; never claim a remote clipboard acknowledgement. */
export async function runCopyWithFeedback(copy: () => Promise<CopyDelivery | null>): Promise<void> {
  clearCopyFeedback()
  const id = requestId
  const sid = getUiState().sid

  try {
    const delivery = await copy()

    if (!delivery || id !== requestId || sid !== getUiState().sid) {
      return
    }

    $copyFeedback.set(delivery)
    timer = setTimeout(clearCopyFeedback, 2400)
    timer.unref?.()
  } catch {
    // A failed copy must not leave a success confirmation from an earlier selection.
  }
}
