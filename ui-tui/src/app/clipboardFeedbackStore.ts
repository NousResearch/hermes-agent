import { atom } from 'nanostores'

import { t } from '../i18n/runtime.js'

import type { Notice } from './interfaces.js'

export const $clipboardFeedback = atom<Notice | null>(null)
let timer: ReturnType<typeof setTimeout> | undefined

/** Count visible characters only after the clipboard operation completes. */
function reportClipboardOperation(text: string, action: 'copy' | 'cut') {
  if (!text) {
    return
  }

  clearTimeout(timer)
  const count = [...new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text)].length

  const key =
    action === 'cut'
      ? count === 1
        ? 'slashCmd.core.copy.cutCharsOne'
        : 'slashCmd.core.copy.cutCharsOther'
      : count === 1
        ? 'slashCmd.core.copy.copiedCharsOne'
        : 'slashCmd.core.copy.copiedCharsOther'

  $clipboardFeedback.set({
    level: 'success',
    text: t(key, String(count))
  })
  timer = setTimeout(() => $clipboardFeedback.set(null), 2000)
}

export const reportClipboardCopy = (text: string) => reportClipboardOperation(text, 'copy')
export const reportClipboardCut = (text: string) => reportClipboardOperation(text, 'cut')
