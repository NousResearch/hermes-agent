import { expect, it } from 'vitest'

import { t } from '../i18n/runtime.js'
import { isMac } from '../lib/platform.js'

// The retained-input hints must name the shortcut `isAction(key, ch, 'k')` actually
// accepts — Cmd+K on macOS, Ctrl+K elsewhere (#148766). A hard-coded Alt+K is a dead
// key outside macOS, so a user who follows the hint retries nothing.
const retryKey = isMac ? 'Cmd+K' : 'Ctrl+K'

it('retained-input hints name the platform retry shortcut', () => {
  const hints = [
    t('canonical.submit.admissionNotConfirmed', retryKey),
    t('canonical.submit.inputRetained', 'store unreachable', retryKey),
    t('canonical.queue.unconfirmedPrefix', retryKey)
  ]

  for (const hint of hints) {
    expect(hint).toContain(retryKey)
    expect(hint).not.toContain('Alt+K')
  }
})
