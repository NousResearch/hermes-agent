import { afterEach, expect, it, vi } from 'vitest'

import * as clipboard from '../lib/clipboard.js'
import { copySelectedText } from '../lib/copySelectedText.js'
import * as osc52 from '../lib/osc52.js'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

it('uses the local clipboard, falling back to OSC 52 when native copying is unavailable', async () => {
  for (const name of ['SSH_CONNECTION', 'SSH_CLIENT', 'SSH_TTY']) {
    vi.stubEnv(name, '')
  }

  const nativeCopy = vi.spyOn(clipboard, 'writeClipboardText').mockResolvedValueOnce(true).mockResolvedValueOnce(false)
  const remoteCopy = vi.spyOn(osc52, 'writeOsc52Clipboard').mockReturnValue(true)

  await copySelectedText('local selection')
  expect(nativeCopy).toHaveBeenCalledWith('local selection')
  expect(remoteCopy).not.toHaveBeenCalled()
  await copySelectedText('fallback selection')
  expect(remoteCopy).toHaveBeenCalledExactlyOnceWith('fallback selection')
})
