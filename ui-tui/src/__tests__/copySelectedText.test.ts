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

  await expect(copySelectedText('local selection')).resolves.toBe('clipboard')
  expect(nativeCopy).toHaveBeenCalledWith('local selection')
  expect(remoteCopy).not.toHaveBeenCalled()
  await expect(copySelectedText('fallback selection')).resolves.toBe('terminal')
  expect(remoteCopy).toHaveBeenCalledExactlyOnceWith('fallback selection')
})

it('treats a buffered terminal write as sent, rather than as a failed clipboard request', async () => {
  vi.stubEnv('SSH_CONNECTION', '127.0.0.1 1234 127.0.0.1 22')
  vi.spyOn(osc52, 'writeOsc52Clipboard').mockReturnValue(false)

  await expect(copySelectedText('buffered selection')).resolves.toBe('terminal')
})
