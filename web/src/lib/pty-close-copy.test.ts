import { describe, expect, it } from 'vitest'

import {
  PTY_GAVE_UP_BANNER,
  PTY_START_FAILED_MESSAGE,
  PTY_TOKEN_MISSING_BANNER,
  ptyReconnectExhausted,
  ptyRejectionBanner,
  ptyStartFailedMessage
} from './pty-close-copy'

describe('pty close banners', () => {
  it('offers a Reload action for the stale-token and missing-token cases', () => {
    expect(ptyRejectionBanner(4401)?.action).toBe('reload')
    expect(PTY_TOKEN_MISSING_BANNER.action).toBe('reload')
  })

  it('treats server rejections as banners but not transient drops or the agent exit', () => {
    for (const code of [4401, 4403, 4404, 4408]) {
      expect(ptyRejectionBanner(code)).not.toBeNull()
    }
    // Transient drops and the agent's own exit are not rejections: the caller
    // must route them to the reconnect ladder / restart affordance instead.
    expect(ptyRejectionBanner(1006)).toBeNull()
    expect(ptyRejectionBanner(4410)).toBeNull()
  })

  it('stops retrying after the last attempt and points at the server', () => {
    expect(ptyReconnectExhausted(5, 5)).toBe(true)
    expect(ptyReconnectExhausted(4, 5)).toBe(false)
    expect(PTY_GAVE_UP_BANNER.action).toBe('check-server')
  })

  it('surfaces the 1011 close-frame reason, keeping the generic copy when there is none', () => {
    // The server reason is already the full user-facing sentence.
    expect(ptyStartFailedMessage('Chat could not start: Hermes needs Node.js.')).toBe(
      'Chat could not start: Hermes needs Node.js.'
    )
    // Whitespace-only and empty reasons (older servers, platform-PTY closes)
    // fall back to the copy that points at the terminal.
    expect(ptyStartFailedMessage('')).toBe(PTY_START_FAILED_MESSAGE)
    expect(ptyStartFailedMessage('   ')).toBe(PTY_START_FAILED_MESSAGE)
  })
})
