import { describe, expect, it } from 'vitest'

import {
  PTY_GAVE_UP_BANNER,
  PTY_TOKEN_MISSING_BANNER,
  ptyReconnectExhausted,
  ptyRejectionBanner
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
})
