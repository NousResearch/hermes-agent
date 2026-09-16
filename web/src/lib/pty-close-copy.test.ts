import { describe, expect, it } from 'vitest'

import {
  PTY_GAVE_UP_BANNER,
  PTY_RECONNECTING_BANNER,
  PTY_SESSION_ENDED_MESSAGE,
  PTY_START_FAILED_MESSAGE,
  PTY_TOKEN_MISSING_BANNER,
  ptyReconnectExhausted,
  ptyRejectionBanner
} from './pty-close-copy'

describe('pty close copy', () => {
  it('offers a Reload action for the stale-token and missing-token cases without naming auth internals', () => {
    const stale = ptyRejectionBanner(4401)
    expect(stale?.action).toBe('reload')
    expect(stale?.text).toMatch(/reload the page/i)
    expect(stale?.text).not.toMatch(/auth failed|bad-token|4401/i)

    expect(PTY_TOKEN_MISSING_BANNER.action).toBe('reload')
    expect(PTY_TOKEN_MISSING_BANNER.text).not.toMatch(/session token/i)
    expect(PTY_TOKEN_MISSING_BANNER.text).toContain('hermes dashboard')
  })

  it('never prints a WebSocket close code in user-facing text', () => {
    const texts = [
      PTY_RECONNECTING_BANNER,
      PTY_GAVE_UP_BANNER.text,
      PTY_SESSION_ENDED_MESSAGE,
      ...[4401, 4403, 4404, 4408].map(code => ptyRejectionBanner(code)!.text)
    ]
    for (const text of texts) {
      expect(text).not.toMatch(/\b(code\s*)?\d{4}\b/)
    }
    // Transient drops and the agent's own exit are not rejections: the caller
    // must route them to the reconnect ladder / restart affordance instead.
    expect(ptyRejectionBanner(1006)).toBeNull()
    expect(ptyRejectionBanner(4410)).toBeNull()
  })

  it('says retries stopped and points at the server after the last attempt', () => {
    expect(ptyReconnectExhausted(5, 5)).toBe(true)
    expect(ptyReconnectExhausted(4, 5)).toBe(false)
    expect(PTY_GAVE_UP_BANNER.action).toBe('check-server')
    expect(PTY_GAVE_UP_BANNER.text).toContain('hermes dashboard')
    expect(PTY_SESSION_ENDED_MESSAGE).toMatch(/crashed/i)
    expect(PTY_SESSION_ENDED_MESSAGE).toMatch(/logs/i)
  })
})

describe('start-failed overlay copy', () => {
  it('stays neutral: close 1011 also means "no terminal support here", where retrying cannot help', () => {
    expect(PTY_START_FAILED_MESSAGE).toMatch(/printed above/)
    expect(PTY_START_FAILED_MESSAGE).not.toMatch(/fix it|Start new session/)
    expect(PTY_START_FAILED_MESSAGE).not.toMatch(/1011/)
  })
})
