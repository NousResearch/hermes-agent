import { describe, expect, it } from 'vitest'

import {
  enqueuePtyInput,
  PTY_RECONNECT_BASE_MS,
  PTY_RECONNECT_MAX_MS,
  ptyReconnectDelayMs,
  shouldBlockPtyInput,
  shouldQueuePtyInput,
  shouldReconnectPtyOnPageResume
} from './pty-reconnect'

describe('shouldReconnectPtyOnPageResume', () => {
  it('reconnects a missing socket when the active page becomes visible', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: null,
        ptyState: 'reconnecting'
      })
    ).toBe(true)
  })

  it('reconnects closed or closing sockets on visible resume', () => {
    for (const socketReadyState of [2, 3]) {
      expect(
        shouldReconnectPtyOnPageResume({
          isActive: true,
          visibilityState: 'visible',
          online: true,
          socketReadyState,
          ptyState: 'reconnecting'
        })
      ).toBe(true)
    }
  })

  it('does not restart the ladder on focus/online after it gave up; only the Reconnect button does', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: 3,
        ptyState: 'closed',
        reconnectGaveUp: true
      })
    ).toBe(false)
  })

  it('does not reconnect an open socket on visible resume', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: 1,
        ptyState: 'open'
      })
    ).toBe(false)
  })

  it('reconnects a still-connecting socket when the page is already in reconnecting state', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: 0,
        ptyState: 'reconnecting'
      })
    ).toBe(true)
  })

  it('does not reconnect while the page is hidden', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'hidden',
        online: true,
        socketReadyState: 3,
        ptyState: 'reconnecting'
      })
    ).toBe(false)
  })

  it('defers reconnect while offline', () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: false,
        socketReadyState: 3,
        ptyState: 'reconnecting'
      })
    ).toBe(false)
  })

  it('does not fire a redundant reconnect while a connect is in flight (wsRef not yet assigned)', () => {
    // The async socket-open IIFE has begun but not yet assigned wsRef, so
    // socketReadyState reads null. Without the connectInFlight guard this
    // would return true and double-connect.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: null,
        ptyState: 'connecting',
        connectInFlight: true
      })
    ).toBe(false)
  })

  it('still reconnects an in-flight connect when the page already believes it is closed', () => {
    // A stuck attempt the user is actively trying to recover (manual reconnect
    // or a closed state) must not be suppressed by the in-flight guard.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: 'visible',
        online: true,
        socketReadyState: null,
        ptyState: 'closed',
        connectInFlight: true
      })
    ).toBe(true)
  })
})

describe('shouldBlockPtyInput', () => {
  it('allows input only while the PTY socket is open', () => {
    expect(shouldBlockPtyInput('open')).toBe(false)
    expect(shouldBlockPtyInput('connecting')).toBe(true)
    expect(shouldBlockPtyInput('reconnecting')).toBe(true)
    expect(shouldBlockPtyInput('closed')).toBe(true)
    expect(shouldBlockPtyInput('ended')).toBe(true)
  })
})

describe('shouldQueuePtyInput', () => {
  it('holds input only while a socket is on its way', () => {
    expect(shouldQueuePtyInput('connecting')).toBe(true)
    expect(shouldQueuePtyInput('reconnecting')).toBe(true)
    expect(shouldQueuePtyInput('open')).toBe(false)
    expect(shouldQueuePtyInput('closed')).toBe(false)
    expect(shouldQueuePtyInput('ended')).toBe(false)
  })
})

describe('enqueuePtyInput', () => {
  it('appends chunks in typing order without mutating the input queue', () => {
    const queue = ['æ']
    const next = enqueuePtyInput(queue, 'ø')

    expect(next).toEqual(['æ', 'ø'])
    expect(queue).toEqual(['æ'])
  })

  it('ignores empty chunks', () => {
    expect(enqueuePtyInput(['a'], '')).toEqual(['a'])
  })

  it('drops the oldest whole chunks once the limit is passed', () => {
    expect(enqueuePtyInput(['abc', 'de'], 'fg', 5)).toEqual(['de', 'fg'])
  })

  it('never splits an escape sequence when trimming', () => {
    const arrowUp = '\x1b[A'
    const next = enqueuePtyInput([arrowUp, 'x'], 'yz', 4)

    expect(next).toEqual(['x', 'yz'])
    expect(next.join('')).not.toContain('[A')
  })

  it('drops a single chunk larger than the limit instead of truncating it', () => {
    expect(enqueuePtyInput(['a'], 'too long', 4)).toEqual(['a'])
  })
})

describe('ptyReconnectDelayMs', () => {
  it('doubles from the base on each attempt and clamps at the cap', () => {
    expect(ptyReconnectDelayMs(1)).toBe(PTY_RECONNECT_BASE_MS)
    expect(ptyReconnectDelayMs(2)).toBe(Math.min(PTY_RECONNECT_BASE_MS * 2, PTY_RECONNECT_MAX_MS))
    expect(ptyReconnectDelayMs(99)).toBe(PTY_RECONNECT_MAX_MS)
  })
})
