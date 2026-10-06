import { describe, expect, it, vi } from 'vitest'

import {
  interruptCurrentSession,
  isCurrentStopRequest,
  type SessionInterruptGateway
} from './chat-stop'

describe('interruptCurrentSession', () => {
  it('uses the authenticated gateway route with the exact current session id', async () => {
    const gateway = {
      connect: vi.fn(async () => undefined),
      request: vi.fn(async () => ({ status: 'interrupted' }))
    } as unknown as SessionInterruptGateway

    await expect(interruptCurrentSession(gateway, 'runtime-session-7')).resolves.toEqual({
      status: 'interrupted'
    })
    expect(gateway.connect).toHaveBeenCalledOnce()
    expect(gateway.request).toHaveBeenCalledWith('session.interrupt', {
      session_id: 'runtime-session-7'
    })
  })

  it('rejects an acknowledgement that does not confirm interruption', async () => {
    const gateway = {
      connect: vi.fn(async () => undefined),
      request: vi.fn(async () => ({ status: 'not_interrupted', interrupted: false }))
    } as unknown as SessionInterruptGateway

    await expect(interruptCurrentSession(gateway, 'runtime-session-8')).rejects.toThrow(
      'Stop was not acknowledged'
    )
  })

  it('fails closed when no current session id is available', async () => {
    const gateway = {
      connect: vi.fn(async () => undefined),
      request: vi.fn()
    } as unknown as SessionInterruptGateway

    await expect(interruptCurrentSession(gateway, '')).rejects.toThrow(
      'Current chat session is not available'
    )
    expect(gateway.connect).not.toHaveBeenCalled()
    expect(gateway.request).not.toHaveBeenCalled()
  })
})

describe('isCurrentStopRequest', () => {
  it('rejects a late acknowledgement after the runtime session changes', () => {
    expect(isCurrentStopRequest(4, 5, 'chat-a', 'chat-a', 'old-session', 'new-session')).toBe(false)
    expect(isCurrentStopRequest(4, 4, 'chat-a', 'chat-a', 'old-session', 'new-session')).toBe(false)
  })

  it('accepts an acknowledgement only for the current channel and session', () => {
    expect(isCurrentStopRequest(4, 4, 'chat-a', 'chat-a', 'session-4', 'session-4')).toBe(true)
    expect(isCurrentStopRequest(4, 4, 'chat-a', 'chat-b', 'session-4', 'session-4')).toBe(false)
  })
})

