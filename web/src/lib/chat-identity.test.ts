// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest'

import {
  chatIdentityScope,
  getChatChannelId,
  getPtyAttachToken
} from './chat-identity'

describe('dashboard chat identity', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        clear: () => values.clear(),
        getItem: (key: string) => values.get(key) ?? null,
        removeItem: (key: string) => void values.delete(key),
        setItem: (key: string, value: string) => void values.set(key, value)
      }
    })
  })

  it('keeps the PTY attach token and event channel stable across a hard refresh', () => {
    const scope = chatIdentityScope('stored-session-1', 'default')

    const token = getPtyAttachToken(scope)
    const channel = getChatChannelId(scope)

    expect(getPtyAttachToken(scope)).toBe(token)
    expect(getChatChannelId(scope)).toBe(channel)
  })

  it('isolates different routed sessions so resume cannot attach to the wrong live PTY', () => {
    const a = chatIdentityScope('stored-session-a', 'default')
    const b = chatIdentityScope('stored-session-b', 'default')

    expect(getPtyAttachToken(a)).not.toBe(getPtyAttachToken(b))
    expect(getChatChannelId(a)).not.toBe(getChatChannelId(b))
  })

  it('isolates profiles that use the same stored session id', () => {
    const defaultScope = chatIdentityScope('stored-session-1', 'default')
    const workScope = chatIdentityScope('stored-session-1', 'work')

    expect(getPtyAttachToken(defaultScope)).not.toBe(getPtyAttachToken(workScope))
    expect(getChatChannelId(defaultScope)).not.toBe(getChatChannelId(workScope))
  })

  it('rotates both identities for an explicit fresh chat', () => {
    const scope = chatIdentityScope(null, 'default')
    const token = getPtyAttachToken(scope)
    const channel = getChatChannelId(scope)

    expect(getPtyAttachToken(scope, true)).not.toBe(token)
    expect(getChatChannelId(scope, true)).not.toBe(channel)
  })
})
