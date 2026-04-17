import { describe, expect, it } from 'vitest'

import { asCommandDispatch } from '../lib/rpc.js'

describe('asCommandDispatch', () => {
  it('parses exec, alias, and skill', () => {
    expect(asCommandDispatch({ type: 'exec', output: 'hi' })).toEqual({ type: 'exec', output: 'hi' })
    expect(asCommandDispatch({ type: 'alias', target: 'help' })).toEqual({ type: 'alias', target: 'help' })
    expect(asCommandDispatch({ type: 'skill', name: 'x', message: 'do' })).toEqual({
      type: 'skill',
      name: 'x',
      message: 'do'
    })
  })

  it('rejects malformed payloads', () => {
    expect(asCommandDispatch(null)).toBeNull()
    expect(asCommandDispatch({ type: 'alias' })).toBeNull()
    expect(asCommandDispatch({ type: 'skill', name: 1 })).toBeNull()
  })
})
