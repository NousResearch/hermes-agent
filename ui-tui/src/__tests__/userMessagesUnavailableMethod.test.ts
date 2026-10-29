import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { describe, expect, it } from 'vitest'

import { describeRpcError, isVersionSkewError } from '../app/userMessages.js'

// Both server routes now answer a missing method with what the backend knows (it does not serve
// the method); a JSON-RPC client sends no version, so the TUI must not turn that into a claim
// that the versions differ, nor into update advice.
describe('a method the backend says it does not provide', () => {
  const unavailable = new JsonRpcGatewayError(
    'unknown method: groups.no_such_verb — not available on this Hermes backend',
    { code: -32601 }
  )

  it('is not read as version skew and names the method without update advice', () => {
    expect(isVersionSkewError(unavailable)).toBe(false)

    const text = describeRpcError(unavailable)

    expect(text).toContain('groups.no_such_verb')
    expect(text).not.toMatch(/out of sync|different versions|\/update|hermes update/)
  })

  it('keeps the skew warning for a schema the backend rejects', () => {
    const extra = new JsonRpcGatewayError(
      'invalid params for prompt.submit: turn_author: Extra inputs are not permitted',
      { code: 4000 }
    )

    expect(isVersionSkewError(extra)).toBe(true)
    expect(describeRpcError(extra)).toContain('/update')
  })
})
