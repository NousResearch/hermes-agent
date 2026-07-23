import { describe, expect, it } from 'vitest'

import {
  asRpcResult,
  rpcErrorMessage,
  RpcMethodUnavailableError,
  shouldRethrowRpcError
} from '../lib/rpc.js'

describe('asRpcResult', () => {
  it('keeps plain object payloads', () => {
    expect(asRpcResult({ ok: true, value: 'x' })).toEqual({ ok: true, value: 'x' })
  })

  it('rejects missing or non-object payloads', () => {
    expect(asRpcResult(undefined)).toBeNull()
    expect(asRpcResult(null)).toBeNull()
    expect(asRpcResult('oops')).toBeNull()
    expect(asRpcResult(['bad'])).toBeNull()
  })
})

describe('rpcErrorMessage', () => {
  it('prefers Error messages', () => {
    expect(rpcErrorMessage(new Error('boom'))).toBe('boom')
  })

  it('falls back for unknown errors', () => {
    expect(rpcErrorMessage('broken')).toBe('broken')
    expect(rpcErrorMessage({ code: 500 })).toBe('request failed')
  })
})

describe('shouldRethrowRpcError', () => {
  const unavailable = Object.assign(new Error('unknown method: sudo.cancel'), { code: -32601 })

  it('preserves the default null-on-error RPC contract', () => {
    expect(shouldRethrowRpcError(unavailable)).toBe(false)
    expect(shouldRethrowRpcError(unavailable, {})).toBe(false)
  })

  it('opts into method-unavailable propagation without propagating transient errors', () => {
    expect(shouldRethrowRpcError(unavailable, { rethrowMethodUnavailable: true })).toBe(true)
    expect(
      shouldRethrowRpcError(new Error('timeout: sudo.cancel'), { rethrowMethodUnavailable: true })
    ).toBe(false)
    expect(new RpcMethodUnavailableError('unknown method: sudo.cancel')).toBeInstanceOf(Error)
  })
})
