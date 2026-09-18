import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { describe, expect, it } from 'vitest'

import {
  backendGaveUp,
  describeCredentialWarning,
  describeRpcError,
  describeSlashExecError,
  describeTurnFailure,
  isVersionSkewError,
  lastStderrLine,
  promptTimeoutNotice,
  setRpcErrorLogSink,
  shouldFallbackToDispatch,
  stderrLooksLikeProblem,
  stderrProblemActivity
} from '../app/userMessages.js'

// Behaviour contracts for the user-facing wording, not snapshots: each test
// asserts the message names what happened and cites the real next step.

describe('describeTurnFailure', () => {
  it('turns an auth error_surface into a plain title, a Details line and the /model + /retry hint', () => {
    const raw =
      'Error code: 401 - {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}'

    const text = describeTurnFailure({
      error: raw,
      error_surface: { code: 'auth', layer: 'auth', provider: 'openai', retryable: false },
      recoverable: true
    })

    const [title, details] = text.split('\n')

    expect(title).toMatch(/rejected the API key \(openai\)/)
    expect(title).not.toMatch(/Error code|401|\{/)
    expect(details).toMatch(/^Details: /)
    expect(details).toContain('Incorrect API key provided')
    expect(text).toContain('/model')
    expect(text).toContain('/retry')
  })

  it('falls back to the layer copy, and to a generic title, when the code is unknown', () => {
    const streaming = describeTurnFailure({
      error: 'peer closed connection',
      error_surface: { code: 'weird', layer: 'streaming', retryable: true }
    })

    expect(streaming).toMatch(/dropped mid-reply/)
    expect(streaming).toContain('/retry')

    const bare = describeTurnFailure({ error: 'boom' })
    expect(bare.split('\n')[0]).toMatch(/^The request failed\./)
    expect(bare).toContain('Details: boom')
  })

  it('drops the /retry pointer when the backend says the turn is not recoverable', () => {
    const text = describeTurnFailure({
      error: 'x',
      error_surface: { code: 'model_not_found', layer: 'provider', retryable: false },
      recoverable: false
    })

    expect(text).toContain('/model')
  })

  it('honours error_surface.retryable=false even though the backend always sets recoverable=true', () => {
    const notRetryable = describeTurnFailure({
      error: 'x',
      error_surface: { code: 'weird', layer: 'provider', retryable: false },
      recoverable: true
    })

    expect(notRetryable).not.toContain('/retry')
    expect(notRetryable).toContain('/model')

    const retryable = describeTurnFailure({
      error: 'x',
      error_surface: { code: 'weird', layer: 'provider', retryable: true },
      recoverable: true
    })

    expect(retryable).toContain('/retry')
  })
})

describe('describeRpcError', () => {
  it('rewrites transport errors without exposing RPC method names', () => {
    for (const raw of ['gateway not connected: prompt.submit', 'gateway not running']) {
      const text = describeRpcError(new Error(raw))

      expect(text).not.toContain('prompt.submit')
      expect(text).not.toMatch(/\bgateway\b/)
      expect(text).toMatch(/not connected/)
      expect(text).toContain('/logs')
    }
  })

  it('keeps the timeout seconds but drops the method', () => {
    const text = describeRpcError(new Error('request timed out after 120s: slash.exec'))

    expect(text).toContain('120s')
    expect(text).not.toContain('slash.exec')
    expect(text).toContain('/logs')
  })

  it('explains a stale session id as saved-and-reopenable, not as loss', () => {
    const text = describeRpcError(new JsonRpcGatewayError('session not found', { code: 4001 }))

    expect(text).not.toMatch(/not found/)
    expect(text).toMatch(/saved/)
    expect(text).toContain('/resume')
  })

  it('passes ordinary domain errors through unchanged', () => {
    expect(describeRpcError(new JsonRpcGatewayError('hash required', { code: 4014 }))).toBe('hash required')
  })

  it('does not treat every 4001 as a stale session: the backend reuses the code for other refusals', () => {
    for (const raw of ['no active session to retry', 'slug and api_key are required', 'session ownership changed']) {
      expect(describeRpcError(new JsonRpcGatewayError(raw, { code: 4001 }))).toBe(raw)
    }

    expect(
      describeRpcError(new JsonRpcGatewayError('session not found or not owned by this transport', { code: 4001 }))
    ).toContain('/resume')
  })

  it('records the raw wire text it replaced in the log sink', () => {
    const lines: string[] = []
    setRpcErrorLogSink(line => lines.push(line))

    try {
      describeRpcError(new JsonRpcGatewayError('session not found', { code: 4001 }))
      describeRpcError(new Error('gateway not connected: prompt.submit'))
      describeRpcError(new JsonRpcGatewayError('hash required', { code: 4014 }))
    } finally {
      setRpcErrorLogSink(null)
    }

    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('4001')
    expect(lines[0]).toContain('session not found')
    expect(lines[1]).toContain('prompt.submit')
  })
})

describe('isVersionSkewError', () => {
  it('recognises both skew shapes and points at /update', () => {
    const extra = new JsonRpcGatewayError(
      'invalid params for prompt.submit: turn_author: Extra inputs are not permitted',
      { code: 4000 }
    )

    const unknown = new JsonRpcGatewayError('unknown method: session.control.read', { code: -32601 })

    expect(isVersionSkewError(extra)).toBe(true)
    expect(isVersionSkewError(unknown)).toBe(true)
    expect(describeRpcError(extra)).toContain('/update')
    expect(describeRpcError(extra)).not.toContain('turn_author')
    expect(isVersionSkewError(new JsonRpcGatewayError('session_id required', { code: 4000 }))).toBe(false)
  })
})

describe('slash.exec fallback policy', () => {
  it('falls back to command.dispatch only for the two "slash.exec does not own this" 4018 refusals', () => {
    expect(
      shouldFallbackToDispatch(new JsonRpcGatewayError('skill command: use command.dispatch for /x', { code: 4018 }))
    ).toBe(true)
    expect(
      shouldFallbackToDispatch(
        new JsonRpcGatewayError(
          'snapshot restore mutates live config/state; use command.dispatch for /snapshot restore',
          { code: 4018 }
        )
      )
    ).toBe(true)
    expect(shouldFallbackToDispatch(new JsonRpcGatewayError('slash worker timed out', { code: 5030 }))).toBe(false)
    expect(shouldFallbackToDispatch(new JsonRpcGatewayError('session not found', { code: 4001 }))).toBe(false)
  })

  it('does not re-dispatch a 4018 that command.dispatch itself already returned (would re-run /retry, /undo, bundles)', () => {
    for (const raw of [
      'retry cannot safely reconstruct or combine attached media',
      'bundle dispatch failed: boom',
      'not a quick/plugin/bundle/skill command: zzz',
      'quick command failed with exit code 1'
    ]) {
      expect(shouldFallbackToDispatch(new JsonRpcGatewayError(raw, { code: 4018 }))).toBe(false)
    }

    // slash.exec never emits 4011; a code the TUI does not know is not a fallback ticket either.
    expect(shouldFallbackToDispatch(new JsonRpcGatewayError('unknown command: zzz', { code: 4011 }))).toBe(false)
  })

  it('names the command and the helper failure instead of the fallback refusal', () => {
    const timeout = describeSlashExecError('status', new JsonRpcGatewayError('slash worker timed out', { code: 5030 }))

    expect(timeout).toMatch(/^\/status did not finish/)
    expect(timeout).toMatch(/timed out/)
    expect(timeout).toContain('/logs')
    expect(timeout).not.toMatch(/quick\/plugin\/bundle\/skill/)

    const crash = describeSlashExecError(
      'journey',
      new JsonRpcGatewayError('slash worker closed pipe: ValueError: bad', { code: 5030 })
    )

    expect(crash).toMatch(/^\/journey did not finish/)
    expect(crash).toContain('Details: ValueError: bad')
  })
})

describe('backend lifecycle copy', () => {
  it('names the exit code, the last real stderr line, /logs and hermes doctor', () => {
    const tail =
      '[lifecycle] child exit code=1\nModuleNotFoundError: No module named pydantic\n[lifecycle] scheduling gateway reconnect in 1000ms (attempt 1)'

    const text = backendGaveUp(1, lastStderrLine(tail))

    expect(text).toContain('exit code 1')
    expect(text).toContain('Details: ModuleNotFoundError: No module named pydantic')
    expect(text).not.toContain('[lifecycle]')
    expect(text).toContain('/logs')
    expect(text).toContain('hermes doctor')
    expect(text).toContain('/resume')
    expect(text).not.toMatch(/\bgateway\b/)
  })

  it('only failure-looking stderr earns an activity row', () => {
    expect(stderrLooksLikeProblem('  File "/x/run_agent.py", line 812, in _call_model')).toBe(false)
    expect(stderrLooksLikeProblem('Traceback (most recent call last):')).toBe(true)
    expect(stderrLooksLikeProblem('[gateway-turn] ValueError: nope')).toBe(true)
    expect(stderrLooksLikeProblem('ValueError: nope')).toBe(true)
    expect(stderrLooksLikeProblem('INFO hermes.mcp: discovered 3 servers')).toBe(false)
  })

  it('ignores dependency warning lines and does not call the failure a "backend" problem', () => {
    expect(stderrLooksLikeProblem('/x/site-packages/foo.py:12: DeprecationWarning: use bar instead')).toBe(false)
    expect(stderrLooksLikeProblem('UserWarning: something is odd')).toBe(false)

    const row = stderrProblemActivity('ValueError: nope')
    expect(row).toContain('(ValueError)')
    expect(row).toContain('/logs')
    expect(row).not.toMatch(/\bbackend\b/)
  })
})

describe('promptTimeoutNotice', () => {
  it('explains a timed-out password/vault prompt and stays silent for other reasons', () => {
    const sudo = promptTimeoutNotice('sudo', 'timeout')

    expect(sudo).toMatch(/Password prompt closed/)
    expect(sudo).toMatch(/skipped/)
    // The timeout lengths live in Python (agent_callbacks.py); the copy must not hard-code them.
    expect(sudo).not.toMatch(/\d+ minutes?/)
    expect(promptTimeoutNotice('vault.code', 'timeout')).not.toMatch(/\d+ minutes?/)
    expect(promptTimeoutNotice('vault.code', 'timeout')).toMatch(/code/)
    expect(promptTimeoutNotice('sudo', 'interrupted')).toBeNull()
    expect(promptTimeoutNotice('approval', 'timeout')).toBeNull()
  })
})

describe('describeCredentialWarning', () => {
  it('adds the /model fix to the backend missing-key warning', () => {
    const text = describeCredentialWarning("No API key configured for provider 'openai'. First message will fail.")

    expect(text).toContain('openai')
    expect(text).toContain('/model')
    expect(text).toContain('/setup')
    expect(describeCredentialWarning('something else')).toBe('something else')
  })
})
