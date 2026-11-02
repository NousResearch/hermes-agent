import { describe, expect, it } from 'vitest'

import { SIDECAR_DISCONNECTED_MESSAGE, credentialWarning, sidecarErrorMessage } from './chat-sidebar-banner'

describe('credentialWarning', () => {
  it('rewrites the gateway probe into a sentence that names the provider and the fix', () => {
    const w = credentialWarning("No API key configured for provider 'openrouter'. First message will fail.")
    expect(w?.provider).toBe('openrouter')
    expect(w?.message).toContain('openrouter')
    expect(w?.message).not.toContain('First message will fail')
  })

  it('passes unknown warnings through untouched and ignores empty input', () => {
    expect(credentialWarning('OAuth token for anthropic expires in 2 minutes')?.message).toBe(
      'OAuth token for anthropic expires in 2 minutes'
    )
    expect(credentialWarning(undefined)).toBeNull()
    expect(credentialWarning('')).toBeNull()
  })
})

describe('sidecarErrorMessage', () => {
  it('collapses transport jargon into the side-panel sentence and keeps real errors', () => {
    for (const raw of [
      'WebSocket connection failed',
      'WebSocket closed',
      'gateway not connected',
      'Session token not available — page must be served by the Hermes dashboard server'
    ]) {
      expect(sidecarErrorMessage(raw)).toBe(SIDECAR_DISCONNECTED_MESSAGE)
    }
    expect(sidecarErrorMessage("Profile 'nope' not found")).toBe("Profile 'nope' not found")
  })
})
