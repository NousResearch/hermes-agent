import { describe, expect, it } from 'vitest'

import { shouldHideAuthWidget } from './auth-widget-visibility'

describe('AuthWidget loopback identity', () => {
  it('hides the auth and logout affordance for loopback identity', () => {
    expect(
      shouldHideAuthWidget({
        display_name: 'Local',
        email: '',
        expires_at: 0,
        org_id: '',
        provider: 'loopback',
        user_id: 'local'
      })
    ).toBe(true)
  })

  it('keeps the widget visible for gated identities', () => {
    expect(
      shouldHideAuthWidget({
        display_name: '',
        email: '',
        expires_at: 123,
        org_id: 'org',
        provider: 'portal',
        user_id: 'user-1'
      })
    ).toBe(false)
  })
})
