import { describe, expect, it, vi } from 'vitest'

import { createFreeTierChallengePresenter } from '../app/gatewayBrowserLinks.js'

const url = 'https://portal.example/challenge?code=t'

const challenge = (required: boolean) => ({
  attempt: 0,
  expires_in: 600,
  message: 'A quick check first.',
  required,
  type: 'browser' as const,
  url
})

describe('free_tier.challenge presenter', () => {
  it('shows the link and opens one tab per ticket; an optional check opens nothing', () => {
    const sys = vi.fn()
    const open = vi.fn()
    const show = createFreeTierChallengePresenter(sys, open)

    show(challenge(false))
    expect(open).not.toHaveBeenCalled()

    show(challenge(true))
    show(challenge(true))

    expect(sys.mock.calls.map(c => c[0]).join('\n')).toContain(url)
    expect(open).toHaveBeenCalledTimes(1)
  })
})
