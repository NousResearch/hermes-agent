import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { creditsCommands } from '../app/slash/commands/credits.js'
import type { CreditsViewResponse } from '../gatewayTypes.js'

// The command opens the top-up URL through this helper on confirm. Mock it so
// the test never shells out to a real browser/`xdg-open` and we can assert the
// success/failure messaging deterministically.
vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: vi.fn(() => true)
}))

import { openExternalUrl } from '../lib/openExternalUrl.js'

const openExternalUrlMock = vi.mocked(openExternalUrl)

const creditsCommand = creditsCommands.find(cmd => cmd.name === 'credits')!

const buildView = (overrides: Partial<CreditsViewResponse> = {}): CreditsViewResponse => ({
  balance_lines: ['Grant: $9.50 left', 'Top-up: $25.00'],
  depleted: false,
  identity_line: 'Signed in as ada@example.com',
  logged_in: true,
  topup_url: 'https://portal.nousresearch.com/billing/topup',
  ...overrides
})

// Mirror createSlashHandler's real `guarded` wrapper: skip the handler when the
// command is stale OR the response is falsy. Tests stay non-stale, so this is a
// straightforward "run the handler when we got a response" shim.
const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

const buildCtx = (rpcResult: CreditsViewResponse) => {
  const sys = vi.fn()
  const rpc = vi.fn(() => Promise.resolve(rpcResult))
  const guardedErr = vi.fn()

  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr,
    sid: 'sid-abc',
    stale: () => false,
    transcript: { page: vi.fn(), panel: vi.fn(), sys },
    ui: { locale: 'en' }
  }

  // Run the command, then await the rpc promise so the .then() handler has
  // flushed before assertions — deterministic, no polling/timeouts.
  const run = async () => {
    creditsCommand.run('', ctx as any, 'credits')
    await rpc.mock.results[0]?.value
    // Allow the chained .then() microtask to settle.
    await Promise.resolve()
  }

  return { ctx, rpc, run, sys }
}

describe('/credits slash command', () => {
  beforeEach(() => {
    resetOverlayState()
    openExternalUrlMock.mockClear()
    openExternalUrlMock.mockReturnValue(true)
  })

  it('renders the balance (including top-up URL) and arms the confirm overlay', async () => {
    const view = buildView()
    const { rpc, run, sys } = buildCtx(view)

    await run()

    expect(rpc).toHaveBeenCalledWith('credits.view', { session_id: 'sid-abc' })

    // (a) sys received the balance text including the topup_url
    const printed = sys.mock.calls.map(call => call[0]).join('\n')
    expect(printed).toContain('💳 Nous credits')
    expect(printed).toContain('Grant: $9.50 left')
    expect(printed).toContain('Signed in as ada@example.com')
    expect(printed).toContain(view.topup_url)

    // (b) confirm overlay set with the expected label + detail
    const confirm = getOverlayState().confirm
    expect(confirm).toBeTruthy()
    expect(confirm?.confirmLabel).toBe('Open top-up in browser')
    expect(confirm?.cancelLabel).toBe('Cancel')
    expect(confirm?.title).toBe('Add credits?')
    expect(confirm?.detail).toBe(view.topup_url)

    // onConfirm opens the URL and reports success back to the transcript
    confirm?.onConfirm()
    expect(openExternalUrlMock).toHaveBeenCalledWith(view.topup_url)
    expect(sys).toHaveBeenCalledWith('Complete your top-up in the browser — credits will appear in /credits shortly.')
  })

  it('falls back to printing the URL when the browser open is rejected', async () => {
    openExternalUrlMock.mockReturnValue(false)
    const view = buildView()
    const { run, sys } = buildCtx(view)

    await run()

    const confirm = getOverlayState().confirm
    expect(confirm).toBeTruthy()
    confirm?.onConfirm()
    expect(sys).toHaveBeenCalledWith(`Open this URL to top up: ${view.topup_url}`)
  })

  it('does not arm the confirm overlay when there is no top-up URL', async () => {
    const view = buildView({ topup_url: null })
    const { run, sys } = buildCtx(view)

    await run()

    const printed = sys.mock.calls.map(call => call[0]).join('\n')
    expect(printed).toContain('💳 Nous credits')
    expect(getOverlayState().confirm).toBeNull()
  })

  it('shows the not-logged-in message and does NOT arm the confirm overlay', async () => {
    const view = buildView({
      balance_lines: [],
      identity_line: null,
      logged_in: false,
      topup_url: null
    })

    const { run, sys } = buildCtx(view)

    await run()

    expect(sys).toHaveBeenCalledWith('💳 Not logged into Nous Portal — run /portal to log in.')
    expect(getOverlayState().confirm).toBeNull()
    expect(openExternalUrlMock).not.toHaveBeenCalled()
  })
})
