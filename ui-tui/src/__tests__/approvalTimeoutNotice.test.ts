import { beforeEach, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createServerRequestHandler } from '../app/createServerRequestHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { resetUiState } from '../app/uiStore.js'

beforeEach(() => {
  resetOverlayState()
  resetUiState()
  resetServerRequestsForTests()
})

it('explains a timed-out approval without announcing other cancellations', () => {
  const sys = vi.fn()

  const onEvent = createGatewayEventHandler({
    composer: {},
    gateway: {},
    session: {},
    submission: {},
    system: { sys },
    transcript: {},
    voice: {}
  } as any)

  const onRequest = createServerRequestHandler({ ringPromptBell: vi.fn(), setStatus: vi.fn() })

  for (const reason of ['timeout', 'resolved', 'session_closed', 'interrupted']) {
    const id = `approval-${reason}`

    onRequest({
      fail: vi.fn(),
      id,
      method: 'approval',
      params: { command: 'gh run list', description: 'network command' },
      respond: vi.fn()
    })
    expect(getOverlayState().approval).not.toBeNull()
    onEvent({ payload: { id, method: 'approval', reason }, type: 'request.cancel' } as any)
    expect(getOverlayState().approval).toBeNull()
  }

  expect(sys).toHaveBeenCalledTimes(1)
  expect(sys.mock.calls[0]![0]).toMatch(/did not run/)
})
