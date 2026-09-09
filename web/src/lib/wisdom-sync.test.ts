import { expect, it, vi } from 'vitest'
import { createWisdomSyncController, type WisdomSyncSnapshot } from '@hermes/shared'

const counts = { pending: 0, syncing: 0, retryable: 0, conflict: 0, uncertain: 0, waiting_for_receipt: 0 }
const snapshot: WisdomSyncSnapshot = { delivery: counts, operation: { ...counts, retryable: 1 }, can_retry: true }
function setup() {
  const deps = {
    read: vi.fn().mockResolvedValue(snapshot),
    retry: vi.fn().mockResolvedValue({ ...snapshot, can_retry: false }),
    changed: vi.fn()
  }
  return { deps, controller: createWisdomSyncController(deps) }
}
it('reads without retrying and requires current retryable state', async () => {
  const { deps, controller } = setup()
  await controller.retry()
  expect(deps.retry).not.toHaveBeenCalled()
  await controller.refresh()
  expect(deps.retry).not.toHaveBeenCalled()
  await controller.retry()
  await controller.retry()
  expect(deps.retry).toHaveBeenCalledOnce()
})
it('serializes reads/retries and discards responses after disposal', async () => {
  const { deps, controller } = setup()
  let done!: (value: WisdomSyncSnapshot) => void
  deps.read.mockReturnValue(
    new Promise(resolve => {
      done = resolve
    })
  )
  const first = controller.refresh()
  await controller.refresh()
  await controller.retry()
  expect(deps.read).toHaveBeenCalledOnce()
  controller.dispose()
  const changed = deps.changed.mock.calls.length
  done(snapshot)
  await first
  expect(deps.changed).toHaveBeenCalledTimes(changed)
  await controller.retry()
  expect(deps.retry).not.toHaveBeenCalled()
})
it('a failed read disables retries until a successful refresh', async () => {
  const { deps, controller } = setup()
  await controller.refresh()
  deps.read.mockRejectedValueOnce(new Error('private backend details'))
  await controller.refresh()
  expect(deps.changed.mock.lastCall![0].error).toBe(true)
  expect(JSON.stringify(deps.changed.mock.lastCall)).not.toContain('private backend details')
  await controller.retry()
  expect(deps.retry).not.toHaveBeenCalled()
  await controller.refresh()
  await controller.retry()
  expect(deps.retry).toHaveBeenCalledOnce()
})
