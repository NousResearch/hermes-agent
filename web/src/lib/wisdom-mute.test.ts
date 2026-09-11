import { describe, expect, it, vi } from 'vitest'
import {
  createWisdomMuteController,
  type WisdomMuteControl,
  type WisdomMuteSnapshot,
  type WisdomMuteState
} from '@hermes/shared'

const control: WisdomMuteControl = {
  id: 'a'.repeat(32),
  expires_at: 2000,
  organization_id: 'org',
  sync: null,
  mute: { org_id: 'org', revision: 3, duration: null, muted: false, forever: false, muted_until: null }
}
const snapshot: WisdomMuteSnapshot = { organization_id: 'org', gateway_available: true, mute: control.mute, sync: null }
function setup() {
  const states: WisdomMuteState[] = []
  const deps = {
    prepare: vi.fn().mockResolvedValue(structuredClone(control)),
    read: vi.fn().mockResolvedValue(structuredClone(snapshot)),
    choose: vi.fn().mockResolvedValue(structuredClone(snapshot)),
    now: () => 1000_000,
    changed: (state: WisdomMuteState) => states.push(state)
  }
  return { deps, controller: createWisdomMuteController(deps), state: () => states.at(-1)! }
}
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => {
    resolve = done
  })
  return { promise, resolve }
}

describe('native Wisdom mute state', () => {
  it('opening and selecting do not apply', async () => {
    const { controller, deps } = setup()
    await controller.refresh()
    controller.select('1_week')
    expect(deps.choose).not.toHaveBeenCalled()
    await controller.apply()
    expect(deps.choose).toHaveBeenCalledExactlyOnceWith(control.id, '1_week')
  })
  it('deduplicates submissions and retries the original choice after an uncertain response', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    const response = deferred<WisdomMuteSnapshot>()
    deps.choose.mockReturnValueOnce(response.promise)
    controller.select('1_day')
    const first = controller.apply()
    await controller.apply()
    controller.select('forever')
    expect(deps.choose).toHaveBeenCalledOnce()
    response.resolve({ ...snapshot, organization_id: 'wrong' })
    await first
    expect(state().retryingChoice).toBe(true)
    controller.select('forever')
    await controller.apply()
    expect(deps.choose.mock.calls).toEqual([
      [control.id, '1_day'],
      [control.id, '1_day']
    ])
  })
  it('retains last known shared state through read failures', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    deps.read.mockRejectedValue(new Error('offline'))
    await controller.poll()
    expect(state().snapshot?.mute?.muted).toBe(false)
    expect(state().error).toBe(true)
  })
  it('ignores an old poll after a new explicit refresh', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    const old = deferred<WisdomMuteSnapshot>()
    deps.read.mockReturnValue(old.promise)
    const reading = controller.poll()
    await controller.refresh()
    old.resolve({ ...snapshot, mute: { ...control.mute, revision: 0 } })
    await reading
    expect(state().control?.mute.revision).toBe(3)
  })
  it('invalidates stale revision and expired menus without rebasing the choice', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    controller.select('forever')
    deps.read.mockResolvedValue({ ...snapshot, mute: { ...control.mute, revision: 4 } })
    await controller.poll()
    expect(state().control).toBeNull()
    await controller.apply()
    expect(deps.choose).not.toHaveBeenCalled()
    await controller.refresh()
    controller.select('forever')
    deps.now = () => 3000_000
    // A separate controller uses the new clock at construction.
    const expired = createWisdomMuteController(deps)
    await expired.refresh()
    expect(state().control).toBeNull()
    expect(state().error).toBe(true)
  })
  it('drops foreign organization state and never updates disposed surfaces', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    deps.read.mockResolvedValue({ ...snapshot, organization_id: 'other' })
    await controller.poll()
    expect(state().snapshot).toBeNull()
    const old = deferred<WisdomMuteControl>()
    deps.prepare.mockReturnValue(old.promise)
    const preparing = controller.refresh()
    controller.dispose()
    const before = state()
    old.resolve(control)
    await preparing
    expect(state()).toBe(before)
  })
  it('does not silently renew after a rejected choice; pending state stays visible', async () => {
    const { controller, deps, state } = setup()
    await controller.refresh()
    deps.choose.mockResolvedValue({
      ...snapshot,
      sync: {
        mutation_id: control.id,
        requested_duration: 'forever',
        requested_until: null,
        request_expires_at: 9000,
        preference_sync: 'conflict'
      }
    })
    deps.prepare.mockRejectedValue(new Error('offline'))
    controller.select('forever')
    await controller.apply()
    expect(state().snapshot?.sync?.preference_sync).toBe('conflict')
    expect(state().control).toBeNull()
    expect(deps.choose).toHaveBeenCalledOnce()
  })
})
