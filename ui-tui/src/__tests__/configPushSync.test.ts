import { EventEmitter } from 'node:events'
import { afterEach, expect, it, vi } from 'vitest'

const { effects } = vi.hoisted(() => ({ effects: [] as Array<() => void | (() => void)> }))
vi.mock('react', async original => ({
  ...(await original<typeof import('react')>()),
  useEffect: (fn: () => void | (() => void)) => effects.push(fn),
  useRef: (current: unknown) => ({ current })
}))
vi.mock('../app/createGatewayEventHandler.js', () => ({ applyConfiguredTuiTheme: vi.fn() }))
import { useConfigSync } from '../app/useConfigSync.js'

let cleanup: Array<() => void> = []
afterEach(() => {
  cleanup.forEach(fn => fn())
  cleanup = []
  effects.length = 0
  vi.useRealTimers()
})

function mount(push = true) {
  vi.useFakeTimers()
  const data = { mtime: 1, mcp_rev: 'a', change_events: push }
  const config = { display: { bell_on_complete: false } }
  const reload = vi.fn(async () => ({ status: 'reloaded', loaded_rev: data.mcp_rev }))
  const full = vi.fn(async () => ({ config }))
  const gw = Object.assign(new EventEmitter(), {
    request: vi.fn(async (method: string, params: Record<string, unknown>) =>
      method === 'reload.mcp' ? await reload() : params.key === 'mtime' ? { ...data } : await full()
    )
  })
  const bell = vi.fn()
  useConfigSync({ gw: gw as any, sid: 's', setBellOnComplete: bell, setVoiceEnabled: vi.fn() })
  cleanup.push(
    ...effects
      .splice(0)
      .map(fn => fn())
      .filter((fn): fn is () => void => typeof fn === 'function')
  )
  return { gw, data, config, reload, full, bell }
}

it('uses a slow backstop instead of steady five-second config polling', async () => {
  const { gw } = mount()
  await vi.advanceTimersByTimeAsync(0)
  gw.request.mockClear()
  await vi.advanceTimersByTimeAsync(30_000)
  expect(gw.request.mock.calls.filter(([, params]) => params.key === 'mtime')).toHaveLength(0)
  await vi.advanceTimersByTimeAsync(30_000)
  expect(gw.request.mock.calls.filter(([, params]) => params.key === 'mtime')).toHaveLength(1)
})

it('keeps multiple idle clients off the five-second polling clock', async () => {
  const first = mount()
  const second = mount()
  await vi.advanceTimersByTimeAsync(0)
  first.gw.request.mockClear()
  second.gw.request.mockClear()
  await vi.advanceTimersByTimeAsync(30_000)
  expect(first.gw.request).not.toHaveBeenCalled()
  expect(second.gw.request).not.toHaveBeenCalled()
  first.gw.emit('event', { type: 'config.changed' })
  second.gw.emit('event', { type: 'config.changed' })
  await vi.advanceTimersByTimeAsync(0)
  expect(first.full).toHaveBeenCalledTimes(2)
  expect(second.full).toHaveBeenCalledTimes(2)
})

it('rehydrates cosmetic events without MCP reconnection', async () => {
  const { gw, data, config, reload, bell } = mount()
  await vi.advanceTimersByTimeAsync(0)
  data.mtime++
  config.display.bell_on_complete = true
  gw.emit('event', { type: 'config.changed' })
  await vi.advanceTimersByTimeAsync(0)
  expect(bell).toHaveBeenLastCalledWith(true)
  expect(reload).not.toHaveBeenCalled()
})

it.each(['throw', 'unconfirmed', 'wrong-revision'])(
  'retries %s MCP reloads until the revision is confirmed, then becomes idle',
  async failure => {
    const { gw, data, reload } = mount()
    await vi.advanceTimersByTimeAsync(0)
    data.mcp_rev = 'b'
    reload.mockImplementationOnce(async () => {
      if (failure === 'throw') throw new Error('temporary')
      return { status: failure === 'unconfirmed' ? 'confirm_required' : 'reloaded', loaded_rev: 'a' }
    })
    gw.emit('event', { type: 'config.changed' })
    await vi.advanceTimersByTimeAsync(0)
    expect(reload).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(5000)
    expect(reload).toHaveBeenCalledTimes(2)
    gw.request.mockClear()
    await vi.advanceTimersByTimeAsync(30_000)
    expect(gw.request).not.toHaveBeenCalled()
  }
)

it('retains polling for older gateways', async () => {
  const { gw, data, config, bell } = mount(false)
  await vi.advanceTimersByTimeAsync(0)
  delete (data as Partial<typeof data>).change_events
  data.mtime++
  config.display.bell_on_complete = true
  await vi.advanceTimersByTimeAsync(5000)
  expect(bell).toHaveBeenLastCalledWith(true)
  expect(gw.request.mock.calls.filter(([, params]) => params.key === 'mtime')).toHaveLength(2)
})

it('retains last-good display and retries failed hydration', async () => {
  const { gw, data, full, bell } = mount()
  await vi.advanceTimersByTimeAsync(0)
  bell.mockClear()
  full.mockRejectedValueOnce(new Error('offline'))
  data.mtime++
  gw.emit('event', { type: 'config.changed' })
  await vi.advanceTimersByTimeAsync(0)
  expect(bell).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(5000)
  expect(bell).toHaveBeenCalledOnce()
})

it('queues events arriving during a hydration and rechecks on reconnect', async () => {
  const { gw, full } = mount()
  await vi.advanceTimersByTimeAsync(0)
  let release!: (value: any) => void
  full.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        release = resolve
      })
  )
  gw.emit('event', { type: 'config.changed' })
  await vi.advanceTimersByTimeAsync(0)
  gw.emit('event', { type: 'config.changed' })
  release({ config: { display: {} } })
  await vi.advanceTimersByTimeAsync(0)
  expect(full).toHaveBeenCalledTimes(3)
  gw.emit('event', { type: 'gateway.ready' })
  await vi.advanceTimersByTimeAsync(0)
  expect(full).toHaveBeenCalledTimes(4)
})

it('ignores late results and removes timers and subscriptions on disposal', async () => {
  const { gw, full, bell } = mount()
  await vi.advanceTimersByTimeAsync(0)
  let release!: (value: any) => void
  full.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        release = resolve
      })
  )
  gw.emit('event', { type: 'config.changed' })
  await vi.advanceTimersByTimeAsync(0)
  cleanup.forEach(fn => fn())
  cleanup = []
  bell.mockClear()
  release({ config: { display: {} } })
  await vi.advanceTimersByTimeAsync(60_000)
  expect(bell).not.toHaveBeenCalled()
  expect(gw.listenerCount('event')).toBe(0)
  expect(vi.getTimerCount()).toBe(0)
})
