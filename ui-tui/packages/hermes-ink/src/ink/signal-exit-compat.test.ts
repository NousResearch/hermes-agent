import { describe, expect, it, vi } from 'vitest'

import { resolveOnProcessExit, type SignalExitHandler } from './signal-exit-compat.js'

describe('resolveOnProcessExit', () => {
  it('uses signal-exit v4 named onExit export', () => {
    const unsubscribe = vi.fn()
    const onExit = vi.fn(() => unsubscribe)
    const handler: SignalExitHandler = () => undefined

    const resolved = resolveOnProcessExit({ onExit })
    const result = resolved(handler, { alwaysLast: false })

    expect(result).toBe(unsubscribe)
    expect(onExit).toHaveBeenCalledWith(handler, { alwaysLast: false })
  })

  it('uses signal-exit v3 default export', () => {
    const unsubscribe = vi.fn()
    const defaultExport = vi.fn(() => unsubscribe)
    const handler: SignalExitHandler = () => undefined

    const resolved = resolveOnProcessExit({ default: defaultExport })
    const result = resolved(handler, { alwaysLast: true })

    expect(result).toBe(unsubscribe)
    expect(defaultExport).toHaveBeenCalledWith(handler, { alwaysLast: true })
  })

  it('throws when no compatible export exists', () => {
    expect(() => resolveOnProcessExit({})).toThrow('Unsupported signal-exit export shape')
  })
})
