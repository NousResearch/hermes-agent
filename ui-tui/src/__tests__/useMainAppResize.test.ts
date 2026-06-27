import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { subscribeResizeSignals } from '../app/useMainApp.js'

class MockStdout extends EventEmitter {
  override on(event: 'resize', listener: () => void): this {
    return super.on(event, listener)
  }

  override off(event: 'resize', listener: () => void): this {
    return super.off(event, listener)
  }
}

describe('subscribeResizeSignals', () => {
  it('listens to both stdout resize and SIGWINCH', () => {
    const stdout = new MockStdout()
    const signalBus = new EventEmitter()
    const onResize = vi.fn()

    const dispose = subscribeResizeSignals(stdout, onResize, signalBus)

    stdout.emit('resize')
    signalBus.emit('SIGWINCH')

    expect(onResize).toHaveBeenCalledTimes(2)

    dispose()
  })

  it('removes both listeners on cleanup', () => {
    const stdout = new MockStdout()
    const signalBus = new EventEmitter()
    const onResize = vi.fn()

    const dispose = subscribeResizeSignals(stdout, onResize, signalBus)
    dispose()

    stdout.emit('resize')
    signalBus.emit('SIGWINCH')

    expect(onResize).not.toHaveBeenCalled()
  })

  it('ignores unsupported SIGWINCH registration and still tracks stdout resize', () => {
    const stdout = new MockStdout()
    const onResize = vi.fn()

    const signalBus = {
      off: vi.fn(),
      on: vi.fn(() => {
        throw new Error('unsupported signal')
      })
    }

    const dispose = subscribeResizeSignals(stdout, onResize, signalBus)

    stdout.emit('resize')

    dispose()

    expect(onResize).toHaveBeenCalledTimes(1)
    expect(signalBus.off).not.toHaveBeenCalled()
  })
})
