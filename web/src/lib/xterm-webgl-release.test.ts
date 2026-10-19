import { describe, expect, it, vi } from 'vitest'

import { loseWebglContexts } from './xterm-webgl-release'

function canvasWith(gl: unknown) {
  return {
    getContext: vi.fn((type: string) => (type === 'webgl2' ? gl : null))
  }
}

describe('loseWebglContexts', () => {
  it('loses every WebGL context under the host and leaves 2D canvases alone', () => {
    // Dashboard reconnect churn (#111909): xterm disposes its WebGL canvas
    // without losing the context, so the browser's context budget fills up.
    const loseContext = vi.fn()
    const gl = { getExtension: vi.fn((name: string) => (name === 'WEBGL_lose_context' ? { loseContext } : null)) }
    const twoD = canvasWith(null)
    const host = { querySelectorAll: () => [canvasWith(gl), twoD, canvasWith(gl)] } as unknown as ParentNode

    expect(loseWebglContexts(host)).toBe(2)
    expect(loseContext).toHaveBeenCalledTimes(2)
    // A canvas that holds no WebGL context is never handed one.
    expect(twoD.getContext).toHaveBeenCalledWith('webgl2')
    expect(twoD.getContext).toHaveBeenCalledWith('webgl')
  })
})
