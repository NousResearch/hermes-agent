import { describe, expect, it, vi } from 'vitest'

import { createPetSingleFlight, requestPetUpdate } from '../lib/petPolling.js'

const gateway = (request: ReturnType<typeof vi.fn>) => ({ request }) as never

describe('requestPetUpdate', () => {
  it('does not enqueue pet.cells while pets are disabled', async () => {
    const request = vi.fn().mockResolvedValue({ enabled: false })
    const needsCells = vi.fn(() => true)

    const update = await requestPetUpdate(gateway(request), 'idle', false, needsCells)

    expect(update).toEqual({ cells: null, meta: { enabled: false } })
    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith('pet.info.meta')
    expect(needsCells).not.toHaveBeenCalled()
  })

  it('uses metadata only when the enabled state is already cached', async () => {
    const request = vi.fn().mockResolvedValue({
      enabled: true,
      scale: 0.33,
      slug: 'boba',
      spritesheetRevision: '1:2'
    })

    const update = await requestPetUpdate(gateway(request), 'idle', false, () => false)

    expect(update?.cells).toBeNull()
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('fetches cells only for an enabled uncached state', async () => {
    const cells = { enabled: true, frames: [], slug: 'boba' }
    const request = vi.fn().mockResolvedValueOnce({ enabled: true, slug: 'boba' }).mockResolvedValueOnce(cells)

    const update = await requestPetUpdate(gateway(request), 'review', false, () => true)

    expect(update?.cells).toEqual(cells)
    expect(request).toHaveBeenNthCalledWith(1, 'pet.info.meta')
    expect(request).toHaveBeenNthCalledWith(2, 'pet.cells', {
      graphics: false,
      state: 'review'
    })
  })

  it('silently drops cosmetic gateway failures', async () => {
    const request = vi.fn().mockRejectedValue(new Error('timeout: pet.info.meta'))

    await expect(requestPetUpdate(gateway(request), 'idle', false, () => true)).resolves.toBeNull()
  })
})

describe('createPetSingleFlight', () => {
  it('suppresses overlapping polls and permits the next completed poll', async () => {
    let release = () => undefined

    const blocked = new Promise<void>(resolve => {
      release = resolve
    })

    const operation = vi.fn(() => blocked)
    const run = createPetSingleFlight()

    const first = run(operation)
    await expect(run(operation)).resolves.toBe(false)
    expect(operation).toHaveBeenCalledTimes(1)

    release()
    await expect(first).resolves.toBe(true)
    await expect(run(async () => undefined)).resolves.toBe(true)
  })
})
