import { describe, expect, it } from 'vitest'

import { mutateCanonical, retireCanonicalControls } from '../app/slash/canonicalSessionControls.js'

// A lost-reply control may be retried only as the session's very next control: after another verb,
// re-issuing it must mint a fresh request_id instead of replaying the stale receipt.
function gateway() {
  const mutations: Array<Record<string, unknown>> = []
  let lose = true

  return {
    mutations,
    deliver: () => (lose = false),
    request: async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { revision: 7, execution_generation: 3 }
      }

      mutations.push(params!)

      if (lose) {
        throw new Error('transport closed') // a lost reply: not a structured refusal
      }

      return { revision: 8, execution_generation: 3 }
    }
  }
}

describe('retained canonical controls', () => {
  it('another control retires a lost-reply control', async () => {
    const gw = gateway()
    await expect(mutateCanonical(gw, 's', 'model', { model: 'A' })).rejects.toThrow()
    gw.deliver()
    await mutateCanonical(gw, 's', 'model', { model: 'B' })
    await mutateCanonical(gw, 's', 'model', { model: 'A' })
    expect(gw.mutations[2].request_id).not.toBe(gw.mutations[0].request_id)
  })

  it('a prompt submit retires the session lost-reply controls', async () => {
    const gw = gateway()
    await expect(mutateCanonical(gw, 's', 'model', { model: 'A' })).rejects.toThrow()
    gw.deliver()
    retireCanonicalControls(gw, 's')
    await mutateCanonical(gw, 's', 'model', { model: 'A' })
    expect(gw.mutations[1].request_id).not.toBe(gw.mutations[0].request_id)
  })

  it('the immediate retry of the same control keeps its identity', async () => {
    const gw = gateway()
    await expect(mutateCanonical(gw, 's', 'model', { model: 'A' })).rejects.toThrow()
    gw.deliver()
    await mutateCanonical(gw, 's', 'model', { model: 'A' })
    expect(gw.mutations[1].request_id).toBe(gw.mutations[0].request_id)
  })

  // Another viewer's write between the fence read and the write: a value pick is re-sent at the new
  // revision (setting the same value again is harmless); a CAS edit of shared metadata is not.
  it.each<['model' | 'rename', Record<string, unknown>, number]>([
    ['model', { model: 'A' }, 2],
    ['rename', { title: 'mine' }, 1]
  ])('a %s %o refused at a moved fence is sent %i time(s)', async (operation, payload, sends) => {
    let revision = 7
    const mutations: Array<Record<string, unknown>> = []

    const gw = {
      request: async (method: string, params?: Record<string, unknown>) => {
        if (method === 'session.resume') {
          return { revision, execution_generation: 3 }
        }

        mutations.push(params!)

        if (mutations.length === 1) {
          revision = 8 // someone else's write landed first

          throw Object.assign(new Error('revision_conflict'), { data: { reason: 'revision_conflict', revision } })
        }

        return { revision: revision + 1, execution_generation: 3 }
      }
    }

    const outcome = mutateCanonical(gw, 's', operation, payload)
    await (sends === 2 ? expect(outcome).resolves.toBeTruthy() : expect(outcome).rejects.toThrow('revision_conflict'))
    expect(mutations.map(m => m.expected_revision)).toEqual(sends === 2 ? [7, 8] : [7])
  })
})
