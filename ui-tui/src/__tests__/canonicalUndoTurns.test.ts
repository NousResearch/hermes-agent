import { expect, it } from 'vitest'

import { lastTurnRewind, turnsBackRewind } from '../app/slash/canonicalSessionCommands.js'
import type { MutationSnapshot } from '../app/slash/canonicalSessionControls.js'

// `/undo N` on the shared gateway rewinds to the Nth-newest user turn (the classic CLI's and the
// messaging gateway's `/undo N`), so its file revert covers every undone turn, not only the last.
const snapshot = {
  messages: [
    { role: 'user', content: 'create a.txt', row_id: 1 },
    { role: 'assistant', content: 'done', row_id: 2 },
    { role: 'user', content: 'edit a.txt, create b.txt', row_id: 3 },
    { role: 'assistant', content: 'done', row_id: 4 }
  ]
} as unknown as MutationSnapshot

it('targets the Nth-newest user turn and clamps to the oldest', () => {
  expect(turnsBackRewind(1)(snapshot)).toEqual(lastTurnRewind(snapshot))
  expect(turnsBackRewind(2)(snapshot)).toEqual({ target_message_id: 1 })
  expect(turnsBackRewind(9)(snapshot)).toEqual({ target_message_id: 1 })
  expect(turnsBackRewind(1)({ messages: [] } as unknown as MutationSnapshot)).toBeNull()
})
