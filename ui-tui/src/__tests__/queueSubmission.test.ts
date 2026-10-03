import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import {
  applyQueueCommand,
  expandPasteTokens,
  prepareSlashSubmission,
  queueItemFromSlash
} from '../app/useSubmission.js'
import { imageToken } from '../domain/attachments.js'
import { queueItem } from '../hooks/useQueue.js'

describe('/queue collapsed paste submission', () => {
  it('keeps the collapsed argument for display and the full multiline payload for execution', () => {
    const display = '[[ first.. [3 lines] .. last ]]'

    expect(queueItemFromSlash(`/queue ${display}`, '/queue first\nmiddle\nlast')).toEqual({
      display,
      text: 'first\nmiddle\nlast'
    })
  })

  it('supports the /q alias and rejects an empty queue command', () => {
    expect(queueItemFromSlash('/q [[ payload ]]', '/q complete payload')).toEqual({
      display: '[[ payload ]]',
      text: 'complete payload'
    })
    expect(queueItemFromSlash('/queue', '/queue')).toBeUndefined()
  })

  it('expands paste tokens without consuming image tokens', () => {
    const paste: ComposerToken = { kind: 'paste', label: '[[ paste [2 lines] ]]', text: 'one\ntwo' }
    const image: ComposerToken = { kind: 'image', index: 1, label: imageToken(1), path: '/tmp/image.png' }

    expect(expandPasteTokens([paste, image])(`${paste.label} and ${image.label}`)).toBe(`one\ntwo and ${image.label}`)
  })
})

describe('prepareSlashSubmission', () => {
  const label = '[[ Done — verified.. [412 lines] .. already on it. ]]'
  const text = 'Done — verified through the real resolver\nline two\nline three'
  const tokens: ComposerToken[] = [{ kind: 'paste', label, text }]

  // The reported bug: `/pr-triage <paste>` dispatched the LABEL, so the skill
  // received "[412 lines]" as its argument and the agent reported the paste as
  // truncated. The command has to carry the full text; only the transcript
  // stays collapsed.
  it('dispatches the full paste while the transcript keeps the collapsed label', () => {
    expect(prepareSlashSubmission(`/pr-triage ${label}`, tokens)).toEqual({
      command: `/pr-triage ${text}`,
      display: `/pr-triage ${label}`
    })
  })

  it('leaves image tokens as labels — the gateway already holds the file', () => {
    const image: ComposerToken = { kind: 'image', index: 1, label: imageToken(1), path: '/tmp/shot.png' }

    expect(prepareSlashSubmission(`/pr-triage ${image.label}`, [image]).command).toBe(`/pr-triage ${image.label}`)
  })

  it('is a no-op on a token-free command', () => {
    expect(prepareSlashSubmission('/model opus', [])).toEqual({ command: '/model opus', display: '/model opus' })
  })
})

describe('/queue management verbs (applyQueueCommand)', () => {
  // Mirrors the classic CLI's _cmd_queue routing (#132026): the registry
  // advertises list/edit/rm/move/clear/add, so the same input must manage the
  // queue here instead of enqueueing "list" as a literal prompt.
  const q = (...texts: string[]) => texts.map(text => queueItem(text))

  it('lists on a bare /queue and on /queue list / ls / show', () => {
    expect(applyQueueCommand('/queue', '/queue', q())).toBe('Queue is empty.')
    expect(applyQueueCommand('/queue list', '/queue list', q('one', 'two'))).toBe(
      'Queued prompts (2):\n  1. one\n  2. two'
    )
    expect(applyQueueCommand('/queue ls', '/queue ls', q('one'))).toContain('1. one')
    expect(applyQueueCommand('/queue show', '/queue show', q('one'))).toContain('1. one')
  })

  it('clears the queue only when clear stands alone', () => {
    const queue = q('one', 'two')

    expect(applyQueueCommand('/queue clear', '/queue clear', queue)).toBe('Cleared 2 queued prompts.')
    expect(queue).toEqual([])
    expect(applyQueueCommand('/queue clear the logs', '/queue clear the logs', q('x'))).toBeUndefined()
  })

  it('removes by 1-based index and reports out-of-range or usage', () => {
    const queue = q('one', 'two')

    expect(applyQueueCommand('/queue rm 1', '/queue rm 1', queue)).toBe('Removed #1: "one"')
    expect(queue.map(item => item.text)).toEqual(['two'])
    expect(applyQueueCommand('/queue del 5', '/queue del 5', queue)).toBe('No queued prompt #5 — queue has 1.')
    expect(applyQueueCommand('/queue pop abc', '/queue pop abc', queue)).toBeUndefined()
    expect(applyQueueCommand('/queue rm 1 2', '/queue rm 1 2', queue)).toBe('usage: /queue rm <n>')
    expect(applyQueueCommand('/queue rm', '/queue rm', queue)).toBe('usage: /queue rm <n>')
  })

  it('edits item N, keeping the collapsed display when the argument carried a paste', () => {
    const queue = q('old prompt')

    expect(applyQueueCommand('/queue edit 1 new prompt', '/queue edit 1 new prompt', queue)).toBe(
      'Updated #1: "new prompt"'
    )
    expect(queue[0]).toEqual({ display: 'new prompt', text: 'new prompt' })

    const pasted = q('old')

    expect(applyQueueCommand('/queue set 1 [[ payload ]]', '/queue set 1 complete payload', pasted)).toBe(
      'Updated #1: "complete payload"'
    )
    expect(pasted[0]).toEqual({ display: '[[ payload ]]', text: 'complete payload' })

    expect(applyQueueCommand('/queue edit 1', '/queue edit 1', q('x'))).toBe('usage: /queue edit <n> <prompt>')
    expect(applyQueueCommand('/queue edit the config', '/queue edit the config', q('x'))).toBeUndefined()
    expect(applyQueueCommand('/queue edit 9 later', '/queue edit 9 later', q('x'))).toBe(
      'No queued prompt #9 — queue has 1.'
    )
  })

  it('moves an item between positions and validates both indices', () => {
    const queue = q('a', 'b', 'c')

    expect(applyQueueCommand('/queue move 1 3', '/queue move 1 3', queue)).toBe('Moved #1 → #3.')
    expect(queue.map(item => item.text)).toEqual(['b', 'c', 'a'])
    expect(applyQueueCommand('/queue move 1 9', '/queue move 1 9', queue)).toBe(
      'Move out of range — queue has 3 queued prompts.'
    )
    expect(applyQueueCommand('/queue move 1', '/queue move 1', queue)).toBe('usage: /queue move <from> <to>')
    expect(applyQueueCommand('/queue move a b', '/queue move a b', queue)).toBeUndefined()
  })

  it('keeps add and unknown words on the enqueue path', () => {
    expect(applyQueueCommand('/queue add hello', '/queue add hello', q())).toBeUndefined()
    expect(applyQueueCommand('/queue add', '/queue add', q())).toBe('usage: /queue add <prompt>')
    expect(applyQueueCommand('/queue hello world', '/queue hello world', q())).toBeUndefined()
  })
})
