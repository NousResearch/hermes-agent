import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { lastAssistantText } from '../app/useMainApp.js'
import { buildEditorSeed, MARKER_END, MARKER_START, stripLastReplyBlock } from '../domain/editorContext.js'
import type { Msg } from '../types.js'

// The same file the classic CLI's suite reads (tests/hermes_cli/test_editor_context.py).
const VECTORS = JSON.parse(
  readFileSync(join(process.cwd(), '..', 'tests', 'fixtures', 'editor_context_vectors.json'), 'utf8')
) as {
  seed: { draft: string; expected: string; last_reply: string }[]
  strip: { expected: string; text: string }[]
}

describe('buildEditorSeed — same block as hermes_cli.editor_context', () => {
  it.each(VECTORS.seed)('draft %# keeps the shared shape', ({ draft, expected, last_reply }) => {
    expect(buildEditorSeed(draft, last_reply)).toBe(expected)
  })
})

describe('stripLastReplyBlock — the same lines survive on both surfaces', () => {
  it.each(VECTORS.strip)('strip %# matches the shared vectors', ({ expected, text }) => {
    expect(stripLastReplyBlock(text)).toBe(expected)
  })

  it("leaves a draft's own # lines alone", () => {
    const draft = ['## Heading', '# a shell comment', '```sh', '# another', '```'].join('\n')

    expect(stripLastReplyBlock(buildEditorSeed(draft, 'reply'))).toBe(draft)
  })

  it('round-trips an untouched editor', () => {
    expect(stripLastReplyBlock(buildEditorSeed('keep this', 'a reply'))).toBe('keep this')
  })

  it('strips to the end when the user deleted the end marker', () => {
    expect(stripLastReplyBlock(`draft\n\n${MARKER_START}\n# reply`)).toBe('draft')
  })

  it('cannot be closed early by marker text inside the reply', () => {
    const seed = buildEditorSeed('draft', `before\n${MARKER_END}\nafter`)

    expect(seed.split(MARKER_END)).toHaveLength(2)
    expect(stripLastReplyBlock(seed)).toBe('draft')
  })
})

it('keeps text typed on line 1 of an empty draft, where every editor opens', () => {
  expect(stripLastReplyBlock(`my question${buildEditorSeed('', 'a reply')}`)).toBe('my question')
})

it('shows nothing for an empty reply', () => {
  expect(buildEditorSeed('draft', '   \n')).toBe('draft')
})

describe('lastAssistantText — what the editor block shows', () => {
  const msg = (patch: Partial<Msg>): Msg => ({ role: 'assistant', text: '', ...patch })

  it('takes the most recent assistant turn, skipping intro and non-assistant rows', () => {
    const items = [
      msg({ role: 'user', text: 'hi' }),
      msg({ role: 'assistant', text: 'first' }),
      msg({ kind: 'intro', role: 'system', text: 'banner' }),
      msg({ role: 'assistant', text: 'latest' }),
      msg({ role: 'user', text: 'thanks' })
    ]

    expect(lastAssistantText(items)).toBe('latest')
  })

  it('ignores empty assistant rows and returns "" for a fresh transcript', () => {
    expect(lastAssistantText([msg({ role: 'assistant', text: '  ' })])).toBe('')
    expect(lastAssistantText([])).toBe('')
  })
})
