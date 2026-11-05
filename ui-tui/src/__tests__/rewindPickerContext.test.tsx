import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { rewindTargets, withReplies } from '../app/slash/canonicalRewindPicker.js'
import { RewindPicker } from '../components/rewindPicker.js'
import { DEFAULT_THEME } from '../theme.js'

// Native-scrollback mode: the transcript is the terminal's own scrollback, which the app cannot scroll,
// so the picker shows the selected prompt and the reply that followed it instead of a silent no-op.

// `messages` is the owner's display projection (`text`, compaction handoffs stripped); `stored_messages`
// the raw rows the rewind targets come from.
const snapshot = {
  messages: [
    { role: 'user', text: 'first', row_id: 11 },
    { role: 'assistant', text: '', row_id: 12, tool_calls: [{ id: 'c1' }] },
    { role: 'tool', text: 'tool output', row_id: 13 },
    { role: 'assistant', text: 'answer one', row_id: 14 },
    { role: 'user', text: '[image]', row_id: 15 },
    { role: 'assistant', text: 'media answer', row_id: 16 },
    { role: 'user', text: 'third', row_id: 17 }
  ],
  stored_messages: [
    { role: 'user', content: 'first', row_id: 11 },
    { role: 'assistant', content: '', row_id: 12, tool_calls: [{ id: 'c1' }] },
    { role: 'tool', content: 'tool output', row_id: 13 },
    { role: 'assistant', content: [{ type: 'output_text', text: 'answer one' }], row_id: 14 },
    { role: 'user', content: [{ type: 'input_image', image_url: 'x' }], row_id: 15 },
    { role: 'assistant', content: 'media answer', row_id: 16 },
    { role: 'user', content: 'third', row_id: 17 }
  ]
}

describe('rewind targets carry the reply that followed', () => {
  it('collects assistant text up to the next user turn, never tool output or a skipped turn’s reply', () => {
    const targets = withReplies(snapshot, rewindTargets(snapshot))

    expect(targets.map(({ reply, text }) => ({ reply, text }))).toEqual([
      { reply: 'answer one', text: 'first' },
      { reply: undefined, text: 'third' }
    ])
  })

  it('ends an exchange at an archived prompt and never shows a compaction summary as the reply', () => {
    // After a compaction: an archived prompt + answer sit between two live targets, and a handoff
    // summary row that the display projection dropped sits after the newer live prompt.
    const compacted = {
      messages: [
        { role: 'user', text: 'live one', row_id: 21 },
        { role: 'assistant', text: 'answer to live one', row_id: 22 },
        { archived: true, role: 'user', text: 'archived prompt', row_id: 5 },
        { archived: true, role: 'assistant', text: 'archived answer', row_id: 6 },
        { role: 'user', text: 'live two', row_id: 23 },
        { role: 'assistant', text: 'answer to live two', row_id: 25 }
      ],
      stored_messages: [
        { role: 'user', content: 'live one', row_id: 21 },
        { role: 'assistant', content: 'answer to live one', row_id: 22 },
        { archived: true, role: 'user', content: 'archived prompt', row_id: 5 },
        { archived: true, role: 'assistant', content: 'archived answer', row_id: 6 },
        { role: 'user', content: 'live two', row_id: 23 },
        { role: 'assistant', content: '[CONTEXT COMPACTION — REFERENCE ONLY] summary', row_id: 24 },
        { role: 'assistant', content: 'answer to live two', row_id: 25 }
      ]
    }

    const targets = withReplies(compacted, rewindTargets(compacted))

    expect(targets.map(({ reply, text }) => ({ reply, text }))).toEqual([
      { reply: 'answer to live one', text: 'live one' },
      { reply: 'answer to live two', text: 'live two' }
    ])
  })
})

// Components read the terminal height from `process.stdout` (useStdout), so the test sizes that.
const realRows = Object.getOwnPropertyDescriptor(process.stdout, 'rows')

afterEach(() => {
  if (realRows) {
    Object.defineProperty(process.stdout, 'rows', realRows)
  } else {
    delete (process.stdout as { rows?: number }).rows
  }
})

function mount(contextView: boolean) {
  Object.defineProperty(process.stdout, 'rows', { configurable: true, value: 30 })
  const lines = Array.from({ length: 30 }, (_, i) => `reply line ${i + 1}`).join('\n')

  const req = {
    targets: [
      { reply: lines, rowId: 11, text: 'older prompt', turn: 1 },
      { reply: 'short answer', rowId: 13, throughRowId: 12, text: 'newer prompt', turn: 2 }
    ],
    onFork: vi.fn(),
    onSubmit: vi.fn()
  }

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  const out = { text: '' }

  stdout.on('data', chunk => {
    out.text += chunk.toString()
  })

  const element = (
    <RewindPicker
      cols={100}
      contextView={contextView}
      onClose={vi.fn()}
      onFork={req.onFork}
      onSubmit={req.onSubmit}
      req={req}
      t={DEFAULT_THEME}
    />
  )

  const instance = renderSync(element, {
    stdin: stdin as any,
    stdout: stdout as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  // Each key is sent ONCE; only the wait retries, so a dropped key fails instead of being re-sent.
  const press = async (seq: string, expectText: string) => {
    out.text = ''
    stdin.write(seq)
    await vi.waitFor(() => {
      instance.rerender(React.cloneElement(element))
      expect(out.text).toContain(expectText)
    })
  }

  return { instance, out, press, req }
}

describe('native-scrollback rewind picker context view', () => {
  it('shows the selected exchange, follows the selection, and pages a long reply with PgDn/PgUp', async () => {
    const { instance, out, press, req } = mount(true)

    try {
      await vi.waitFor(() => expect(out.text).toContain('short answer'))
      expect(out.text).toContain('Hermes replied:')

      await press('\u001B[B', 'reply line 1')
      expect(out.text).toContain('older prompt')
      expect(out.text).not.toContain('reply line 30')

      // 30 rows leave a 12-row context window: the 34-row exchange takes two pages to reach its end.
      await press('\u001B[6~', 'lines 12–23 of 34')
      expect(out.text).not.toContain('reply line 30')
      await press('\u001B[6~', 'lines 23–34 of 34')
      expect(out.text).toContain('reply line 30')
      await press('\u001B[5~', 'lines 12–23 of 34')

      // Enter still opens the edit, not a send; nothing is cut by moving or scrolling.
      expect(req.onSubmit).not.toHaveBeenCalled()
      expect(req.onFork).not.toHaveBeenCalled()
    } finally {
      instance.unmount()
    }
  })

  it('keeps the compact prompt preview when the transcript itself scrolls to the prompt', async () => {
    const { instance, out } = mount(false)

    try {
      await vi.waitFor(() => expect(out.text).toContain('newer prompt'))
      expect(out.text).not.toContain('short answer')
      expect(out.text).not.toContain('Hermes replied:')
    } finally {
      instance.unmount()
    }
  })
})
