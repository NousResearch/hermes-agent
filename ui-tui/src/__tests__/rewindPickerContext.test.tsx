import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { rewindTargets, withReplies } from '../app/slash/canonicalRewindPicker.js'
import { RewindPicker } from '../components/rewindPicker.js'
import { DEFAULT_THEME } from '../theme.js'

// Native-scrollback mode: the transcript is the terminal's own scrollback, which the app cannot scroll,
// so the picker shows the selected prompt and the reply that followed it instead of a silent no-op.

const snapshot = {
  messages: [
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
})

function mount(contextView: boolean) {
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
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 40, isTTY: false })
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

  const press = async (seq: string, expectText: string) => {
    await vi.waitFor(() => {
      out.text = ''
      stdin.write(seq)
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

      await press('\u001B[6~', 'reply line 30')
      await press('\u001B[5~', 'reply line 1 ')

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
