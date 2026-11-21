import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { getInputSelection } from '../app/inputSelectionStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { TextInput } from '../components/textInput.js'
import { expandTokens } from '../domain/attachments.js'

class Input extends EventEmitter {
  chunks: string[] = []
  isTTY = true
  readableLength = 0
  read() {
    return this.chunks.shift() ?? null
  }
  ref() {}
  unref() {}
  setEncoding() {}
  setRawMode() {}
  send(text: string) {
    this.chunks.push(text)
    this.emit('readable')
  }
}

function mount(initial = '', initialColumns = 40) {
  resetUiState()
  patchUiState({ sid: 'edge-test' })
  const stdin = new Input()
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })
  const submit = vi.fn()
  const request = vi.fn(async () => ({}))
  let composer!: ReturnType<typeof useComposerState>
  let columns = initialColumns
  let initialized = false

  function Harness() {
    composer = useComposerState({ gw: { request } as any, submitRef: { current: submit }, sys: vi.fn() })

    if (!initialized) {
      initialized = true

      if (initial) {
        composer.actions.setInput(initial)
      }
    }

    return (
      <TextInput
        columns={columns}
        onChange={composer.actions.setInput}
        onPaste={composer.actions.handleTextPaste}
        onSubmit={submit}
        value={composer.state.input}
      />
    )
  }

  const instance = renderSync(<Harness />, {
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stderr: stdout as unknown as NodeJS.WriteStream,
    patchConsole: false
  })

  return {
    stdin,
    submit,
    request,
    get composer() {
      return composer
    },
    resize(next: number) {
      columns = next
      instance.rerender(<Harness />)
    },
    close() {
      instance.unmount()
      instance.cleanup()
      resetUiState()
    }
  }
}

const caret = async (position: number) => {
  await vi.waitFor(() => expect(getInputSelection()?.start).toBe(position))
}

it('replaces selected input with a bracketed paste without submitting', async () => {
  const h = mount('red blue green')

  try {
    await caret(14)

    for (let i = 0; i < 6; i++) {
      h.stdin.send('\u001b[D')
      await caret(13 - i)
    }

    for (let i = 0; i < 4; i++) {
      h.stdin.send('\u001b[1;2D')
      await caret(7 - i)
    }

    expect(getInputSelection()?.end).toBe(8)
    h.stdin.send('\u001b[200~pink\u001b[201~')
    await vi.waitFor(() => expect(h.composer.state.input).toBe('red pink green'))
    expect(h.submit).not.toHaveBeenCalled()
  } finally {
    h.close()
  }
})

it('pastes three lines without submitting, then edits the middle line with Up', async () => {
  const h = mount()

  try {
    await caret(0)
    h.stdin.send('\u001b[200~aaa\r\nbbb\r\nccc\u001b[201~')
    await caret(11)
    expect(h.composer.state.input).toBe('aaa\nbbb\nccc')
    expect(h.submit).not.toHaveBeenCalled()
    h.stdin.send('\u001b[A')
    await caret(7)
    h.stdin.send('X')
    await vi.waitFor(() => expect(h.composer.state.input).toBe('aaa\nbbbX\nccc'))
    expect(h.submit).not.toHaveBeenCalled()
  } finally {
    h.close()
  }
})

it('keeps every line of a collapsed paste in its token', async () => {
  const h = mount()
  const text = 'first\nsecond\nthird\nfourth\nfifth\nsixth\nlast'

  try {
    await caret(0)
    h.stdin.send(`\u001b[200~${text}\u001b[201~`)
    await vi.waitFor(() => expect(h.composer.refs.tokensRef.current).toHaveLength(1))
    const token = h.composer.refs.tokensRef.current[0]!
    expect(token).toMatchObject({ kind: 'paste', text })
    await vi.waitFor(() => expect(h.composer.state.input).toContain(token.label))
    expect(expandTokens(h.composer.refs.tokensRef.current)(h.composer.state.input)).toBe(text)
    expect(h.submit).not.toHaveBeenCalled()
  } finally {
    h.close()
  }
})

it('shows all seven lines when collapse thresholds are zero', async () => {
  const h = mount()
  const text = 'first\nsecond\nthird\nfourth\nfifth\nsixth\nlast'

  try {
    patchUiState({ pasteCollapseLines: 0, pasteCollapseChars: 0 })
    await caret(0)
    h.stdin.send(`\u001b[200~${text}\u001b[201~`)
    await vi.waitFor(() => expect(h.composer.state.input).toBe(text))
    expect(h.composer.refs.tokensRef.current).toHaveLength(0)
    expect(h.submit).not.toHaveBeenCalled()
  } finally {
    h.close()
  }
})

it('restores the desired column after crossing a short line', async () => {
  const text = 'abcdefghij\nx\nabcdefghij'
  const h = mount(text)

  try {
    await caret(text.length)
    h.stdin.send('\u001b[A')
    await caret(12)
    h.stdin.send('\u001b[A')
    await caret(10)
    h.stdin.send('\u001b[B')
    await caret(12)
    h.stdin.send('\u001b[B')
    await caret(text.length)
    expect(h.composer.state.input).toBe(text)
  } finally {
    h.close()
  }
})

it('moves and deletes whole emoji, combining accents and Arabic characters', async () => {
  const h = mount('a🧪e\u0301ب')

  try {
    await caret(6)
    h.stdin.send('\u001b[D')
    await caret(5)
    h.stdin.send('\u001b[D')
    await caret(3)
    h.stdin.send('\u001b[D')
    await caret(1)
    h.stdin.send('\u001b[C')
    await caret(3)
    h.stdin.send('\u001b[C')
    await caret(5)
    h.stdin.send('\u007f')
    await vi.waitFor(() => expect(h.composer.state.input).toBe('a🧪ب'))
    h.stdin.send('e\u0301')
    await vi.waitFor(() => expect(h.composer.state.input).toBe('a🧪e\u0301ب'))
  } finally {
    h.close()
  }
})

it('preserves the draft across narrow and wide layouts and uses the new wrapped rows', async () => {
  const text = 'abcdefghijklmnopqrstuvwxyz'
  const h = mount(text, 40)

  try {
    await caret(text.length)
    expect(getInputSelection()?.canMoveUp).toBe(false)
    h.resize(12)
    await vi.waitFor(() => expect(getInputSelection()?.canMoveUp).toBe(true))
    expect(h.composer.state.input).toBe(text)
    h.stdin.send('\u001b[A')
    await caret(14)
    h.stdin.send('X')
    const edited = 'abcdefghijklmnXopqrstuvwxyz'
    await vi.waitFor(() => expect(h.composer.state.input).toBe(edited))
    h.resize(80)
    await vi.waitFor(() => expect(getInputSelection()?.canMoveUp).toBe(false))
    expect(h.composer.state.input).toBe(edited)
    expect(h.submit).not.toHaveBeenCalled()
  } finally {
    h.close()
  }
})
