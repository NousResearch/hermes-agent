import { PassThrough } from 'node:stream'

import { AlternateScreen, renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type PasteEvent, TextInput } from '../components/textInput.js'
import * as clipboard from '../lib/clipboard.js'

vi.mock('../lib/clipboard.js', async () => {
  const actual = await vi.importActual<typeof clipboard>('../lib/clipboard.js')

  return { ...actual, readPrimarySelectionText: vi.fn() }
})

const readPrimarySelectionTextMock = vi.mocked(clipboard.readPrimarySelectionText)

function makeStdin() {
  const stdin = new PassThrough()

  Object.assign(stdin, {
    isRaw: false,
    isTTY: true,
    ref: vi.fn(),
    setRawMode: vi.fn((enabled: boolean) => {
      Object.assign(stdin, { isRaw: enabled })
    }),
    unref: vi.fn()
  })

  return stdin
}

function renderTextInput(
  onChange: (value: string) => void,
  onPaste: (event: PasteEvent) => { cursor: number; value: string }
) {
  const stdin = makeStdin()
  const stderr = new PassThrough()

  Object.assign(stderr, { isTTY: false })
  stderr.on('data', () => {})

  const writeSpy = vi
    .spyOn(process.stdout, 'write')
    .mockImplementation((() => true) as unknown as typeof process.stdout.write)

  const instance = renderSync(
    <AlternateScreen mouseTracking="buttons">
      <TextInput focus onChange={onChange} onPaste={onPaste} value="" />
    </AlternateScreen>,
    {
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: process.stdout
    }
  )

  const cleanup = () => {
    instance.unmount()
    instance.cleanup()
    writeSpy.mockRestore()
  }

  return { cleanup, stdin }
}

function middleClick(stdin: PassThrough) {
  stdin.write('\x1b[<1;1;1M')
}

afterEach(() => {
  readPrimarySelectionTextMock.mockReset()
})

describe('TextInput middle-click paste', () => {
  it('does not fall back to CLIPBOARD when PRIMARY succeeds with empty text', async () => {
    const onChange = vi.fn()
    const onPaste = vi.fn(() => ({ cursor: 0, value: 'clipboard fallback' }))
    readPrimarySelectionTextMock.mockResolvedValue('')
    const { cleanup, stdin } = renderTextInput(onChange, onPaste)

    try {
      middleClick(stdin)
      await vi.waitFor(() => expect(readPrimarySelectionTextMock).toHaveBeenCalledOnce())

      expect(onChange).not.toHaveBeenCalled()
      expect(onPaste).not.toHaveBeenCalled()
    } finally {
      cleanup()
    }
  })

  it('uses the ordinary paste pipeline only when PRIMARY is unavailable', async () => {
    const onChange = vi.fn()
    const onPaste = vi.fn(() => ({ cursor: 8, value: 'fallback' }))
    readPrimarySelectionTextMock.mockResolvedValue(null)
    const { cleanup, stdin } = renderTextInput(onChange, onPaste)

    try {
      middleClick(stdin)
      await vi.waitFor(() => expect(onPaste).toHaveBeenCalledOnce())

      expect(onPaste).toHaveBeenCalledWith({ cursor: 0, hotkey: true, text: '', value: '' })
    } finally {
      cleanup()
    }
  })
})
