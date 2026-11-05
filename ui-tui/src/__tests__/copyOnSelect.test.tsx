import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { $copyFeedback, clearCopyFeedback } from '../app/copyFeedbackStore.js'
import { $uiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay } from '../app/useConfigSync.js'
import { useCopyOnSelect } from '../app/useCopyOnSelect.js'
import { TextInput, type TextInputMouseApi } from '../components/textInput.js'
import * as clipboard from '../lib/clipboard.js'
import * as osc52 from '../lib/osc52.js'

afterEach(() => {
  clearCopyFeedback()
  resetUiState()
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

const configure = (copy_on_select: 'auto' | boolean) =>
  applyDisplay({ config: { display: { copy_on_select } } }, () => {})

const mount = (element: React.ReactNode) => {
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })

  return renderSync(element, {
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream,
    patchConsole: false
  })
}

it('copies completed transcript selections once, keeps the highlight, and honors config reloads', async () => {
  const listeners = new Set<() => void>()
  let selected = false
  let isDragging = false
  let version = 0
  const notify = () => listeners.forEach(listener => listener())

  const copy = vi.fn(async () => {
    notify() // A reentrant notification must not copy the same version again.

    return 'selected transcript'
  })

  const selection = {
    copySelectionNoClear: copy,
    getState: () => ({ isDragging }),
    hasSelection: () => selected,
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    },
    version: () => version
  }

  const select = (dragging: boolean) => {
    selected = true
    isDragging = dragging
    version += 1
    notify()
  }

  function Harness() {
    useCopyOnSelect(selection, useStore($uiState).copyOnSelect)

    return <Text>transcript</Text>
  }

  configure(true)
  const view = mount(<Harness />)

  try {
    await vi.waitFor(() => expect(listeners.size).toBe(1))
    select(true)
    expect(copy).not.toHaveBeenCalled()
    select(false)
    notify()
    expect(copy).toHaveBeenCalledOnce()
    expect(selected).toBe(true)
    await vi.waitFor(() => expect($copyFeedback.get()).toBe('terminal'))
    selected = false
    version += 1
    notify()
    expect(copy).toHaveBeenCalledOnce()

    configure(false)
    await vi.waitFor(() => expect(listeners.size).toBe(0))
    select(false)
    expect(copy).toHaveBeenCalledOnce()
    configure(true)
    await vi.waitFor(() => expect(listeners.size).toBe(1))
    expect(copy).toHaveBeenCalledOnce() // Enabling does not overwrite the clipboard with an old selection.
    select(false)
    expect(copy).toHaveBeenCalledTimes(2)
  } finally {
    view.unmount()
    view.cleanup()
  }

  expect(listeners.size).toBe(0)
})

it('copies composer drags to the SSH terminal on release only, and respects opt-out', async () => {
  vi.stubEnv('SSH_CONNECTION', '127.0.0.1 1234 127.0.0.1 22')
  const nativeCopy = vi.spyOn(clipboard, 'writeClipboardText').mockResolvedValue(true)
  const remoteCopy = vi.spyOn(osc52, 'writeOsc52Clipboard').mockReturnValue(true)
  const mouseApiRef = { current: null as TextInputMouseApi | null }
  configure(true)
  const view = mount(<TextInput mouseApiRef={mouseApiRef} onChange={() => {}} value="hello world" />)

  try {
    await vi.waitFor(() => expect(mouseApiRef.current).not.toBeNull())
    mouseApiRef.current!.startAtBeginning()
    mouseApiRef.current!.dragAt(0, 5)
    expect(remoteCopy).not.toHaveBeenCalled()
    mouseApiRef.current!.end()
    expect(remoteCopy).toHaveBeenCalledExactlyOnceWith('hello')
    expect(nativeCopy).not.toHaveBeenCalled()
    await vi.waitFor(() => expect($copyFeedback.get()).toBe('terminal'))

    configure(false)
    mouseApiRef.current!.startAtBeginning()
    mouseApiRef.current!.dragAt(0, 3)
    mouseApiRef.current!.end()
    expect(remoteCopy).toHaveBeenCalledOnce()
    configure(true)
    mouseApiRef.current!.startAtBeginning()
    mouseApiRef.current!.end() // A click without a selection must not replace the clipboard.
    expect(remoteCopy).toHaveBeenCalledOnce()
  } finally {
    view.unmount()
    view.cleanup()
  }
})

it('does not automatically copy masked composer text', async () => {
  const nativeCopy = vi.spyOn(clipboard, 'writeClipboardText').mockResolvedValue(true)
  const remoteCopy = vi.spyOn(osc52, 'writeOsc52Clipboard').mockReturnValue(true)
  const mouseApiRef = { current: null as TextInputMouseApi | null }
  configure(true)
  const view = mount(<TextInput mask="*" mouseApiRef={mouseApiRef} onChange={() => {}} value="secret" />)

  try {
    await vi.waitFor(() => expect(mouseApiRef.current).not.toBeNull())
    mouseApiRef.current!.startAtBeginning()
    mouseApiRef.current!.dragAt(0, 6)
    mouseApiRef.current!.end()
    expect(nativeCopy).not.toHaveBeenCalled()
    expect(remoteCopy).not.toHaveBeenCalled()
  } finally {
    view.unmount()
    view.cleanup()
  }
})
