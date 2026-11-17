import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSelectionState } from '../selection.js'
import { getTerminalFocusState, resetTerminalFocusState } from '../terminal-focus-state.js'
import { FOCUS_IN, FOCUS_OUT } from '../termio/csi.js'

import App from './App.js'

function makeApp(onTerminalFocusChange = vi.fn()) {
  const stdin = {
    isTTY: true,
    readableLength: 0,
    read: vi.fn(),
    ref: vi.fn(),
    unref: vi.fn(),
    setEncoding: vi.fn(),
    setRawMode: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn()
  } as unknown as NodeJS.ReadStream

  const stdout = {
    isTTY: true,
    columns: 80,
    rows: 24,
    write: vi.fn(),
    on: vi.fn(),
    off: vi.fn()
  } as unknown as NodeJS.WriteStream

  return new App({
    children: null,
    dispatchKeyboardEvent: vi.fn(),
    exitOnCtrlC: false,
    getHyperlinkAt: vi.fn(),
    onClickAt: vi.fn(() => false),
    onCursorDeclaration: vi.fn(),
    onExit: vi.fn(),
    onHoverAt: vi.fn(),
    onMouseDownAt: vi.fn(() => undefined),
    onMouseDragAt: vi.fn(),
    onMouseUpAt: vi.fn(),
    onMultiClick: vi.fn(),
    onOpenHyperlink: vi.fn(),
    onSelectionChange: vi.fn(),
    onSelectionDrag: vi.fn(),
    onTerminalFocusChange,
    selection: createSelectionState(),
    stderr: stdout,
    stdin,
    stdout,
    terminalColumns: 80,
    terminalRows: 24
  })
}

describe('App terminal focus events', () => {
  beforeEach(() => {
    resetTerminalFocusState()
  })

  it('notifies the renderer on DECSET 1004 focus transitions', () => {
    const onTerminalFocusChange = vi.fn()
    const app = makeApp(onTerminalFocusChange)

    app.processInput(FOCUS_OUT)
    expect(getTerminalFocusState()).toBe('blurred')
    expect(onTerminalFocusChange).toHaveBeenLastCalledWith(false)

    app.processInput(FOCUS_IN)
    expect(getTerminalFocusState()).toBe('focused')
    expect(onTerminalFocusChange).toHaveBeenLastCalledWith(true)
    expect(onTerminalFocusChange).toHaveBeenCalledTimes(2)
  })
})
