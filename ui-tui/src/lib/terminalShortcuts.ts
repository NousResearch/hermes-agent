import type { TerminalCapabilities } from './terminalCapabilities.js'

export type ShortcutKey = {
  ctrl: boolean
  shift?: boolean
  meta: boolean
  super?: boolean
  return?: boolean
  escape?: boolean
}

export type ShortcutState = {
  hasSelection: boolean
  busy: boolean
}

export type ClassifyInput = {
  input: string
  raw: string
  key: ShortcutKey
  caps: TerminalCapabilities
  state: ShortcutState
}

export type TerminalAction =
  | { type: 'paste'; source: 'hotkey' | 'bracketed' | 'fallback' }
  | { type: 'copy' }
  | { type: 'interrupt' }
  | { type: 'noop' }
  | { type: 'text'; text: string }

function isMacLikeTerminal(caps: TerminalCapabilities): boolean {
  return caps.keyboard.pasteShortcutShapes.includes('cmd+v')
}

function isPasteHotkey(i: ClassifyInput): boolean {
  const ch = i.input.toLowerCase()

  if (ch !== 'v') {
    return false
  }

  if (i.key.ctrl && i.key.shift && i.caps.keyboard.pasteShortcutShapes.includes('ctrl+shift+v')) {
    return true
  }

  if (!i.key.ctrl && (i.key.meta || i.key.super)) {
    if (isMacLikeTerminal(i.caps) && i.caps.keyboard.pasteShortcutShapes.includes('cmd+v')) {
      return true
    }

    if (!isMacLikeTerminal(i.caps) && i.caps.keyboard.pasteShortcutShapes.includes('alt+v')) {
      return true
    }
  }

  if (i.key.ctrl && !i.key.shift && !i.key.meta && !i.key.super && i.caps.keyboard.pasteShortcutShapes.includes('ctrl+v')) {
    return true
  }

  return false
}

function classifyCopyChord(i: ClassifyInput): TerminalAction | undefined {
  const ch = i.input.toLowerCase()

  if (ch !== 'c') {
    return undefined
  }

  if (i.key.meta || i.key.super) {
    return i.state.hasSelection ? { type: 'copy' } : { type: 'noop' }
  }

  if (!isMacLikeTerminal(i.caps) && i.key.ctrl && i.key.shift && !i.key.meta && !i.key.super) {
    return i.state.hasSelection ? { type: 'copy' } : { type: 'noop' }
  }

  if (i.key.ctrl && !i.key.shift && !i.key.meta && !i.key.super) {
    return { type: 'interrupt' }
  }

  return undefined
}

export function classifyKeyEvent(i: ClassifyInput): TerminalAction {
  if (i.raw.startsWith('\x1b[200~')) {
    return { type: 'paste', source: 'bracketed' }
  }

  if (isPasteHotkey(i)) {
    return { type: 'paste', source: 'hotkey' }
  }

  const copy = classifyCopyChord(i)

  if (copy) {
    return copy
  }

  if (i.raw === '\x03' && i.key.ctrl && !i.key.shift && !i.key.meta && !i.key.super) {
    return { type: 'interrupt' }
  }

  return i.input ? { type: 'text', text: i.input } : { type: 'noop' }
}

export const classifyTerminalInput = classifyKeyEvent
