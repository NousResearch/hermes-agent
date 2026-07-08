import { describe, expect, it } from 'vitest'

import {
  initialViState,
  processViKey,
  viModeIndicator,
  type ViKeyEvent,
  type ViState
} from '../lib/viMode.js'

const mkEvent = (input: string, key: Partial<ViKeyEvent['key']> = {}): ViKeyEvent => ({
  input,
  key: {
    ctrl: false,
    shift: false,
    meta: false,
    escape: false,
    backspace: false,
    delete: false,
    return: false,
    upArrow: false,
    downArrow: false,
    leftArrow: false,
    rightArrow: false,
    ...key
  }
})

describe('viMode', () => {
  describe('initialViState', () => {
    it('starts in insert mode', () => {
      const state = initialViState()
      expect(state.mode).toBe('insert')
      expect(state.operator).toBeNull()
      expect(state.count).toBe(0)
    })
  })

  describe('viModeIndicator', () => {
    it('returns correct labels for each mode', () => {
      expect(viModeIndicator('normal')).toBe('NORMAL')
      expect(viModeIndicator('insert')).toBe('INSERT')
      expect(viModeIndicator('visual')).toBe('VISUAL')
      expect(viModeIndicator('operator-pending')).toBe('OPERATOR')
    })
  })

  describe('processViKey - insert mode', () => {
    it('passes through regular keys in insert mode', () => {
      const state = initialViState()
      const { action, newState } = processViKey(state, 'hello', 5, mkEvent('x'))
      expect(action.type).toBe('passthrough')
      expect(newState.mode).toBe('insert')
    })

    it('exits insert mode on Escape', () => {
      const state = initialViState()
      const { action, newState } = processViKey(state, 'hello', 5, mkEvent('', { escape: true }))
      expect(newState.mode).toBe('normal')
      expect(action.type).toBe('cursor')
      // Cursor moves back one on exit from insert mode
      expect(action.cursor).toBe(4)
    })

    it('exits insert mode on Ctrl+C', () => {
      const state = initialViState()
      const { action, newState } = processViKey(state, 'hello', 3, mkEvent('c', { ctrl: true }))
      expect(newState.mode).toBe('normal')
      expect(action.cursor).toBe(2)
    })
  })

  describe('processViKey - normal mode navigation', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('h moves cursor left', () => {
      const { action } = processViKey(normalState, 'hello', 3, mkEvent('h'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(2)
    })

    it('l moves cursor right', () => {
      const { action } = processViKey(normalState, 'hello', 2, mkEvent('l'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(3)
    })

    it('0 moves to line start', () => {
      const { action } = processViKey(normalState, 'hello', 3, mkEvent('0'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(0)
    })

    it('$ moves to line end', () => {
      const { action } = processViKey(normalState, 'hello', 1, mkEvent('$'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(4) // End is length - 1 in normal mode
    })

    it('w moves to next word start', () => {
      const { action } = processViKey(normalState, 'hello world', 0, mkEvent('w'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(6)
    })

    it('b moves to previous word start', () => {
      const { action } = processViKey(normalState, 'hello world', 8, mkEvent('b'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(6)
    })

    it('numeric prefix multiplies movement', () => {
      const stateWith3 = { ...normalState, count: 3 }
      const { action } = processViKey(stateWith3, 'hello world test', 0, mkEvent('l'))
      expect(action.cursor).toBe(3)
    })
  })

  describe('processViKey - insert mode entry', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('i enters insert mode at cursor', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('i'))
      expect(newState.mode).toBe('insert')
      expect(action.type).toBe('insert')
      expect(action.cursor).toBe(2)
    })

    it('a enters insert mode after cursor', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('a'))
      expect(newState.mode).toBe('insert')
      expect(action.type).toBe('insert')
      expect(action.cursor).toBe(3)
    })

    it('A enters insert mode at end of line', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('A'))
      expect(newState.mode).toBe('insert')
      expect(action.cursor).toBe(5)
    })

    it('I enters insert mode at first non-whitespace', () => {
      const { action, newState } = processViKey(normalState, '  hello', 4, mkEvent('I'))
      expect(newState.mode).toBe('insert')
      expect(action.cursor).toBe(2)
    })
  })

  describe('processViKey - delete operations', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('x deletes character under cursor', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('x'))
      expect(action.type).toBe('delete')
      expect(action.deleteRange).toEqual({ start: 2, end: 3 })
      expect(newState.register).toBe('l')
    })

    it('X deletes character before cursor', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('X'))
      expect(action.type).toBe('delete')
      expect(action.deleteRange).toEqual({ start: 1, end: 2 })
      expect(newState.register).toBe('e')
    })

    it('dd deletes entire line', () => {
      const stateWithD = { ...normalState, operator: 'd' }
      const { action, newState } = processViKey(stateWithD, 'hello', 2, mkEvent('d'))
      expect(action.type).toBe('delete')
      expect(action.deleteRange).toEqual({ start: 0, end: 5 })
      expect(newState.register).toBe('hello')
    })

    it('dw deletes word', () => {
      const stateWithD: ViState = { ...normalState, operator: 'd', mode: 'operator-pending' }
      const { action, newState } = processViKey(stateWithD, 'hello world', 0, mkEvent('w'))
      expect(action.type).toBe('delete')
      expect(action.deleteRange).toEqual({ start: 0, end: 6 })
      expect(newState.register).toBe('hello ')
    })
  })

  describe('processViKey - change operations', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('s substitutes character and enters insert mode', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('s'))
      expect(action.type).toBe('change')
      expect(newState.mode).toBe('insert')
      expect(action.deleteRange).toEqual({ start: 2, end: 3 })
    })

    it('C changes to end of line', () => {
      const { action, newState } = processViKey(normalState, 'hello', 2, mkEvent('C'))
      expect(action.type).toBe('change')
      expect(newState.mode).toBe('insert')
      expect(action.deleteRange).toEqual({ start: 2, end: 5 })
    })

    it('cw changes word and enters insert mode', () => {
      const stateWithC: ViState = { ...normalState, operator: 'c', mode: 'operator-pending' }
      const { action, newState } = processViKey(stateWithC, 'hello world', 0, mkEvent('w'))
      expect(action.type).toBe('change')
      expect(newState.mode).toBe('insert')
    })
  })

  describe('processViKey - yank and paste', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('yy yanks entire line', () => {
      const stateWithY = { ...normalState, operator: 'y' }
      const { action, newState } = processViKey(stateWithY, 'hello', 2, mkEvent('y'))
      expect(action.type).toBe('yank')
      expect(newState.register).toBe('hello')
    })

    it('p pastes after cursor', () => {
      const stateWithRegister = { ...normalState, register: 'world' }
      const { action } = processViKey(stateWithRegister, 'hello', 2, mkEvent('p'))
      expect(action.type).toBe('paste')
      expect(action.cursor).toBe(3)
      expect(action.text).toBe('world')
    })

    it('P pastes before cursor', () => {
      const stateWithRegister = { ...normalState, register: 'world' }
      const { action } = processViKey(stateWithRegister, 'hello', 2, mkEvent('P'))
      expect(action.type).toBe('paste')
      expect(action.cursor).toBe(2)
      expect(action.text).toBe('world')
    })
  })

  describe('processViKey - undo/redo', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('u triggers undo', () => {
      const { action } = processViKey(normalState, 'hello', 2, mkEvent('u'))
      expect(action.type).toBe('undo')
    })

    it('Ctrl+R triggers redo', () => {
      const { action } = processViKey(normalState, 'hello', 2, mkEvent('\x12', { ctrl: true }))
      expect(action.type).toBe('redo')
    })
  })

  describe('processViKey - find character (f/F/t/T) and replace (r)', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    const pending = (op: string): ViState => ({ ...normalState, operator: op, mode: 'operator-pending' })

    it('f<char> jumps onto the character', () => {
      const { action, newState } = processViKey(pending('f'), 'hello world', 0, mkEvent('o'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(4)
      expect(newState.mode).toBe('normal')
      expect(newState.lastFindChar).toEqual({ char: 'o', forward: true, till: false })
    })

    it('t<char> stops before the character', () => {
      const { action } = processViKey(pending('t'), 'hello world', 0, mkEvent('w'))
      expect(action.cursor).toBe(5)
    })

    it('F<char> searches backward', () => {
      const { action } = processViKey(pending('F'), 'hello world', 10, mkEvent('o'))
      expect(action.cursor).toBe(7)
    })

    it('f<char> with no match is a no-op that clears the pending state', () => {
      const { action, newState } = processViKey(pending('f'), 'hello', 0, mkEvent('z'))
      expect(action.type).toBe('none')
      expect(newState.operator).toBeNull()
      expect(newState.mode).toBe('normal')
    })

    it('fh consumes h as the target char, not as a motion', () => {
      const { action } = processViKey(pending('f'), 'ahb', 0, mkEvent('h'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(1)
    })

    it('f3 consumes 3 as the target char, not as a count', () => {
      const { action, newState } = processViKey(pending('f'), 'a3b', 0, mkEvent('3'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(1)
      expect(newState.count).toBe(0)
    })

    it('r<char> replaces the character under the cursor', () => {
      const { action, newState } = processViKey(pending('r'), 'hello', 1, mkEvent('a'))
      expect(action.type).toBe('replace')
      expect(action.deleteRange).toEqual({ start: 1, end: 2 })
      expect(action.text).toBe('a')
      expect(newState.mode).toBe('normal')
    })

    it('r at end of buffer is a no-op', () => {
      const { action } = processViKey(pending('r'), 'hello', 5, mkEvent('a'))
      expect(action.type).toBe('none')
    })

    it('; repeats the last find', () => {
      const withFind: ViState = { ...normalState, lastFindChar: { char: 'o', forward: true, till: false } }
      const { action } = processViKey(withFind, 'hello world', 4, mkEvent(';'))
      expect(action.type).toBe('cursor')
      expect(action.cursor).toBe(7)
    })
  })

  describe('processViKey - escape resets pending state', () => {
    it('escape cancels a pending operator', () => {
      const pendingD: ViState = {
        mode: 'operator-pending',
        operator: 'd',
        count: 0,
        visualAnchor: null,
        lastFindChar: null,
        register: ''
      }
      const { newState } = processViKey(pendingD, 'hello', 2, mkEvent('', { escape: true }))
      expect(newState.mode).toBe('normal')
      expect(newState.operator).toBeNull()
    })

    it('escape clears a pending count', () => {
      const withCount: ViState = {
        mode: 'normal',
        operator: null,
        count: 42,
        visualAnchor: null,
        lastFindChar: null,
        register: ''
      }
      const { newState } = processViKey(withCount, 'hello', 2, mkEvent('', { escape: true }))
      expect(newState.count).toBe(0)
    })

    it('escape in normal mode does not move the cursor', () => {
      const normal: ViState = {
        mode: 'normal',
        operator: null,
        count: 0,
        visualAnchor: null,
        lastFindChar: null,
        register: ''
      }
      const { action } = processViKey(normal, 'hello', 3, mkEvent('', { escape: true }))
      expect(action.cursor).toBe(3)
    })
  })

  describe('processViKey - submit', () => {
    const normalState: ViState = {
      mode: 'normal',
      operator: null,
      count: 0,
      visualAnchor: null,
      lastFindChar: null,
      register: ''
    }

    it('Enter in normal mode submits', () => {
      const { action } = processViKey(normalState, 'hello', 2, mkEvent('\r', { return: true }))
      expect(action.type).toBe('submit')
    })
  })
})
