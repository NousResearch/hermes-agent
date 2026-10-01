import { beforeEach, describe, expect, it, vi } from 'vitest'

import { parseOscColor } from './terminal.js'

// setTerminalBackgroundHex is first-writer-wins module state — re-import a
// fresh module instance per test so cases don't contaminate each other.
async function freshTerminal() {
  vi.resetModules()

  return import('./terminal.js')
}

beforeEach(() => {
  vi.resetModules()
})

describe('parseOscColor', () => {
  it('parses the canonical 16-bit X11 rgb: form', () => {
    expect(parseOscColor('rgb:1e1e/2a2a/3f3f')).toBe('#1e2a3f')
    expect(parseOscColor('rgb:ffff/ffff/ffff')).toBe('#ffffff')
    expect(parseOscColor('rgb:0000/0000/0000')).toBe('#000000')
  })

  it('scales shorter per-channel widths to 8 bits', () => {
    expect(parseOscColor('rgb:f/f/f')).toBe('#ffffff')
    expect(parseOscColor('rgb:ff/ff/ff')).toBe('#ffffff')
    expect(parseOscColor('rgb:fff/fff/fff')).toBe('#ffffff')
    expect(parseOscColor('rgb:8/0/0')).toBe('#880000')
  })

  it('accepts rgba: (alpha ignored) and plain hex forms', () => {
    expect(parseOscColor('rgba:1e1e/2a2a/3f3f/ffff')).toBe('#1e2a3f')
    expect(parseOscColor('#1e2a3f')).toBe('#1e2a3f')
    expect(parseOscColor('1e2a3f')).toBe('#1e2a3f')
    expect(parseOscColor('#abc')).toBe('#aabbcc')
    expect(parseOscColor('#1e1e2a2a3f3f')).toBe('#1e2a3f')
  })

  it('rejects garbage', () => {
    expect(parseOscColor('')).toBeUndefined()
    expect(parseOscColor('rgb:zz/zz/zz')).toBeUndefined()
    expect(parseOscColor('rgb:1e1e/2a2a')).toBeUndefined()
    expect(parseOscColor('notacolor')).toBeUndefined()
  })
})

describe('terminal background storage', () => {
  it('fires queued listeners once when the background arrives', async () => {
    const t = await freshTerminal()
    const seen: string[] = []

    t.onTerminalBackground(hex => seen.push(hex))
    expect(t.terminalBackgroundHex()).toBeUndefined()

    t.setTerminalBackgroundHex('#ffffff')
    expect(seen).toEqual(['#ffffff'])
    expect(t.terminalBackgroundHex()).toBe('#ffffff')
  })

  it('fires immediately for listeners registered after the answer', async () => {
    const t = await freshTerminal()

    t.setTerminalBackgroundHex('#1e1e2e')

    const seen: string[] = []

    t.onTerminalBackground(hex => seen.push(hex))
    expect(seen).toEqual(['#1e1e2e'])
  })

  it('is first-writer-wins (re-probe defense)', async () => {
    const t = await freshTerminal()

    t.setTerminalBackgroundHex('#ffffff')
    t.setTerminalBackgroundHex('#000000')
    expect(t.terminalBackgroundHex()).toBe('#ffffff')
  })

  it('foreground (OSC 10) is an independent slot with the same semantics', async () => {
    const t = await freshTerminal()
    const seen: string[] = []

    t.onTerminalForeground(hex => seen.push(hex))
    t.setTerminalForegroundHex('#cccccc')
    t.setTerminalForegroundHex('#000000')

    expect(seen).toEqual(['#cccccc'])
    expect(t.terminalForegroundHex()).toBe('#cccccc')
    // The background slot is untouched by foreground writes.
    expect(t.terminalBackgroundHex()).toBeUndefined()
  })
})
