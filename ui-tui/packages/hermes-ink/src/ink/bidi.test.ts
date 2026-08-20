import { afterEach, describe, expect, it, vi } from 'vitest'

import type { reorderBidi } from './bidi.js'

type BidiCharacter = Parameters<typeof reorderBidi>[0][number]

const charactersFrom = (text: string): BidiCharacter[] =>
  Array.from(text, value => ({
    value,
    width: 1,
    styleId: 0,
    hyperlink: undefined
  }))

const textFrom = (characters: BidiCharacter[]) => characters.map(character => character.value).join('')

const importBidiWithSoftwareReordering = async () => {
  vi.resetModules()
  vi.stubEnv('TERM_PROGRAM', 'vscode')

  return import('./bidi.js')
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

describe('reorderBidi', () => {
  it('leaves pure LTR text unchanged', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('hello /help gpt-5')
    const output = reorderBidi(input)

    expect(output).toBe(input)
    expect(textFrom(output)).toBe('hello /help gpt-5')
  })

  it('detects Arabic text through the RTL reorder path', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('مرحبا')
    const output = reorderBidi(input)

    expect(output).not.toBe(input)
    expect(textFrom(output)).toBe('ابحرم')
  })

  it('keeps an English technical token readable in mixed Arabic text', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('مرحبا gpt-5')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('gpt-5 ابحرم')
  })
})
