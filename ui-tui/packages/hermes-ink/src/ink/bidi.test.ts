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

  it('uses an RTL base for English-first Persian-majority technical prose', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('React یک کتابخانه جاوااسکریپت بسیار محبوب است.')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('.تسا بوبحم رایسب تپیرکسااواج هناخباتک کی React')
  })

  it('keeps an LTR base when English remains the strong-character majority', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('React is a popular library: محبوب')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('React is a popular library: بوبحم')
  })

  it('uses an LTR base for RTL-first, LTR-majority technical prose', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('محبوب React is a popular library.')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('بوبحم React is a popular library.')
  })

  it('falls back to first-strong behavior for an LTR-first tie', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('Aب.')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('Aب.')
  })

  it('falls back to first-strong behavior for an RTL-first tie', async () => {
    const { reorderBidi } = await importBidiWithSoftwareReordering()
    const input = charactersFrom('بA.')
    const output = reorderBidi(input)

    expect(textFrom(output)).toBe('.Aب')
  })
})
