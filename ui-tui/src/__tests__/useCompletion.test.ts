import { afterEach } from 'vitest'

import { resetLocale } from '../i18n/runtime.js'

import { activateZh } from './localeFixture.js'
afterEach(resetLocale)
import { describe, expect, it } from 'vitest'

import { completionRequestForInput, localizeCompletionItems } from '../hooks/useCompletion.js'

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
      replaceFrom: 1
    })
  })

  it('does not route absolute paths through slash completion', () => {
    expect(
      completionRequestForInput('/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md')
    ).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md' },
      replaceFrom: 0
    })
  })

  it('keeps path completion for trailing absolute path tokens', () => {
    expect(completionRequestForInput('read /home/d/Desktop/file.md')).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/file.md' },
      replaceFrom: 5
    })
  })

  it('leaves plain text alone', () => {
    expect(completionRequestForInput('hello there')).toBeNull()
  })
})

describe('localized completion metadata', () => {
  it('uses stable presentation keys while preserving English wire fallbacks', () => {
    const item = {
      text: '@file:',
      display: '@file:',
      meta: 'attach file',
      meta_key: 'completion.attachFile'
    }

    activateZh()
    expect(localizeCompletionItems([item])[0]).toMatchObject({ text: '@file:', display: '@file:', meta: '附加文件' })
    resetLocale()
    expect(localizeCompletionItems([item])[0]).toMatchObject({ text: '@file:', display: '@file:', meta: 'attach file' })
  })

  it('interpolates dynamic argument metadata', () => {
    const item = {
      text: 'expanded',
      display: 'expanded',
      meta: 'set thinking',
      meta_key: 'completion.setSection',
      meta_vars: { section: 'thinking' }
    }

    activateZh()
    expect(localizeCompletionItems([item])[0]?.meta).toBe('设置 thinking')
  })
})

it('keeps the gateway description for a newer completion key the client does not know', () => {
  const item = {
    text: 'new-option',
    meta: 'New gateway option',
    meta_key: 'completion.futureOption'
  }

  expect(localizeCompletionItems([item])[0].meta).toBe('New gateway option')
})
