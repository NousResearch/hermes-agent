import { describe, expect, it } from 'vitest'

import {
  completionRequestForInput,
  localizableCompletionItem,
  localizeCompletionItems
} from '../hooks/useCompletion.js'

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
    const item = localizableCompletionItem({
      text: '@file:',
      display: '@file:',
      meta: 'attach file',
      meta_key: 'completion.attachFile'
    })

    expect(localizeCompletionItems([item], 'zh')).toEqual([{ text: '@file:', display: '@file:', meta: '附加文件' }])
    expect(localizeCompletionItems([item], 'ja')).toEqual([{ text: '@file:', display: '@file:', meta: 'attach file' }])
  })

  it('interpolates dynamic argument metadata', () => {
    const item = localizableCompletionItem({
      text: 'expanded',
      display: 'expanded',
      meta: 'set thinking',
      meta_key: 'completion.setSection',
      meta_vars: { section: 'thinking' }
    })

    expect(localizeCompletionItems([item], 'zh')[0]?.meta).toBe('设置 thinking')
  })
})
