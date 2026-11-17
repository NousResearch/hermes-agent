import { afterEach, expect, it } from 'vitest'

import { en } from '../i18n/en.js'
import { flattenKeys } from '../i18n/keys.js'
import { applyLocale, resetLocale, t } from '../i18n/runtime.js'

import { activateZh, zhMessages } from './localeFixture.js'
afterEach(resetLocale)
it('bundled Chinese strings preserve the presentation keys and positional arguments', () => {
  const keys = flattenKeys(en)
  // Complete served coverage, including reused core command copy, is checked by
  // test_i18n_tui_packs.py and the gateway's requested-language contract.
  expect(Object.keys(zhMessages).every(key => keys.includes(key))).toBe(true)

  for (const key of Object.keys(zhMessages)) {
    const leaf = key.split('.').reduce<any>((node, part) => node[part], en)
    const args = typeof leaf === 'function' ? Array.from({ length: leaf.length }, (_, i) => `{${i}}`) : []
    const template = typeof leaf === 'function' ? leaf(...args) : leaf
    const placeholders = (s: string) => [...new Set(s.match(/\{\d+\}/g) ?? [])].sort()
    expect(placeholders(zhMessages[key]), key).toEqual(placeholders(template))
  }
})
it('language changes update presentation while partial packs fall directly back to English', () => {
  activateZh()
  expect(t('status.ready')).not.toBe(en.status.ready)
  applyLocale('zh-hant', { lang: 'zh-hant', surface: 'tui', messages: { 'status.ready': '準備就緒' } })
  expect(t('status.ready')).toBe('準備就緒')
  expect(t('status.running')).toBe(en.status.running)
  applyLocale('new-pack', { lang: 'new-pack', surface: 'tui', messages: { 'status.ready': 'new-ready' } })
  expect(t('status.ready')).toBe('new-ready')
})
