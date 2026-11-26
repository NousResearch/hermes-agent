// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { I18nContext, formatTranslation, resolveTranslations } from '@/i18n/runtime'
import type { Locale } from '@/i18n/types'

import { AuthWidget } from './AuthWidget'

let root: Root
let container: HTMLDivElement
const reload = vi.fn()

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  vi.stubGlobal('window', {
    __HERMES_SESSION_TOKEN__: 'local-token',
    __HERMES_BASE_PATH__: '',
    location: { pathname: '/', search: '', reload }
  })
  container = document.createElement('div')
  root = createRoot(container)
})

afterEach(async () => {
  await act(async () => root.unmount())
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  reload.mockClear()
})

async function render(locale: Locale) {
  await act(async () => {
    root.render(
      <I18nContext.Provider value={{
        format: formatTranslation,
        locale,
        setLocale: async () => {},
        t: resolveTranslations(locale)
      }}>
        <AuthWidget />
      </I18nContext.Provider>
    )
  })
}

it.each([401, 403])('hides an HTTP %i without depending on translated error text', async status => {
  const fetchMock = vi.fn(async () => new Response('', { status }))
  vi.stubGlobal('fetch', fetchMock)

  await render('zh')

  expect(fetchMock).toHaveBeenCalledTimes(1)
  expect(container.innerHTML).toBe('')
  expect(reload).not.toHaveBeenCalled()
})

it.each(['en', 'zh'] as const)('offers localized recovery in %s when identity is unavailable', async locale => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))

  await render(locale)

  const copy = resolveTranslations(locale)
  expect(container.querySelector('[role="status"]')?.textContent).toContain(copy.auth.statusUnavailable)
  const button = container.querySelector('button')
  expect(button?.textContent).toBe(copy.chatSidebar.reloadPage)
  await act(async () => button?.click())
  expect(reload).toHaveBeenCalledTimes(1)
})
