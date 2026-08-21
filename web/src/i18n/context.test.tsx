// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from './context'

// Node 26 exposes a global `localStorage` accessor that resolves to undefined
// unless --localstorage-file is configured. In Vitest's jsdom environment that
// accessor shadows jsdom storage on `window`, so install the same standards-
// shaped in-memory Storage fallback used by Desktop's shared test setup.
if (typeof window.localStorage === 'undefined') {
  const store = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return store.size
    },
    clear: () => store.clear(),
    getItem: key => store.get(String(key)) ?? null,
    key: index => [...store.keys()][index] ?? null,
    removeItem: key => void store.delete(String(key)),
    setItem: (key, value) => void store.set(String(key), String(value))
  }

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: storage,
    writable: true
  })
}

function LanguageProbe() {
  const { locale, setLocale, t } = useI18n()

  return (
    <div>
      <p data-testid="locale">{locale}</p>
      <p data-testid="save">{t.common.save}</p>
      <button onClick={() => setLocale('pl')} type="button">
        switch to Polish
      </button>
    </div>
  )
}

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
  window.localStorage.clear()
  document.documentElement.lang = ''
  document.documentElement.dir = ''
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  window.localStorage.clear()
  delete (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT
})

describe('Web I18nProvider browser seam', () => {
  it('hydrates Polish from hermes-locale and applies the document language', () => {
    window.localStorage.setItem('hermes-locale', 'pl')

    act(() => {
      root.render(
        <I18nProvider>
          <LanguageProbe />
        </I18nProvider>
      )
    })

    expect(container.querySelector('[data-testid="locale"]')?.textContent).toBe('pl')
    expect(container.querySelector('[data-testid="save"]')?.textContent).toBe('Zapisz')
    expect(document.documentElement.lang).toBe('pl')
    expect(document.documentElement.dir).toBe('ltr')
  })

  it('persists a language switch in hermes-locale and updates documentElement.lang', () => {
    act(() => {
      root.render(
        <I18nProvider>
          <LanguageProbe />
        </I18nProvider>
      )
    })

    expect(document.documentElement.lang).toBe('en')
    expect(window.localStorage.getItem('hermes-locale')).toBeNull()

    act(() => {
      container.querySelector('button')?.click()
    })

    expect(container.querySelector('[data-testid="locale"]')?.textContent).toBe('pl')
    expect(container.querySelector('[data-testid="save"]')?.textContent).toBe('Zapisz')
    expect(window.localStorage.getItem('hermes-locale')).toBe('pl')
    expect(document.documentElement.lang).toBe('pl')
    expect(document.documentElement.dir).toBe('ltr')
  })
})
