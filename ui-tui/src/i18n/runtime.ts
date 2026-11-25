// Locale + merged catalog store for the TUI, mirroring apps/desktop/src/i18n/runtime.ts.
//
// Resolution for `t(key)`: active catalog (en merged with the RPC-delivered pack
// for the active language) → bundled en → the bare key. Packs are flat dotted
// maps of strings; a string overriding a function-valued en leaf is wrapped into
// `(...args) => format(str, args)` with positional `{0}`, `{1}` placeholders.

import { atom } from 'nanostores'

import { en } from './en.js'
import { isRecord, mergeTranslations, type TranslationOverride } from './merge.js'
import type { CatalogPack, TranslationKey, Translations } from './types.js'

export const DEFAULT_LOCALE = 'en'

export const $locale = atom<string>(DEFAULT_LOCALE)
export const $catalog = atom<Translations>(en)

/** Fill `{0}`, `{1}`… from `args`; unknown indices are left as-is so a pack
 *  with a stray placeholder degrades to visible text, not `undefined`. */
export function formatPositional(template: string, args: readonly unknown[]): string {
  return template.replace(/\{(\d+)\}/g, (match, index: string) => {
    const value = args[Number(index)]

    return value === undefined ? match : String(value)
  })
}

/** Walk a dot-path (`a.b.c`) into a nested message tree. */
function resolvePath(source: unknown, key: string): unknown {
  return key.split('.').reduce<unknown>((current, part) => (isRecord(current) ? current[part] : undefined), source)
}

/** A string is returned as-is, a function is called with `args`, else `null`. */
function render(value: unknown, args: readonly unknown[]): null | string {
  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'function') {
    return (value as (...args: unknown[]) => string)(...args)
  }

  return null
}

/** The active → en → key resolution every translator shares. */
export function translateFrom(
  source: (locale: string) => unknown,
  locale: string,
  key: string,
  args: readonly unknown[]
): string {
  const active = render(resolvePath(source(locale), key), args)

  if (active !== null) {
    return active
  }

  if (locale !== DEFAULT_LOCALE) {
    const fallback = render(resolvePath(source(DEFAULT_LOCALE), key), args)

    if (fallback !== null) {
      return fallback
    }
  }

  return key
}

/** Nest a flat `{ 'a.b': 'x' }` pack into `{ a: { b: 'x' } }`, wrapping strings
 *  that override function-valued base leaves. Keys absent from the base are
 *  dropped (the validator already warned about them on install). */
export function nestPack(base: unknown, messages: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {}

  for (const [flatKey, text] of Object.entries(messages)) {
    if (typeof text !== 'string') {
      continue
    }

    const baseLeaf = resolvePath(base, flatKey)

    if (baseLeaf === undefined || isRecord(baseLeaf)) {
      continue
    }

    const parts = flatKey.split('.')
    const leafKey = parts.pop() as string
    let cursor = out

    for (const part of parts) {
      const next = cursor[part]

      if (isRecord(next)) {
        cursor = next
      } else {
        const created: Record<string, unknown> = {}
        cursor[part] = created
        cursor = created
      }
    }

    cursor[leafKey] = typeof baseLeaf === 'function' ? (...args: unknown[]) => formatPositional(text, args) : text
  }

  return out
}

/** Merge a pack (flat dotted map) over the bundled English catalog. */
export function catalogFromPack(messages: Record<string, string> | undefined): Translations {
  if (!messages || Object.keys(messages).length === 0) {
    return en
  }

  return mergeTranslations<Translations>(en, nestPack(en, messages) as TranslationOverride<Translations>)
}

/** Install a language: `pack` may be null when the backend has no TUI strings
 *  for `lang` (or predates `i18n.catalog`) — the UI then renders English under
 *  that locale id. */
export function applyLocale(lang: string, pack: CatalogPack | null | undefined): void {
  $locale.set(lang || DEFAULT_LOCALE)
  $catalog.set(catalogFromPack(pack?.messages))
}

export function resetLocale(): void {
  applyLocale(DEFAULT_LOCALE, null)
}

export const getLocale = () => $locale.get()

/** The active merged catalog for non-React code that wants the typed tree. */
export const messages = (): Translations => $catalog.get()

/** Resolve one key for non-React code. Function-valued leaves take `args`. */
export function t(key: TranslationKey, ...args: unknown[]): string {
  return translateFrom(locale => (locale === DEFAULT_LOCALE ? en : $catalog.get()), $locale.get(), key, args)
}
