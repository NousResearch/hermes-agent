import { useStore } from '@nanostores/react'

import { $catalog, $locale } from './runtime.js'
import type { Translations } from './types.js'

/** The active merged catalog, re-rendering the caller when the language changes. */
export function useT(): Translations {
  return useStore($catalog)
}

export function useLocale(): string {
  return useStore($locale)
}
