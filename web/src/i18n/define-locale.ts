import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { en } from './en'
import type { Translations } from './types'

// Partial-locale helper: a translation file supplies only the strings it has
// translated and every missing key falls back to English, while unknown keys
// still fail the type-check.
export type TranslationOverrides = TranslationOverride<Translations>

export const defineLocale = (overrides: TranslationOverrides): Translations =>
  mergeTranslations<Translations>(en, overrides)
