import type { TranslationOverride } from '@hermes/shared/i18n'
export type { Locale } from '@hermes/shared/locale-registry'
export type { SchemaTranslations } from './types/schema'
import type { ShellTranslations } from './types/shell'
import type { WorkspaceTranslations } from './types/workspace'
import type { SettingsTranslations } from './types/settings'
import type { ExtensionsTranslations } from './types/extensions'

/** Presentation contracts grouped by Dashboard surface. */
export interface Translations
  extends ShellTranslations, WorkspaceTranslations, SettingsTranslations, ExtensionsTranslations {}

/** Locale packs independently overlay the complete English catalog. */
export type TranslationOverlay = TranslationOverride<Translations>
