import { getTranslations } from '../i18n/index.js'
import type { PanelSection } from '../types.js'

// Retained for existing consumers that need the English constant.
export const SETUP_REQUIRED_TITLE = 'Setup Required'
export const setupRequiredTitle = () => getTranslations().setup.title

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: getTranslations().setup.description
  },
  {
    rows: [
      ['/model', getTranslations().setup.model],
      ['/setup', getTranslations().setup.wizard],
      ['Ctrl+C', getTranslations().setup.exit]
    ],
    title: getTranslations().setup.actions
  }
]
