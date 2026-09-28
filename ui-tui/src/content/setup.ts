import { type Locale, translate } from '../i18n/index.js'
import type { PanelSection } from '../types.js'

export const setupRequiredTitle = (locale: Locale = 'en') => translate(locale, 'setup.title')
export const SETUP_REQUIRED_TITLE = setupRequiredTitle()

export const buildSetupRequiredSections = (locale: Locale = 'en'): PanelSection[] => [
  {
    text: translate(locale, 'setup.body')
  },
  {
    rows: [
      ['/setup', translate(locale, 'setup.action.runWizard')],
      ['/model', translate(locale, 'setup.action.configureModel')],
      ['Ctrl+C', translate(locale, 'setup.action.exitSetup')]
    ],
    title: translate(locale, 'setup.actions')
  },
  {
    text: translate(locale, 'setup.dashboardModels')
  }
]
