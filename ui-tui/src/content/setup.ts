import type { PanelSection } from '../types.js'

export const SETUP_REQUIRED_TITLE = 'Setup Required'

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: 'Hermes needs a model provider before the TUI can start a session.'
  },
  {
    rows: [
      ['/setup', 'run the first-time setup wizard in-place (adds a provider)'],
      ['/model', 'pick a model (needs a session — add a provider first)'],
      ['Ctrl+C', 'exit and run `hermes setup` manually']
    ],
    title: 'Actions'
  },
  {
    text: 'In the dashboard the Models page sets the profile default; on Desktop it is Settings -> Models.'
  }
]
