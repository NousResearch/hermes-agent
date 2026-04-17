import type { PanelSection } from '../types.js'

export const SETUP_REQUIRED_TITLE = 'Setup Required'

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: 'Hermes needs a model provider before the TUI can start a session.'
  },
  {
    rows: [
      ['1.', 'Exit with Ctrl+C'],
      ['2.', 'Run `hermes model` to choose a provider + model'],
      ['3.', 'Or run `hermes setup` for full first-time setup'],
      ['4.', 'Re-open `hermes --tui` when setup is done']
    ],
    title: 'Next Steps'
  }
]
