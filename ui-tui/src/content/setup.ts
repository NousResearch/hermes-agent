import type { PanelSection } from '../types.js'

export const SETUP_REQUIRED_TITLE = 'Setup Required'

export const FREE_TIER_LIMIT_KEY = 'free_tier.limit'
export const FREE_TIER_LIMIT_ERROR = 'free_tier_limit'
export const FREE_TIER_LIMIT_TITLE = 'Choose how to continue'

export const buildFreeTierLimitSections = (message: string): PanelSection[] => [
  { text: message },
  {
    title: 'Continue with',
    rows: [
      ['/login', 'sign in or create a Nous account'],
      ['/model', 'use a local model or another provider']
    ]
  },
  { text: 'Your conversation is saved. Choosing a working provider lets you continue here.' }
]

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: 'Hermes needs a model provider before the TUI can start a session.'
  },
  {
    rows: [
      ['/model', 'configure provider + model in-place'],
      ['/setup', 'run full first-time setup wizard in-place'],
      ['Ctrl+C', 'exit and run `hermes setup` manually']
    ],
    title: 'Actions'
  }
]
