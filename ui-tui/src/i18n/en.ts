import type { TuiTranslations } from './types.js'

// English is the source of truth: the full key set lives here, and every other
// locale is a (possibly partial) override merged on top of it.
export const en: TuiTranslations = {
  branding: {
    gateway: {
      disabled: 'disabled',
      connecting: 'connecting',
      configured: 'configured',
      failed: 'failed'
    },
    noSystemPrompt: 'No system prompt loaded.',
    sessionLabel: 'Session: '
  }
}
