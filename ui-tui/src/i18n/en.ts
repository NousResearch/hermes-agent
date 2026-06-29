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
  },
  skills: {
    loading: 'loading skills…',
    none: 'no skills available',
    selectCategory: 'select a category',
    noneInCategory: 'no skills in this category',
    loadingShort: 'loading…',
    installing: 'installing…'
  },
  plugins: {
    loading: 'loading plugins…',
    none: 'no plugins installed',
    installHint: 'install: hermes plugins install owner/repo',
    updating: 'updating…'
  },
  sessions: {
    loading: 'loading sessions…',
    noOther: 'no other sessions — Enter on +new to start one'
  },
  pets: {
    loading: 'loading pets…',
    adopting: 'adopting…'
  },
  models: {
    loading: 'loading models…',
    noProviders: 'no providers available'
  },
  agents: {
    compareHint: 'baseline vs candidate · esc/q close',
    noSubagents: 'No subagents this turn. Trigger delegate_task to populate the tree.'
  }
}
