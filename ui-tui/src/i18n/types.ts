// Static, user-facing TUI strings. This is the first migrated slice — the
// catalog grows as more hard-coded strings move here. Every leaf is a plain
// string (no template functions yet); interpolation stays in the call site.

export interface TuiTranslations {
  branding: {
    gateway: {
      disabled: string
      connecting: string
      configured: string
      failed: string
    }
    noSystemPrompt: string
    sessionLabel: string
  }
  skills: {
    loading: string
    none: string
    selectCategory: string
    noneInCategory: string
    loadingShort: string
    installing: string
  }
  plugins: {
    loading: string
    none: string
    installHint: string
    updating: string
  }
  sessions: {
    loading: string
    noOther: string
  }
  pets: {
    loading: string
    adopting: string
  }
  models: {
    loading: string
    noProviders: string
  }
  agents: {
    compareHint: string
    noSubagents: string
  }
}

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K]
}

export const TUI_LOCALES = ['en', 'ja'] as const
export type TuiLocale = (typeof TUI_LOCALES)[number]
export const DEFAULT_TUI_LOCALE: TuiLocale = 'en'
