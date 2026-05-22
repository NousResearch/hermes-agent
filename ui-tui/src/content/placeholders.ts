// Placeholders are now driven by i18n catalog keys (input.placeholder1–7).
// The runtime selection happens in appLayout.tsx via useMemo<TranslationKey>.
// This file is kept as a registry stub — translators can reference the
// original English strings here when authoring a new locale pack.
export const PLACEHOLDERS = [
  'Ask me anything…',
  'Try "explain this codebase"',
  'Try "write a test for…"',
  'Try "refactor the auth module"',
  'Try "/help" for commands',
  'Try "fix the lint errors"',
  'Try "how does the config loader work?"',
] as const
