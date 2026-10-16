import { STARTUP_TEMPORARY } from '../../config/env.js'
import { t } from '../../i18n/runtime.js'

// In a `hermes --temporary` TUI, /sessions (/resume, /switch) would swap a saved chat in under the
// "temporary chat" status mark, and /branch would copy the temporary transcript into a saved chat
// (the gateway refuses that too). Same contract as the classic CLI.
const BLOCKED = new Set(['branch', 'sessions'])

/** The refusal to print for `name` in temporary mode, else null. */
export function temporaryRefusal(name: string, temporary = STARTUP_TEMPORARY): null | string {
  return temporary && BLOCKED.has(name) ? t('session.lifecycle.temporaryBlocked', name) : null
}
