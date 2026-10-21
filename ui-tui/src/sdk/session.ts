import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'

import { $uiState } from '../app/uiStore.js'

/** The session a widget is docked in, as widgets see it. */
export interface WidgetSession {
  /** Folder the session works in (the session's own cwd, else the launch folder); null before either is known. */
  cwd: null | string
  /** Durable session id (the state.db row `hermes --resume` takes); null until the session exists. It changes
   *  on /new, /resume and compression rotation, so key per-session data on it rather than caching it. */
  id: null | string
}

// Primitive-valued stores: an atom only notifies on a changed value, so widgets re-render on a session
// switch, not on every streamed token that rewrites $uiState.
const $widgetSessionId = computed($uiState, state => state.storedSid || null)
const $widgetSessionCwd = computed($uiState, state => state.info?.cwd || process.env.HERMES_CWD || null)

/** Current session, outside React (timers, async bridges). */
export const getWidgetSession = (): WidgetSession => ({ cwd: $widgetSessionCwd.get(), id: $widgetSessionId.get() })

/** Current session as a hook: the calling component re-renders when it changes. */
export function useWidgetSession(): WidgetSession {
  const id = useStore($widgetSessionId)
  const cwd = useStore($widgetSessionCwd)

  return { cwd, id }
}
