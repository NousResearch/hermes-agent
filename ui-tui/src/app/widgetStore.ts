import { atom } from 'nanostores'

import { WIDGET_CATALOG } from '../widgets.js'

export interface WidgetState {
  enabled: Record<string, boolean>
  params: Record<string, Record<string, string>>
}

function defaults(): WidgetState {
  const enabled: Record<string, boolean> = {}

  for (const w of WIDGET_CATALOG) {
    enabled[w.id] = w.defaultOn
  }

  return { enabled, params: {} }
}

export const $widgetState = atom<WidgetState>(defaults())

export function toggleWidget(id: string, force?: boolean) {
  const s = $widgetState.get()
  const next = force ?? !s.enabled[id]

  $widgetState.set({ ...s, enabled: { ...s.enabled, [id]: next } })

  return next
}

export function setWidgetParam(id: string, key: string, value: string) {
  const s = $widgetState.get()
  const prev = s.params[id] ?? {}

  $widgetState.set({ ...s, params: { ...s.params, [id]: { ...prev, [key]: value } } })
}

export function getWidgetEnabled(id: string): boolean {
  return $widgetState.get().enabled[id] ?? false
}
