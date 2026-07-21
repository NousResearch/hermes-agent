import type { WidgetApp } from './types.js'

const apps = new Map<string, WidgetApp<never>>()

/** Identity helper that pins the state type, then registers. Last writer
 *  wins so a user/plugin app can shadow a built-in of the same id. */
export function defineWidgetApp<S>(app: WidgetApp<S>): WidgetApp<S> {
  apps.set(app.id, app as WidgetApp<never>)

  return app
}

export const getWidgetApp = (id: string): undefined | WidgetApp<never> => apps.get(id)

export const listWidgetApps = (): string[] => [...apps.keys()].sort()
