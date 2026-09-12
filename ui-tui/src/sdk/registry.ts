import { type Locale, translate } from '../i18n/index.js'

import type { WidgetApp } from './types.js'

const apps = new Map<string, WidgetApp<never>>()

/** Identity helper that pins the state type, then registers. Last writer
 *  wins so a user/plugin app can shadow a built-in of the same id. */
export function defineWidgetApp<S>(app: WidgetApp<S>): WidgetApp<S> {
  apps.set(app.id, app as WidgetApp<never>)

  return app
}

export const getWidgetApp = (id: string): undefined | WidgetApp<never> => apps.get(id)

/** Unregister (user-widget file deleted). Built-ins never call this. */
export const removeWidgetApp = (id: string): boolean => apps.delete(id)

/** All registered apps, id-sorted — the registry IS the catalog: slash
 *  commands and `/` completions derive from it, nothing is hardcoded. */
export const listWidgetApps = (): WidgetApp<never>[] => [...apps.values()].sort((a, b) => a.id.localeCompare(b.id))

/** Resolve built-in presentation metadata without imposing i18n on user widgets. */
export const widgetHelp = (app: WidgetApp<never>, locale: Locale): string =>
  app.helpKey ? translate(locale, app.helpKey) : app.help

/** Resolve a rejected launch to localized built-in usage or extension fallback. */
export const widgetUsage = (app: WidgetApp<never>, locale: Locale): string =>
  app.usageKey ? translate(locale, app.usageKey) : (app.usage ?? translate(locale, 'widget.usage', { id: app.id }))
