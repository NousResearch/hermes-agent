// Side-effect import: ticker registers itself at module load and nothing
// imports its app object, so a bare import is the whole contract.
import './ticker.js'

/** Reference apps. Importing this module registers them (defineWidgetApp
 *  runs at module load) — appLayout imports it once at startup. User widgets
 *  from $HERMES_HOME/tui-widgets ride the same import (async, non-fatal). */
import { loadUserWidgets, watchUserWidgets } from '../userWidgets.js'

void loadUserWidgets()
watchUserWidgets()

export { dialogTestApp } from './dialogTest.js'
export { gridTestApp } from './gridTest.js'
export { weatherApp, type WeatherState } from './weather.js'
