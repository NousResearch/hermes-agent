/** Reference apps. Importing this module registers them (defineWidgetApp
 *  runs at module load) — appLayout imports it once at startup. */
export { dialogTestApp } from './dialogTest.js'
export { gridTestApp } from './gridTest.js'
export { GRID_STREAM_COUNT, type GridTestState } from './gridTestState.js'
export { tickerApp, type TickerState } from './ticker.js'
export { weatherApp, type WeatherState } from './weather.js'
