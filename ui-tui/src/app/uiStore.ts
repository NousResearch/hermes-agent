import { atom } from 'nanostores'

import { MOUSE_TRACKING } from '../config/env.js'
import { ZERO } from '../domain/usage.js'
import { DEFAULT_THEME } from '../theme.js'

import { DEFAULT_INDICATOR_STYLE, type UiState } from './interfaces.js'

const buildUiState = (): UiState => ({
  bgTasks: new Set(),
  busy: false,
  busyInputMode: 'queue',
  compact: false,
  detailsMode: 'collapsed',
  detailsModeCommandOverride: false,
  indicatorStyle: DEFAULT_INDICATOR_STYLE,
  info: null,
  inlineDiffs: true,
  mouseTracking: MOUSE_TRACKING,
  sections: {},
  showCost: false,
  showReasoning: false,
  sid: null,
  status: 'summoning hermes…',
  statusBar: 'top',
  streaming: true,
  theme: DEFAULT_THEME,
  usage: ZERO
})

export const $uiState = atom<UiState>(buildUiState())

export const getUiState = () => $uiState.get()

export const patchUiState = (next: Partial<UiState> | ((state: UiState) => UiState)) =>
  $uiState.set(typeof next === 'function' ? next($uiState.get()) : { ...$uiState.get(), ...next })

export const resetUiState = () => $uiState.set(buildUiState())
