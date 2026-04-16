import { atom } from 'nanostores'

import { ZERO } from '../domain/usage.js'
import { DEFAULT_THEME } from '../theme.js'

import type { UiState } from './interfaces.js'

function buildUiState(): UiState {
  return {
    bgTasks: new Set(),
    busy: false,
    compact: false,
    detailsMode: 'collapsed',
    info: null,
    sid: null,
    status: 'summoning hermes…',
    statusBar: true,
    theme: DEFAULT_THEME,
    usage: ZERO
  }
}

export const $uiState = atom<UiState>(buildUiState())

export function getUiState() {
  return $uiState.get()
}

export function patchUiState(next: Partial<UiState> | ((state: UiState) => UiState)) {
  if (typeof next === 'function') {
    $uiState.set(next($uiState.get()))

    return
  }

  $uiState.set({ ...$uiState.get(), ...next })
}

export function resetUiState() {
  $uiState.set(buildUiState())
}
