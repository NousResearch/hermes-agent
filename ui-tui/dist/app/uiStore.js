import { atom } from 'nanostores';
import { ZERO } from '../domain/usage.js';
import { DEFAULT_THEME } from '../theme.js';
const buildUiState = () => ({
    bgTasks: new Set(),
    busy: false,
    compact: false,
    detailsMode: 'collapsed',
    info: null,
    inlineDiffs: true,
    showCost: false,
    showReasoning: false,
    sid: null,
    status: 'summoning hermes…',
    statusBar: true,
    streaming: true,
    theme: DEFAULT_THEME,
    usage: ZERO
});
export const $uiState = atom(buildUiState());
export const getUiState = () => $uiState.get();
export const patchUiState = (next) => $uiState.set(typeof next === 'function' ? next($uiState.get()) : { ...$uiState.get(), ...next });
export const resetUiState = () => $uiState.set(buildUiState());
