import { atom } from 'nanostores';
const buildTurnState = () => ({
    activity: [],
    outcome: '',
    reasoning: '',
    reasoningActive: false,
    reasoningStreaming: false,
    reasoningTokens: 0,
    streamPendingTools: [],
    streamSegments: [],
    streaming: '',
    subagents: [],
    toolTokens: 0,
    tools: [],
    turnTrail: []
});
export const $turnState = atom(buildTurnState());
export const getTurnState = () => $turnState.get();
export const patchTurnState = (next) => $turnState.set(typeof next === 'function' ? next($turnState.get()) : { ...$turnState.get(), ...next });
export const resetTurnState = () => $turnState.set(buildTurnState());
