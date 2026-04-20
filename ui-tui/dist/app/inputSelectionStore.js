import { atom } from 'nanostores';
export const $inputSelection = atom(null);
export const setInputSelection = (next) => $inputSelection.set(next);
export const getInputSelection = () => $inputSelection.get();
