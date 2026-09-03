import { ModeOpts } from './engine/profiles';
import { OrbSize, OrbState } from './types';

export type ModeKey = 'orbits' | 'globe' | 'rubik' | 'wave' | 'web' | 'braid' | 'ribbon' | 'ring' | 'morph';
export declare const STATE_TO_MODE: Record<OrbState, ModeKey>;
export interface Preset {
    speed: number;
    count: number;
    size: number;
    /** Extra mode opts merged verbatim after scaling. */
    extra?: ModeOpts;
}
/** Exported so `scripts/extract-spec.ts` can emit them for the native ports. */
export declare const PRESETS: Record<ModeKey, Record<OrbSize, Preset>>;
export interface Resolved {
    mode: ModeKey;
    speed: number;
    opts: ModeOpts;
}
/** Resolve a (state, size) pair to its mode + fully-scaled draw options. */
export declare function resolvePreset(state: OrbState, size: OrbSize): Resolved;
