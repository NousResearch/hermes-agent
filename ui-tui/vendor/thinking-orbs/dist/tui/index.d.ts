import { OrbState } from '../types';

export interface TuiRenderOptions {
    /** Terminal-cell width. @default 16 */
    columns?: number;
    /** Terminal-cell height. @default 8 */
    rows?: number;
    /** Elapsed animation time in seconds. @default 0 */
    time?: number;
    /** Multiplier on the state's tuned animation speed. @default 1 */
    speed?: number;
    /** Terminal substrate used to mirror the engine's ink values. @default 'dark' */
    theme?: 'dark' | 'light';
    /** Minimum subpixel energy before ordered dithering. @default 0.18 */
    threshold?: number;
    /** Optional terminal color/style hook. It must preserve the glyph's display width. */
    paint?: (glyph: string, intensity: number) => string;
}
/** A terminal frame plus per-cell intensity for renderers that manage color separately. */
export interface TuiOrbFrame {
    lines: string[];
    intensities: number[][];
}
/**
 * Project the package's exact particle geometry onto Unicode Braille cells.
 * No React, DOM, canvas, ANSI library, or terminal framework is required.
 */
export declare function renderTuiOrb(state: OrbState, options?: TuiRenderOptions): TuiOrbFrame;
