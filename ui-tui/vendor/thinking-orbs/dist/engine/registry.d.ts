import { ModeKey } from '../presets';
import { ModeDraw, ModeFrame } from './types';

/**
 * The portable surface: pure geometry, no canvas. The React Native port
 * imports exactly these functions, so its output is identical to the web's
 * by construction rather than by re-implementation.
 */
export declare const MODE_FRAMES: Record<ModeKey, ModeFrame>;
/** Canvas painters, derived from the geometry. The 2D-canvas binding. */
export declare const MODE_DRAWS: Record<ModeKey, ModeDraw>;
