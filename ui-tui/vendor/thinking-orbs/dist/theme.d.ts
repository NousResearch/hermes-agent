import { RefObject } from 'react';
import { OrbTheme } from './types';

/** Resolve the effective dark/light substrate for a mounted element. */
export declare function useResolvedDark(theme: OrbTheme, hostRef: RefObject<Element | null>): boolean;
/** Live `prefers-reduced-motion` — reduced users get a static frame. */
export declare function useReducedMotion(): boolean;
