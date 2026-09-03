export interface ModeOpts {
    [key: string]: number | undefined;
}
export declare function scaleCounts(opts: ModeOpts, scale: number): ModeOpts;
export declare function scaleRadii(opts: ModeOpts, scale: number): ModeOpts;
/** Base (fine) profiles per mode, before preset multipliers. */
export declare const BASE_PROFILES: Record<string, ModeOpts>;
