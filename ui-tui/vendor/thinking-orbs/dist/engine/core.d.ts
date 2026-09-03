export interface Dot {
    x: number;
    y: number;
    z: number;
    r: number;
    /** Ink value: 0 = darkest ink on paper. Mirrored on dark themes. */
    white: number;
    a?: number;
}
/** A stroked edge between two projected points (the `connecting` web). */
export interface Line {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    /** Ink value, same convention as `Dot.white`. */
    white: number;
    a?: number;
    w: number;
}
/**
 * One rendered instant: a complete, final set of draw instructions.
 * `dots` is already z-sorted into draw order and radius-clamped; `lines`
 * are drawn first. Nothing here needs further interpretation, which is what
 * makes a frame portable to any 2D renderer.
 */
export interface OrbFrame {
    dots: Dot[];
    lines: Line[];
}
export type Projector = (x: number, y: number, z: number) => [number, number, number];
export declare function lerp(a: number, b: number, f: number): number;
export declare function frac(x: number): number;
/** Value noise on a 2D lattice — smooth, deterministic, cheap. */
export declare function vnoise(x: number, y: number): number;
/** Deterministic hash in [0, 1). */
export declare function hashD(a: number, b: number): number;
/** Stable directions on a unit sphere (Fibonacci lattice). */
export declare function fibDir(i: number, n: number): [number, number, number];
/** Shortest signed angular distance, wrapped to (-π, π]. */
export declare function angleDelta(a: number, b: number): number;
/** Shared spin + tilt + orthographic projection. */
export declare function makeProj(yaw: number, tilt: number, cx: number, cy: number, scale: number): Projector;
/**
 * Painter: z-sort far→near, matte grayscale dots. On dark substrates the
 * ink value is mirrored (1 - white) so near dots read bright — the same
 * depth language on an inverted substrate.
 */
export declare function paint(ctx: CanvasRenderingContext2D, dots: Dot[], dark: boolean, rMin?: number): void;
/** Stroke pass for edge-based modes. Runs before `paint` so nodes sit on top. */
export declare function paintLines(ctx: CanvasRenderingContext2D, lines: Line[], dark: boolean): void;
/**
 * Turn raw mode output into a finished frame: drop invisible marks, clamp
 * radii to the mode's floor, and z-sort far→near into draw order.
 *
 * This runs in the GEOMETRY step, not the painter, so a frame is a complete
 * set of draw instructions: every value is final and the array order is the
 * order to draw in. That is what lets the RN and SwiftUI ports share this
 * output verbatim — a port draws the list, it never re-derives anything —
 * and what lets the golden-vector tests compare numbers instead of pixels.
 */
export declare function finalizeFrame(dots: Dot[], lines: Line[], rMin?: number): OrbFrame;
/** Paint a finished frame. Lines first, so nodes sit on top of their edges. */
export declare function paintFrame(ctx: CanvasRenderingContext2D, frame: OrbFrame, dark: boolean): void;
/**
 * Dot radii were tuned for a 300pt frame; sub-linear scaling keeps small
 * spinners legible. Lower pow = radii shrink less with size.
 */
export declare function radiusScale(size: number, pow: number): number;
