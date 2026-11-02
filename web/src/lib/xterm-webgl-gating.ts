/**
 * Gate the dashboard chat's xterm WebGL renderer.
 *
 * The WebGL renderer is the fast path, but a whole class of hosts and content
 * renders wrong or dies on it:
 *
 *  - Safari's WebGL texture atlas mangles Unicode box-drawing glyphs
 *    (╔╗║╚╝, the HERMES banner) — the DOM renderer draws them fine (#18773).
 *  - Software rasterizers (llvmpipe, SwiftShader, softpipe) succeed at
 *    `getContext("webgl")` but regl then throws
 *    "(regl) webgl not supported" from inside the addon (#45520).
 *  - Atlas drawing is per code point, so complex scripts that need shaping
 *    (Bengali conjuncts যুক্তাক্ষর and friends) come out visually broken;
 *    the DOM renderer runs text through the browser's shaper (#58685).
 *
 * Decision helpers here are pure; `probeWebglSupport` is the only part that
 * touches the DOM.
 */

export interface WebglSupport {
  /** A WebGL(2) context could be created at all. */
  contextAvailable: boolean;
  /** The unmasked renderer string looks like a software rasterizer. */
  softwareRenderer: boolean;
}

// Renderer strings reported by the rasterizers that advertise WebGL but
// can't back regl / the addon usefully.
const SOFTWARE_RENDERER_RE = /llvmpipe|swiftshader|softpipe|software/i;

/**
 * Probe once whether WebGL is usable on this host, then release the probe
 * context: browsers cap live contexts (~16) and chat reconnects already fight
 * that cap (see xterm-webgl-release.ts).
 */
export function probeWebglSupport(doc: Document): WebglSupport {
  let gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  try {
    const canvas = doc.createElement("canvas");
    gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) || null;
  } catch {
    // WebGL disabled or blocked — fall through to "unavailable".
  }
  if (!gl) return { contextAvailable: false, softwareRenderer: false };

  let softwareRenderer = false;
  try {
    const debug = gl.getExtension("WEBGL_debug_renderer_info");
    if (debug) {
      const renderer = String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL));
      softwareRenderer = SOFTWARE_RENDERER_RE.test(renderer);
    }
  } catch {
    // Renderer info stripped (privacy modes) — context creation succeeded,
    // so give the hardware path the benefit of the doubt.
  }
  try {
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    // Already lost — nothing to release.
  }
  return { contextAvailable: true, softwareRenderer };
}

/**
 * Detect Safari (the WebKit browser shipped with macOS/iOS). Chromium
 * derivatives and iOS wrapper apps keep the legacy `Safari/` token in their
 * UA for compat, so exclude every fingerprint we know wraps another engine.
 */
export function isSafari(userAgent: string): boolean {
  if (!userAgent.includes("Safari/")) return false;
  return !/Chrom(e|ium)\/|CriOS\/|FxiOS\/|EdgiOS\/|EdgA\/|SamsungBrowser\/|Android/i.test(
    userAgent,
  );
}

/**
 * Wide layouts get WebGL for crisp box-drawing — but only where it works:
 * never on Safari (#18773), never without a real hardware GL context
 * (#45520). Everything else stays on the default DOM renderer.
 */
export function shouldUseWebglRenderer(input: {
  layoutWidthPx: number;
  userAgent: string;
  support: WebglSupport;
}): boolean {
  return (
    input.layoutWidthPx >= 768 &&
    !isSafari(input.userAgent) &&
    input.support.contextAvailable &&
    !input.support.softwareRenderer
  );
}

// Unicode blocks for scripts that need contextual shaping (conjuncts,
// ligatures, reordering) — impossible for a per-glyph atlas. Southeast Asian
// and Indic families; Bengali (U+0980–U+09FF) is the reported case (#58685).
const COMPLEX_SCRIPT_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x0900, 0x097f], // Devanagari
  [0x0980, 0x09ff], // Bengali
  [0x0a00, 0x0a7f], // Gurmukhi
  [0x0a80, 0x0aff], // Gujarati
  [0x0b00, 0x0b7f], // Oriya
  [0x0b80, 0x0bff], // Tamil
  [0x0c00, 0x0c7f], // Telugu
  [0x0c80, 0x0cff], // Kannada
  [0x0d00, 0x0d7f], // Malayalam
  [0x0d80, 0x0dff], // Sinhala
  [0x1000, 0x109f], // Myanmar
  [0x1780, 0x17ff], // Khmer
];

/**
 * True when the payload contains a script the WebGL/canvas atlas cannot
 * shape. Used to drop the WebGL addon mid-stream so xterm re-renders
 * through the DOM renderer (#58685).
 */
export function textNeedsDomShaping(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const codePoint = text.codePointAt(i) ?? 0;
    if (codePoint > 0xffff) i += 1; // surrogate pair — skip the low half
    for (const [low, high] of COMPLEX_SCRIPT_RANGES) {
      if (codePoint >= low && codePoint <= high) return true;
    }
  }
  return false;
}
