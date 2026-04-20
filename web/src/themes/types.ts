/**
 * Dashboard theme model.
 *
 * Unlike the pre-DS implementation (which overrode 21 shadcn tokens directly),
 * themes are now expressed in the Nous DS's own 3-triplet vocabulary —
 * `background`, `midground`, `foreground` — plus a warm-glow tint for the
 * vignette in <Backdrop />. All downstream shadcn-compat tokens
 * (`--color-card`, `--color-muted-foreground`, `--color-border`, etc.) are
 * defined in `src/index.css` as `color-mix()` expressions over the triplets,
 * so overriding the triplets at runtime cascades to every surface.
 */

/** A color layer: hex base + alpha (0–1). */
export interface ThemeLayer {
  alpha: number;
  hex: string;
}

export interface ThemePalette {
  /** Deepest canvas color (typically near-black). */
  background: ThemeLayer;
  /** Primary text + accent. Most UI chrome reads this. */
  midground: ThemeLayer;
  /** Top-layer highlight. In LENS_0 this is white @ alpha 0 — invisible by
   *  default but still drives `--color-ring`-style accents. */
  foreground: ThemeLayer;
  /** Warm vignette color for <Backdrop />, as an rgba() string. */
  warmGlow: string;
  /** Scalar multiplier (0–1.2) on the noise overlay. Lower for softer themes
   *  like Mono and Rosé, higher for grittier themes like Cyberpunk. */
  noiseOpacity: number;
}

export interface DashboardTheme {
  description: string;
  label: string;
  name: string;
  palette: ThemePalette;
}

export interface ThemeListResponse {
  active: string;
  themes: Array<{ description: string; label: string; name: string }>;
}
