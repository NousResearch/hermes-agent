import { useGpuTier } from "@nous-research/ui/hooks/use-gpu-tier";

import fillerBgUrl from "@nous-research/ui/assets/filler-bg0.webp";

/**
 * Replicates the visual layer stack of `<Overlays dark />` from
 * `@nous-research/ui` without pulling in its leva / gsap / three peer deps.
 *
 * See `design-language/src/ui/components/overlays/index.tsx` for the source of
 * truth. Defaults match LENS_0 (the Hermes teal dark preset); the deep canvas
 * and the warm vignette both read theme-switchable CSS custom properties so
 * `ThemeProvider` can repaint the stack without remounting.
 *
 *   z-1   bg = `var(--background-base)`, mix-blend-mode driven by
 *         `--component-backdrop-bg-blend-mode` (default `difference`).
 *         Both LENS_0-style dark themes and the LENS_5I-style Nous Blue
 *         light theme keep `difference` here — the canvas is flipped by
 *         the z-200 FG inversion layer, not by changing this blend mode.
 *         The CSS var is exposed as a hook so future presets can override
 *         it (e.g. `multiply` to paint the bg as-is before inversion)
 *         without touching this component.
 *   z-2   bundled filler-bg WebP, inverted, opacity 0.033, difference
 *   z-99  warm top-left vignette (`var(--warm-glow)`), opacity 0.22, lighten
 *   z-200 FG inversion = `var(--foreground)` (opaque white in LENS_5I,
 *         alpha-0 in LENS_0), mix-blend-mode: difference. This is the
 *         layer that flips the dashboard into "light mode" for inverted
 *         themes; for normal dark themes its alpha is 0 so it's a no-op.
 *         Deliberately placed above every UI overlay z-index (modals,
 *         tooltips, and dropUp dropdowns all sit at z-[100]) so portaled
 *         elements get inverted along with the rest of the page instead
 *         of painting with pre-inversion colors on top of the lens.
 *   z-201 noise grain (SVG, ~55% opacity × `--noise-opacity-mul`,
 *         color-dodge) — gated on GPU tier. Sits above the inversion
 *         layer by design so the grain is not flipped.
 *
 * `useGpuTier` returns 0 when WebGL is unavailable, the renderer is a
 * software rasterizer (SwiftShader/llvmpipe), or the user has
 * `prefers-reduced-motion: reduce` set. We skip the animated noise layer
 * in that case so low-power / accessibility-conscious sessions stay crisp,
 * mirroring the DS `<Noise />` component's own opt-out.
 */
export function Backdrop() {
  const gpuTier = useGpuTier();

  return (
    <>
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[1]"
        style={
          {
            backgroundColor: "var(--background-base)",
            mixBlendMode:
              "var(--component-backdrop-bg-blend-mode, difference)",
          } as unknown as React.CSSProperties
        }
      />

      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[2]"
        style={
          {
            // Themes can override the filler background by setting
            // `assets.bg` — the <img> hides itself when a CSS bg is set
            // so the two don't double-darken. CSS var fallbacks keep the
            // default behaviour unchanged when no theme customises these.
            mixBlendMode:
              "var(--component-backdrop-filler-blend-mode, difference)",
            opacity: "var(--component-backdrop-filler-opacity, 0.033)",
            backgroundImage: "var(--theme-asset-bg)",
            backgroundSize: "var(--component-backdrop-background-size, cover)",
            backgroundPosition:
              "var(--component-backdrop-background-position, center)",
          } as unknown as React.CSSProperties
        }
      >
        <img
          alt=""
          className="h-[150dvh] w-auto min-w-[100dvw] object-cover object-top-left invert theme-default-filler"
          fetchPriority="low"
          src={fillerBgUrl}
        />
      </div>

      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[99]"
        style={{
          background:
            "radial-gradient(ellipse at 0% 0%, transparent 60%, var(--warm-glow) 100%)",
          mixBlendMode: "lighten",
          opacity: 0.22,
        }}
      />

      {/* Foreground inversion layer. Source-of-truth: LENS_5I.Lens.fgOpacity
          + fgBlend: 'difference' in `design-language/src/ui/components/
          overlays/lens.ts`. With `--foreground-alpha: 0` (LENS_0 dark default)
          the layer is fully transparent and contributes nothing; with
          alpha 1 + opaque white it inverts the entire stack below it,
          producing the LENS_5I "light mode" look without altering any
          downstream component code.

          z-200 (not 100) so it sits above every portaled UI overlay —
          sidebar tooltips, dropUp dropdowns, and modal dialogs all use
          z-[100], which is what the DS Lens picks too; portals append
          at the end of <body>, so equal z-index + later DOM order means
          they'd paint on top of the inversion and skip the flip. Inlined
          z-index for the same reason the DS does it — Tailwind's JIT
          scan sometimes drops non-default z utilities. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0"
        style={{
          backgroundColor: "var(--foreground)",
          mixBlendMode: "difference",
          zIndex: 200,
        }}
      />

      {gpuTier > 0 && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[201]"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' fill='%23eaeaea' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E\")",
            backgroundSize: "512px 512px",
            mixBlendMode: "color-dodge",
            opacity: "calc(0.55 * var(--noise-opacity-mul, 1))",
          }}
        />
      )}
    </>
  );
}
