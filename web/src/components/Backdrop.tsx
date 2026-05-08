import { useGpuTier } from "@nous-research/ui/hooks/use-gpu-tier";

/**
 * Replicates the visual layer stack of `<Overlays dark />` from
 * `@nous-research/ui` without pulling in its leva / gsap / three peer deps.
 *
 * See `design-language/src/ui/components/overlays/index.tsx` for the source of
 * truth. Defaults match LENS_0 (the Hermes teal dark preset); the deep canvas
 * and the warm vignette both read theme-switchable CSS custom properties so
 * `ThemeProvider` can repaint the stack without remounting.
 *
 *   z-1   bg = `var(--background-base)`, mix-blend-mode: difference
 *   z-2   filler-bg jpeg, inverted, opacity 0.033, difference
 *   z-99  warm top-left vignette (`var(--warm-glow)`), opacity 0.22, lighten
 *   z-101 noise grain (SVG, ~55% opacity × `--noise-opacity-mul`,
 *         color-dodge) — gated on GPU tier
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
        style={{
          backgroundColor: "var(--background-base)",
          mixBlendMode: "difference",
        }}
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
          src="/ds-assets/filler-bg0.jpg"
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

      {gpuTier > 0 && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[101]"
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
