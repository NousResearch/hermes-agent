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
        style={{
          background:
            "radial-gradient(circle at top left, var(--warm-glow) 0%, transparent 42%), linear-gradient(180deg, rgba(255,255,255,0.02), transparent 24%)",
          opacity: 1,
        }}
      />

      {gpuTier > 0 && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[101]"
          style={{
            backgroundImage:
              "radial-gradient(rgba(255,255,255,0.035) 0.8px, transparent 0.8px)",
            backgroundSize: "24px 24px",
            opacity: "calc(0.22 * var(--noise-opacity-mul, 1))",
          }}
        />
      )}
    </>
  );
}
