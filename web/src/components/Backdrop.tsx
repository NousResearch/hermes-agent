import { useEffect, useState } from "react";

/** Legacy backdrop kept for older dashboard themes/plugins. Mission Control's
 * shell does not mount it, but the component no longer depends on the old Nous
 * design-system package.
 */
export function Backdrop() {
  const gpuTier = useLocalGpuTier();

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
            mixBlendMode: "var(--component-backdrop-filler-blend-mode, difference)",
            opacity: "var(--component-backdrop-filler-opacity, 0.033)",
            backgroundImage: "var(--theme-asset-bg)",
            backgroundSize: "var(--component-backdrop-background-size, cover)",
            backgroundPosition: "var(--component-backdrop-background-position, center)",
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

function useLocalGpuTier() {
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setEnabled(!media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  return enabled ? 1 : 0;
}
