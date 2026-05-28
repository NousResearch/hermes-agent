import { useGpuTier } from "@nous-research/ui/hooks/use-gpu-tier";

/**
 * Theme-aware Feishu-style background stack: a quiet neutral workspace with
 * only the faintest depth so content remains the first thing the eye catches.
 */
export function Backdrop() {
  const gpuTier = useGpuTier();

  return (
    <>
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[1]"
        style={{
          background:
            "linear-gradient(180deg, #ffffff 0%, #f8f9fb 46%, #f7f8fa 100%)",
        }}
      />

      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[2]"
        style={
          {
            backgroundImage:
              "var(--theme-asset-bg, none), radial-gradient(circle at 22% 0%, rgba(51, 112, 255, 0.025), transparent 30%), radial-gradient(circle at 92% 4%, rgba(31, 35, 41, 0.018), transparent 28%)",
            backgroundSize: "var(--component-backdrop-background-size, cover)",
            backgroundPosition:
              "var(--component-backdrop-background-position, center)",
            opacity: "var(--component-backdrop-filler-opacity, 1)",
          } as unknown as React.CSSProperties
        }
      />

      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[3] opacity-0"
        style={{
          backgroundImage:
            "linear-gradient(rgba(31, 35, 41, 0.018) 1px, transparent 1px), linear-gradient(90deg, rgba(31, 35, 41, 0.014) 1px, transparent 1px)",
          backgroundSize: "56px 56px",
          maskImage:
            "linear-gradient(180deg, rgba(0,0,0,0.64), rgba(0,0,0,0.16) 58%, transparent)",
        }}
      />

      <div
        aria-hidden
        className="pointer-events-none fixed right-[7vw] top-[8dvh] z-[4] h-72 w-72 rounded-full"
        style={{
          background:
            "linear-gradient(145deg, rgba(51,112,255,0.025), rgba(255,255,255,0.16))",
          filter: "blur(18px)",
          opacity: 0.18,
        }}
      />

      <div
        aria-hidden
        className="pointer-events-none fixed bottom-[13dvh] left-[17vw] z-[4] h-44 w-44 rounded-full"
        style={{
          background:
            "linear-gradient(135deg, rgba(51,112,255,0.018), rgba(255,255,255,0.08))",
          filter: "blur(16px)",
          opacity: 0.14,
        }}
      />

      {gpuTier > 0 && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[5]"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.78' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' fill='%233370ff' filter='url(%23n)' opacity='0.18'/%3E%3C/svg%3E\")",
            backgroundSize: "512px 512px",
            mixBlendMode: "soft-light",
            opacity: "calc(0.12 * var(--noise-opacity-mul, 1))",
          }}
        />
      )}
    </>
  );
}
