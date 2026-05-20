/**
 * Plain, theme-aware dashboard backdrop.
 *
 * Keep the dashboard background intentionally boring: one solid color from
 * the active theme, no image layers, no blend modes, no glow, no noise.
 */
export function Backdrop() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 z-[1]"
      style={{ backgroundColor: "var(--background-base)" }}
    />
  );
}
