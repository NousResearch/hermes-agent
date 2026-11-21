import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * T4.4 — contrast floors on the default theme canvas (#041c1c).
 *
 * Measured before this change (WCAG relative luminance, cream midground
 * composited over the teal canvas):
 *   - text-disabled default (45% alpha)  -> 3.87:1  (below AA 4.5)
 *   - text-muted-foreground/40 call site (80% * 40% = 32% alpha) -> 2.58:1
 *
 * The dashboard now pins `--text-disabled` to 60% alpha (5.92:1) and the
 * `/40` call sites are raised so the effective alpha clears 4.5:1; the
 * base `muted-foreground` slot (text-secondary, 80%) sits at 9.67:1.
 */

const css = readFileSync(path.resolve(process.cwd(), "src/index.css"), "utf8");

function srgbLin(channel: number): number {
  const c = channel / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(r: number, g: number, b: number): number {
  return 0.2126 * srgbLin(r) + 0.7152 * srgbLin(g) + 0.0722 * srgbLin(b);
}

/** Composite `alpha` of the default midground over the default canvas. */
function composite(alpha: number): [number, number, number] {
  const fg = [255, 230, 203]; // LENS_0 midground #ffe6cb
  const bg = [4, 28, 28]; // LENS_0 background #041c1c
  return fg.map((f, i) => alpha * f + (1 - alpha) * bg[i]) as [number, number, number];
}

function contrastAgainstCanvas(alpha: number): number {
  const fgL = luminance(...composite(alpha));
  const bgL = luminance(4, 28, 28);
  return (Math.max(fgL, bgL) + 0.05) / (Math.min(fgL, bgL) + 0.05);
}

describe("default-theme contrast floors (T4.4)", () => {
  it("pins --text-disabled to 60% alpha on :root", () => {
    expect(css).toMatch(/--text-disabled:\s*color-mix\(in srgb,\s*var\(--midground-base\)\s*60%,\s*transparent\)/);
  });

  it("the disabled slot clears WCAG AA 4.5:1 on the default canvas", () => {
    // Default slot before the fix was 45% -> 3.87:1; pinned 60% -> 5.92:1.
    expect(contrastAgainstCanvas(0.45)).toBeLessThan(4.5);
    expect(contrastAgainstCanvas(0.6)).toBeGreaterThanOrEqual(4.5);
  });

  it("the muted-foreground base slot clears AA (text-secondary, 80%)", () => {
    expect(contrastAgainstCanvas(0.8)).toBeGreaterThanOrEqual(4.5);
    // The old `muted-foreground/40` composite (32% effective) did not.
    expect(contrastAgainstCanvas(0.32)).toBeLessThan(4.5);
  });

  it("no call site stacks muted-foreground at /40 any more", () => {
    const auth = readFileSync(path.resolve(process.cwd(), "src/components/AuthWidget.tsx"), "utf8");
    const profiles = readFileSync(path.resolve(process.cwd(), "src/pages/ProfilesPage.tsx"), "utf8");
    expect(auth).not.toContain("muted-foreground/40");
    expect(profiles).not.toContain("muted-foreground/40");
  });
});
