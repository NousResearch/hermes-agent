import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * T4.3 — prefers-reduced-motion. The dashboard animates drawer slides, toast
 * entrances, tooltip fades, and search-hit smooth scroll; users with
 * `prefers-reduced-motion: reduce` must get none of them. Source-level
 * assertions (CSS ships as-is to the browser, and jsdom has no media engine).
 */

const css = readFileSync(path.resolve(process.cwd(), "src/index.css"), "utf8");
const sessions = readFileSync(
  path.resolve(process.cwd(), "src/pages/SessionsPage.tsx"),
  "utf8",
);

describe("prefers-reduced-motion (T4.3)", () => {
  it("index.css neutralises transitions, animations, and smooth scrolling", () => {
    expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)/);
    expect(css).toMatch(/animation:\s*none\s*!important/);
    expect(css).toMatch(/transition-duration:\s*0ms\s*!important/);
    expect(css).toMatch(/scroll-behavior:\s*auto\s*!important/);
  });

  it("SessionsPage gates the search-hit scroll on the media query", () => {
    expect(sessions).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    expect(sessions).toMatch(/behavior:\s*\w+ \? "auto" : "smooth"/);
  });
});
