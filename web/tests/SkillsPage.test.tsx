/**
 * Unit test stub for hermes Plan 004-A SkillsPage.
 *
 * SkillsPage.tsx has heavy app-wide dependencies (api client, toast context,
 * page-header context, plugin system, i18n, Three.js for icons). A full
 * render needs an app shell + mocked context providers. This stub asserts
 * the component file is importable and exports a default React component;
 * deeper render-tree assertions live in the Playwright e2e spec in
 * agentic-hub/tests/e2e/hermes-skills.spec.ts where the running dev server
 * provides the contexts.
 *
 * Plan 004-B/C/D NOTE: PromotionPanel / DriftAlertPanel / RecommendedSkillsPanel
 * are NOT YET shipped in SkillsPage.tsx. Backend modules exist but the UI
 * surface was never wired. Tracked as a Phase G follow-up.
 */
import { describe, it, expect } from "vitest";

describe("hermes-004-A: SkillsPage module shape", () => {
  it("exports a default React component", async () => {
    const mod = await import("@/pages/SkillsPage");
    expect(mod.default).toBeTypeOf("function");
  });
});
