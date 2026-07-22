import { describe, expect, it } from "vitest";
import {
  DASHBOARD_MODAL_BACKDROP,
  DASHBOARD_MODAL_PANEL,
  shouldCloseOuterModalOnEscape,
} from "./dashboard-modal-shell";

describe("dashboard modal shell", () => {
  it("uses an opaque panel (bg-card), not the glass Card default", () => {
    expect(DASHBOARD_MODAL_PANEL).toMatch(/\bbg-card\b/);
    expect(DASHBOARD_MODAL_PANEL).not.toMatch(/bg-background-base\/\d+/);
  });

  it("keeps the backdrop above page chrome (z-[100])", () => {
    expect(DASHBOARD_MODAL_BACKDROP).toMatch(/z-\[100\]/);
    expect(DASHBOARD_MODAL_BACKDROP).toMatch(/\bbg-background\/85\b/);
  });

  it("does not close the outer modal on Escape while a nested picker is open", () => {
    expect(shouldCloseOuterModalOnEscape(true)).toBe(false);
    expect(shouldCloseOuterModalOnEscape(false)).toBe(true);
  });
});
