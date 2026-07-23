import { describe, expect, it } from "vitest";
import {
  DASHBOARD_MODAL_BACKDROP,
  DASHBOARD_MODAL_PANEL,
  DASHBOARD_MOBILE_DRAWER_WIDTH,
  dashboardDrawerA11yProps,
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
    expect(DASHBOARD_MODAL_BACKDROP).toContain("hermes-modal-backdrop");
    expect(DASHBOARD_MODAL_PANEL).toContain("hermes-modal-panel-viewport");
  });

  it("does not close the outer modal on Escape while a nested picker is open", () => {
    expect(shouldCloseOuterModalOnEscape(true)).toBe(false);
    expect(shouldCloseOuterModalOnEscape(false)).toBe(true);
  });

  it("keeps mobile drawer ARIA and inert state synchronized", () => {
    expect(dashboardDrawerA11yProps(false)).toEqual({
      "aria-hidden": true,
      "aria-modal": undefined,
      inert: true,
      role: "dialog",
    });
    expect(dashboardDrawerA11yProps(true)).toEqual({
      "aria-hidden": false,
      "aria-modal": true,
      inert: undefined,
      role: "dialog",
    });
    expect(dashboardDrawerA11yProps(false, false)).toEqual({});
  });

  it("bounds mobile drawers without changing the desktop rail width", () => {
    expect(DASHBOARD_MOBILE_DRAWER_WIDTH).toContain(
      "max-w-[calc(100vw-0.75rem)]",
    );
    expect(DASHBOARD_MOBILE_DRAWER_WIDTH).toContain("lg:max-w-none");
    expect(DASHBOARD_MOBILE_DRAWER_WIDTH).toContain("w-64");
  });
});
