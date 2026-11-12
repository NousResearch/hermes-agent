/** Regression tests for #76381: the sidebar / footer-left / footer-right
 *  slots are declared in KNOWN_SLOT_NAMES and documented in
 *  extending-the-dashboard.md, but the shell never rendered them — a plugin
 *  targeting them mounted into nothing (silent no-op).
 *
 *  The footer slots live in SidebarFooter with the default cells as
 *  `fallback` (plugin content replaces the default when registered); the
 *  sidebar slot renders only when layoutVariant === "cockpit"
 *  (CockpitSidebarSlot). Tests drive the REAL production components via
 *  renderToStaticMarkup — no jsdom needed.
 *
 *  On the pre-fix tree the "replaces" assertions fail (no PluginSlot in
 *  SidebarFooter) and CockpitSidebarSlot does not exist to import. */
import { describe, it, expect, afterEach } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { registerSlot, unregisterPluginSlots } from "@/plugins/slots";
import { SidebarFooter } from "@/components/SidebarFooter";
import { CockpitSidebarSlot } from "@/components/CockpitSidebarSlot";

const Mark = ({ text }: { text: string }) =>
  React.createElement("span", null, text);

afterEach(() => {
  unregisterPluginSlots("test-plugin");
});

describe("footer-left / footer-right slots (SidebarFooter)", () => {
  it("renders the default cells as fallback when no plugin claims them", () => {
    const html = renderToStaticMarkup(
      React.createElement(SidebarFooter, {
        status: { version: "1.2.3" } as never,
      }),
    );
    expect(html).toContain("v1.2.3");
    expect(html).toContain("nousresearch.com");
  });

  it("plugin content replaces the default left cell", () => {
    registerSlot("test-plugin", "footer-left", () =>
      React.createElement(Mark, { text: "PLUGIN-LEFT" }),
    );
    const html = renderToStaticMarkup(
      React.createElement(SidebarFooter, {
        status: { version: "1.2.3" } as never,
      }),
    );
    expect(html).toContain("PLUGIN-LEFT");
    expect(html).not.toContain("v1.2.3");
  });

  it("plugin content replaces the default right cell", () => {
    registerSlot("test-plugin", "footer-right", () =>
      React.createElement(Mark, { text: "PLUGIN-RIGHT" }),
    );
    const html = renderToStaticMarkup(
      React.createElement(SidebarFooter, { status: null }),
    );
    expect(html).toContain("PLUGIN-RIGHT");
    expect(html).not.toContain("nousresearch.com");
  });
});

describe("sidebar slot (CockpitSidebarSlot)", () => {
  it("renders plugin content in the cockpit variant", () => {
    registerSlot("test-plugin", "sidebar", () =>
      React.createElement(Mark, { text: "COCKPIT-PLUGIN" }),
    );
    const html = renderToStaticMarkup(
      React.createElement(CockpitSidebarSlot, { layoutVariant: "cockpit" }),
    );
    expect(html).toContain("COCKPIT-PLUGIN");
  });

  it("renders nothing in the standard variant", () => {
    registerSlot("test-plugin", "sidebar", () =>
      React.createElement(Mark, { text: "COCKPIT-PLUGIN" }),
    );
    const html = renderToStaticMarkup(
      React.createElement(CockpitSidebarSlot, { layoutVariant: "standard" }),
    );
    expect(html).not.toContain("COCKPIT-PLUGIN");
  });

  it("renders nothing in the cockpit variant when unclaimed", () => {
    const html = renderToStaticMarkup(
      React.createElement(CockpitSidebarSlot, { layoutVariant: "cockpit" }),
    );
    expect(html).not.toContain("COCKPIT-PLUGIN");
  });
});
