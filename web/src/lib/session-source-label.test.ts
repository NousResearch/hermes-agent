import { describe, expect, it } from "vitest";

import { sourceLabel } from "./session-source-label";

describe("sourceLabel", () => {
  // A minimal stand-in for the active locale's translations: only
  // sessions.sources matters for this function.
  const faTranslations = {
    sessions: {
      sources: {
        api_server: "سرور API",
        acp: "ACP",
        cli: "CLI",
        tui: "TUI",
        telegram: "Telegram",
        discord: "Discord",
        slack: "Slack",
        whatsapp: "WhatsApp",
        whatsapp_cloud: "واتس‌اپ کلاد",
        sms: "پیامک",
        cron: "کرون",
        tool: "ابزار",
        hermes_flow: "جریان هرمس",
        vulcan_delegate: "نماینده ولکان",
        webhook: "وب‌هوک",
        local: "محلی",
      },
    },
  } as unknown as Parameters<typeof sourceLabel>[0];

  const enTranslations = {
    sessions: {
      sources: {
        api_server: "API server",
        cron: "Cron",
        local: "Local",
      },
    },
  } as unknown as Parameters<typeof sourceLabel>[0];

  it("prefers the active locale's label when the map has the source", () => {
    expect(sourceLabel(faTranslations, "api_server")).toBe("سرور API");
    expect(sourceLabel(faTranslations, "cron")).toBe("کرون");
    expect(sourceLabel(faTranslations, "webhook")).toBe("وب‌هوک");
    expect(sourceLabel(enTranslations, "api_server")).toBe("API server");
  });

  it("falls back to the English default when the locale map omits the source", () => {
    // enTranslations deliberately lacks most keys: these must come from the
    // built-in English defaults, not the (missing) localized entries.
    expect(sourceLabel(enTranslations, "telegram")).toBe("Telegram");
    expect(sourceLabel(enTranslations, "whatsapp_cloud")).toBe("WhatsApp Cloud");
    expect(sourceLabel(enTranslations, "vulcan_delegate")).toBe("Vulcan delegate");
    expect(sourceLabel(enTranslations, "hermes_flow")).toBe("Hermes Flow");
  });

  it("title-cases unknown sources not present in either map or the known list", () => {
    expect(sourceLabel(faTranslations, "my_custom_source")).toBe("My Custom Source");
    expect(sourceLabel(null, "weird_protocol_bridge")).toBe("Weird Protocol Bridge");
  });

  it("handles a null translations object by using the English fallbacks", () => {
    expect(sourceLabel(null, "api_server")).toBe("API server");
    expect(sourceLabel(null, "cron")).toBe("Cron");
    expect(sourceLabel(null, "acp")).toBe("ACP");
  });

  it("treats an empty-string localized label as a real value, not a miss", () => {
    const emptyLabel = {
      sessions: {
        sources: { cli: "" },
      },
    } as unknown as Parameters<typeof sourceLabel>[0];
    expect(sourceLabel(emptyLabel, "cli")).toBe("");
  });

  it("collapses leading/trailing/double underscores when title-casing", () => {
    expect(sourceLabel(null, "_leading")).toBe("Leading");
    expect(sourceLabel(null, "double__underscore")).toBe("Double Underscore");
    expect(sourceLabel(null, "__")).toBe("");
  });
});
