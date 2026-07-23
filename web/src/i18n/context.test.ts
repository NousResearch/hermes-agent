import { afterEach, describe, expect, it, vi } from "vitest";
import { localeIntlTag } from "@hermes/shared/locale-registry";

import { api } from "../lib/api";
import { en } from "./en";
import { getLocaleFormatters } from "./formatters";
import {
  formatTranslation,
  normalizeLocale,
  persistConfiguredLocale,
  readConfiguredLocaleChange,
  resolveNavLabel,
  resolvePluginDescription,
  resolveTranslationOverlay,
  resolveTranslations,
} from "./runtime";
import { zh } from "./zh";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Dashboard i18n framework", () => {
  it("keeps Simplified Chinese complete for every English catalog leaf", () => {
    const missing: string[] = [];
    const visit = (base: unknown, overlay: unknown, path: string[] = []) => {
      if (!base || typeof base !== "object" || Array.isArray(base)) return;
      const translated =
        overlay && typeof overlay === "object" && !Array.isArray(overlay)
          ? (overlay as Record<string, unknown>)
          : {};
      for (const [key, value] of Object.entries(base)) {
        const childPath = [...path, key];
        if (!(key in translated)) {
          missing.push(childPath.join("."));
          continue;
        }
        visit(value, translated[key], childPath);
      }
    };

    visit(en, zh);
    expect(missing).toEqual([]);
  });

  it("keeps required placeholders aligned in every translated Simplified Chinese leaf", () => {
    const placeholders = (value: string) =>
      [...value.matchAll(/\{(\w+)\}/g)]
        .map((match) => match[1])
        // English uses {s} only as an optional plural suffix. Languages that
        // do not pluralize with an English suffix intentionally omit it.
        .filter((name) => name !== "s")
        .sort();
    const mismatches: Array<{
      path: string;
      english: string[];
      chinese: string[];
    }> = [];
    const visit = (base: unknown, overlay: unknown, path: string[] = []) => {
      if (typeof overlay === "string") {
        const english = placeholders(String(base));
        const chinese = placeholders(overlay);
        if (JSON.stringify(english) !== JSON.stringify(chinese)) {
          mismatches.push({ path: path.join("."), english, chinese });
        }
        return;
      }
      if (!overlay || typeof overlay !== "object" || Array.isArray(overlay))
        return;

      for (const [key, value] of Object.entries(overlay)) {
        visit((base as Record<string, unknown>)[key], value, [...path, key]);
      }
    };

    visit(en, zh);
    expect(mismatches).toEqual([]);
  });

  it("normalizes explicit language names and compatible BCP-47 inputs", () => {
    expect(normalizeLocale("Simplified Chinese")).toBe("zh");
    expect(normalizeLocale("traditional-chinese")).toBe("zh-hant");
    expect(normalizeLocale("zh-extra")).toBeNull();
    expect(normalizeLocale("pt_BR")).toBe("pt");
    expect(normalizeLocale("jp")).toBe("ja");
    expect(normalizeLocale("日本語")).toBe("ja");
    expect(normalizeLocale("한국어")).toBe("ko");
    expect(normalizeLocale("turkce")).toBe("tr");
    expect(normalizeLocale("francais")).toBe("fr");
    expect(normalizeLocale("brazilian")).toBe("pt");
    expect(normalizeLocale("ua")).toBe("uk");
  });

  it("uses explicit registry compatibility for an ambiguous language family", () => {
    expect(normalizeLocale("zh-CN")).toBe("zh");
    expect(normalizeLocale("zh_Hans")).toBe("zh");
    expect(normalizeLocale("zh-SG")).toBe("zh");
    expect(normalizeLocale("zh-TW")).toBe("zh-hant");
    expect(normalizeLocale("zh_HK")).toBe("zh-hant");
    expect(normalizeLocale("zh-MO")).toBe("zh-hant");
  });

  it("sources Intl compatibility tags from shared locale metadata", () => {
    expect(localeIntlTag("zh")).toBe("zh-Hans");
    expect(localeIntlTag("zh-hant")).toBe("zh-Hant");
    expect(localeIntlTag("de")).toBe("de");
  });

  it("deep-merges locale overrides onto the complete English source", () => {
    const catalog = resolveTranslations("zh");

    expect(catalog.chatSidebar.model).toBe("模型");
    expect(catalog.common.gateway).toBe("网关");
    expect(catalog.theme.fontTitle).toBe("字体");
  });

  it("localizes session import in the complete Simplified Chinese instance", () => {
    const simplified = resolveTranslations("zh");

    expect(simplified.sessions.importSessions).toBe("导入会话");
    expect(simplified.sessions.importComplete).toContain("{summary}");
    expect(simplified.sessions.importFailed).not.toBe(en.sessions.importFailed);
  });

  it("localizes stable MCP OAuth browser failures in Simplified Chinese", () => {
    const simplified = resolveTranslations("zh");

    expect(simplified.mcp.oauthPopupBlocked).not.toBe(en.mcp.oauthPopupBlocked);
    expect(simplified.mcp.oauthWindowClosed).not.toBe(
      en.mcp.oauthWindowClosed,
    );
  });

  it("localizes current Kanban task workflows in Simplified Chinese", () => {
    const simplified = resolveTranslations("zh");

    expect(simplified.kanban.newTaskTitle).not.toBe(en.kanban.newTaskTitle);
    expect(simplified.kanban.boardSettingsTitle).not.toBe(
      en.kanban.boardSettingsTitle,
    );
    expect(simplified.kanban.commentHintTitle).toContain("kanban_show()");
  });

  it("falls back per missing leaf for an arbitrary independent locale pack", () => {
    const catalog = resolveTranslationOverlay({
      common: { save: "Localized save" },
    });

    expect(catalog.common.save).toBe("Localized save");
    expect(catalog.chatSidebar.model).toBe(en.chatSidebar.model);
    expect(catalog.modelPicker.expensiveWarningTitle).toBe(
      en.modelPicker.expensiveWarningTitle,
    );
  });

  it("persists only display.language so concurrent config edits are preserved", async () => {
    const save = vi.spyOn(api, "saveConfig").mockResolvedValue({ ok: true });

    await persistConfiguredLocale("zh");

    expect(save).toHaveBeenCalledWith({ display: { language: "zh" } });
  });

  it("skips the full config read while its lightweight revision is unchanged", async () => {
    vi.spyOn(api, "getConfigRevision").mockResolvedValue({
      mtime_ns: 10,
      path: "/profile/config.yaml",
      size: 20,
    });
    const getConfig = vi
      .spyOn(api, "getConfig")
      .mockResolvedValue({ display: { language: "zh" } });

    const first = await readConfiguredLocaleChange(null);
    const second = await readConfiguredLocaleChange(first.revision);

    expect(getConfig).toHaveBeenCalledTimes(1);
    expect(first.locale).toBe("zh");
    expect(second).toEqual({ locale: null, revision: first.revision });
  });

  it("returns to the registry default when the configured language is removed or invalid", async () => {
    const getRevision = vi.spyOn(api, "getConfigRevision").mockResolvedValue({
      mtime_ns: 10,
      path: "/profile/config.yaml",
      size: 20,
    });
    const getConfig = vi
      .spyOn(api, "getConfig")
      .mockResolvedValueOnce({ display: {} })
      .mockResolvedValueOnce({ display: { language: "zh-unknown" } });

    const missing = await readConfiguredLocaleChange(null);
    getRevision.mockResolvedValue({
      mtime_ns: 11,
      path: "/profile/config.yaml",
      size: 21,
    });
    const invalid = await readConfiguredLocaleChange(missing.revision);

    expect(missing.locale).toBe("en");
    expect(invalid.locale).toBe("en");
  });

  it("returns a stable changed locale and defers a concurrently changing config", async () => {
    const getRevision = vi
      .spyOn(api, "getConfigRevision")
      .mockResolvedValueOnce({
        mtime_ns: 11,
        path: "/profile/config.yaml",
        size: 20,
      })
      .mockResolvedValueOnce({
        mtime_ns: 11,
        path: "/profile/config.yaml",
        size: 20,
      })
      .mockResolvedValueOnce({
        mtime_ns: 12,
        path: "/profile/config.yaml",
        size: 21,
      })
      .mockResolvedValueOnce({
        mtime_ns: 13,
        path: "/profile/config.yaml",
        size: 22,
      });
    vi.spyOn(api, "getConfig").mockResolvedValue({
      display: { language: "zh" },
    });

    const stable = await readConfiguredLocaleChange("old");
    const changing = await readConfiguredLocaleChange(stable.revision);

    expect(stable.locale).toBe("zh");
    expect(changing).toEqual({ locale: null, revision: stable.revision });
    expect(getRevision).toHaveBeenCalledTimes(4);
  });

  it("interpolates named placeholders independent of language word order", () => {
    expect(
      formatTranslation("{user} via {provider}; {missing}", {
        provider: "portal",
        user: "alice",
      }),
    ).toBe("alice via portal; {missing}");
  });

  it("keeps locale grammar out of feature components", () => {
    expect(getLocaleFormatters("en").ordinal(21)).toBe("21st");
    expect(getLocaleFormatters("zh").ordinal(21)).toBe("21");
  });

  it("falls back safely for unknown or inherited plugin navigation keys", () => {
    const catalog = resolveTranslations("zh");

    expect(resolveNavLabel(catalog, "Kanban", "kanban")).toBe("看板");
    expect(resolveNavLabel(catalog, "Plugin", "toString")).toBe("Plugin");
    expect(resolveNavLabel(catalog, "Plugin", { key: "kanban" })).toBe(
      "Plugin",
    );
    expect(
      resolvePluginDescription(catalog, "Kanban plugin", "kanban"),
    ).toContain("多 Agent 协作看板");
    expect(
      resolvePluginDescription(catalog, "Third-party plugin", "toString"),
    ).toBe("Third-party plugin");
  });
});
