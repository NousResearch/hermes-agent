import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../lib/api";
import { en } from "./en";
import {
  normalizeLocale,
  persistConfiguredLocale,
  resolveNavLabel,
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
    expect(normalizeLocale("zh-CN")).toBe("zh");
    expect(normalizeLocale("zh_Hans")).toBe("zh");
    expect(normalizeLocale("traditional-chinese")).toBe("zh-hant");
    expect(normalizeLocale("zh-extra")).toBeNull();
    expect(normalizeLocale("pt_BR")).toBe("pt");
  });

  it("deep-merges locale overrides onto the complete English source", () => {
    const catalog = resolveTranslations("zh");

    expect(catalog.chatSidebar.model).toBe("模型");
    expect(catalog.common.gateway).toBe("网关");
    expect(catalog.theme.fontTitle).toBe("字体");
  });

  it("persists only display.language so concurrent config edits are preserved", async () => {
    const save = vi.spyOn(api, "saveConfig").mockResolvedValue({ ok: true });

    await persistConfiguredLocale("zh");

    expect(save).toHaveBeenCalledWith({ display: { language: "zh" } });
  });

  it("falls back safely for unknown or inherited plugin navigation keys", () => {
    const catalog = resolveTranslations("zh");

    expect(resolveNavLabel(catalog, "Kanban", "kanban")).toBe("看板");
    expect(resolveNavLabel(catalog, "Plugin", "toString")).toBe("Plugin");
    expect(resolveNavLabel(catalog, "Plugin", { key: "kanban" })).toBe(
      "Plugin",
    );
  });
});
