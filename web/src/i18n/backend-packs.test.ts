import { afterEach, describe, expect, it, vi } from "vitest";

import { applyWebPack, fetchWebCatalog, fetchWebLanguages } from "./backend-packs";
import { en } from "./en";

vi.mock("@/lib/api", () => ({ fetchJSON: vi.fn() }));

const { fetchJSON } = await import("@/lib/api");
const mockedFetch = vi.mocked(fetchJSON);

afterEach(() => {
  mockedFetch.mockReset();
});

describe("applyWebPack", () => {
  it("merges a flat web layer over the base catalog, key by key", () => {
    const merged = applyWebPack(en, { "common.save": "Lưu", "common.cancel": "Huỷ" });

    expect(merged.common.save).toBe("Lưu");
    expect(merged.common.cancel).toBe("Huỷ");
    expect(merged.common.close).toBe(en.common.close); // untouched keys keep the base
    expect(en.common.save).toBe("Save"); // the bundled catalog is never mutated
  });

  it("turns a string over a function-valued entry into a positional formatter", () => {
    const merged = applyWebPack(en, {
      "pluginsPage.updateAllDone": "Đã cập nhật {0}, cần xem {1}, lỗi {2}",
    });

    expect(merged.pluginsPage.updateAllDone(3, 1, 0)).toBe("Đã cập nhật 3, cần xem 1, lỗi 0");
  });

  it("returns the base untouched when the layer is empty", () => {
    expect(applyWebPack(en, {})).toBe(en);
  });
});

describe("backend routes", () => {
  it("offers only languages some layer has web strings for", async () => {
    mockedFetch.mockResolvedValueOnce({
      languages: [
        { id: "en", endonym: "English", surfaces: ["core", "tui", "desktop"] },
        { id: "vi", endonym: "Tiếng Việt", surfaces: ["core", "web"] },
        { id: "pl", endonym: "Polski", surfaces: ["core", "tui"] },
        { bogus: true },
      ],
    });

    const rows = await fetchWebLanguages();

    expect(mockedFetch).toHaveBeenCalledWith("/api/i18n/languages");
    expect(rows?.map((row) => row.id)).toEqual(["vi"]);
  });

  it("asks for the web surface and keeps only string messages", async () => {
    mockedFetch.mockResolvedValueOnce({
      lang: "vi",
      surface: "web",
      messages: { "common.save": "Lưu", "common.count": 3 },
    });

    const messages = await fetchWebCatalog("vi");

    expect(mockedFetch).toHaveBeenCalledWith("/api/i18n/catalog?lang=vi&surface=web");
    expect(messages).toEqual({ "common.save": "Lưu" });
  });

  it("falls back silently when the backend lacks the routes (older dashboard)", async () => {
    mockedFetch.mockRejectedValue(new Error("404 Not Found"));

    expect(await fetchWebLanguages()).toBeNull();
    expect(await fetchWebCatalog("vi")).toEqual({});
  });
});
