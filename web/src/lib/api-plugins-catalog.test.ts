import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function jsonFetchMock(body: unknown = { ok: true }) {
  return vi.fn<typeof fetch>(
    async () =>
      new Response(JSON.stringify(body), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
  );
}

describe("api.getPluginsCatalog", () => {
  it("fetches the dashboard plugins catalog endpoint", async () => {
    vi.stubGlobal("window", {});

    const fetchMock = jsonFetchMock({ entries: [], removed: [], generated_at: "" });
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.getPluginsCatalog();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/dashboard/plugins/catalog",
      expect.objectContaining({ credentials: "include" }),
    );
    expect(result.entries).toEqual([]);
    expect(result.removed).toEqual([]);
  });
});

describe("api.installAgentPlugin with catalog_name", () => {
  it("posts catalog_name through to the install endpoint", async () => {
    vi.stubGlobal("window", {});

    const fetchMock = jsonFetchMock({ ok: true, plugin_name: "alpha-plugin" });
    vi.stubGlobal("fetch", fetchMock);

    await api.installAgentPlugin({
      identifier: "",
      catalog_name: "alpha-plugin",
      enable: false,
    });

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/dashboard/agent-plugins/install");
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body.catalog_name).toBe("alpha-plugin");
    expect(body.identifier).toBe("");
    expect(body.enable).toBe(false);
  });
});
