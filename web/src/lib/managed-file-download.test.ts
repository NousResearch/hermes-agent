import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

it.each(["", "/hermes/"])("streams through a path-bound ticket under %s", async (basePath) => {
  vi.resetModules();
  vi.stubGlobal("window", { __HERMES_BASE_PATH__: basePath, __HERMES_SESSION_TOKEN__: "session-secret" });
  const request = vi.fn<(url: string, init: RequestInit) => Promise<Response>>()
    .mockResolvedValue(Response.json({ ticket: "file-ticket" }));
  vi.stubGlobal("fetch", request);
  const { api } = await import("./api");
  const path = "/workspace/보고서 & 1.mp3";

  const url = await api.fileDownloadUrl(path);

  const prefix = basePath.replace(/\/$/, "");
  expect(request).toHaveBeenCalledOnce();
  const [endpoint, init] = request.mock.calls[0];
  expect(endpoint).toBe(`${prefix}/api/files/ticket`);
  expect(init.method).toBe("POST");
  expect(JSON.parse(init.body as string)).toEqual({ route: "managed_download", path });
  expect(new Headers(init.headers).get("X-Hermes-Session-Token")).toBe("session-secret");
  expect(url).toBe(`${prefix}/api/files/download?path=%2Fworkspace%2F%EB%B3%B4%EA%B3%A0%EC%84%9C+%26+1.mp3&ticket=file-ticket`);
  expect(url).not.toContain("session-secret");
});

it("surfaces ticket failures before opening the download", async () => {
  vi.stubGlobal("window", {});
  vi.stubGlobal("fetch", vi.fn(async () => Response.json({ detail: "Download denied" }, { status: 403 })));
  const { api } = await import("./api");
  await expect(api.fileDownloadUrl("/workspace/report.txt")).rejects.toThrow("Download denied");
});
