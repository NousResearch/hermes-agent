import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listAgentProfiles } from "@/lib/clients/agentProfiles";
import { buildKanbanEventsPath, fetchKanbanState } from "@/lib/clients/kanban";
import { fetchMemoryData, updateMemoryData } from "@/lib/clients/memory";
import { DashboardServiceError } from "@/lib/clients/serviceFetch";

const fetchMock = vi.fn();

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    headers: new Headers({ "content-type": "application/json" }),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response;
}

function textResponse(body: string, status = 500): Response {
  return {
    ok: false,
    status,
    statusText: "Error",
    headers: new Headers({ "content-type": "text/plain" }),
    json: () => Promise.reject(new Error("not json")),
    text: () => Promise.resolve(body),
  } as unknown as Response;
}

describe("dashboard service clients", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("VITE_AGENT_PROFILES_API_BASE_URL", "https://profiles.example.test/root/");
    vi.stubEnv("VITE_KANBAN_API_BASE_URL", "https://kanban.example.test/base");
    vi.stubEnv("VITE_MEMORY_API_BASE_URL", "https://memory.example.test/api-root/");
    window.__HERMES_SESSION_TOKEN__ = "session-token";
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    delete window.__HERMES_SESSION_TOKEN__;
  });

  it("fetches agent profiles from the configured profiles service base URL", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ profiles: [{ name: "coder", path: "/tmp/coder", is_default: false, model: null, provider: null, has_env: true, skill_count: 4 }] }));

    const result = await listAgentProfiles();

    expect(result.profiles[0].name).toBe("coder");
    expect(fetchMock).toHaveBeenCalledWith(
      "https://profiles.example.test/root/api/profiles",
      expect.objectContaining({ credentials: "include", headers: expect.any(Headers) }),
    );
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("X-Hermes-Session-Token")).toBe("session-token");
  });

  it("fetches Kanban board state from the configured Kanban service base URL", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      columns: [{ name: "running", tasks: [{ id: "t_1", title: "Build", status: "running", assignee: "coder", priority: 2 }] }],
      tenants: ["global"],
      assignees: ["coder"],
      latest_event_id: 12,
      now: 1234,
      board: "ai-dev",
    }));

    const result = await fetchKanbanState({ board: "ai-dev" });

    expect(result.columns[0].status).toBe("running");
    expect(fetchMock).toHaveBeenCalledWith(
      "https://kanban.example.test/base/api/plugins/kanban/board?board=ai-dev",
      expect.any(Object),
    );
    expect(buildKanbanEventsPath({ board: "ai-dev", since: 12 })).toBe(
      "/api/plugins/kanban/events?since=12&board=ai-dev",
    );
  });

  it("fetches and updates memory through the configured memory service base URL", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ content: "remember this", entries: [{ text: "remember this" }], char_count: 13, char_limit: 2200, target: "memory" }))
      .mockResolvedValueOnce(jsonResponse({ ok: true, char_count: 14, char_limit: 2200 }));

    const snapshot = await fetchMemoryData("memory");
    const update = await updateMemoryData("user", "updated memory");

    expect(snapshot.entries[0].text).toBe("remember this");
    expect(update.success).toBe(true);
    expect(fetchMock.mock.calls[0][0]).toBe("https://memory.example.test/api-root/api/memory/content?target=memory");
    expect(fetchMock.mock.calls[1][0]).toBe("https://memory.example.test/api-root/api/memory/content");
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "PUT" }));
  });

  it("throws a predictable typed error for non-2xx service responses", async () => {
    fetchMock.mockResolvedValueOnce(textResponse("backend unavailable", 503));

    const promise = listAgentProfiles();

    await expect(promise).rejects.toMatchObject({
      name: "DashboardServiceError",
      status: 503,
      body: "backend unavailable",
    });

    fetchMock.mockResolvedValueOnce(textResponse("backend unavailable", 503));
    await expect(listAgentProfiles()).rejects.toBeInstanceOf(DashboardServiceError);
  });
});
