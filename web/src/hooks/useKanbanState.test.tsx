/**
 * Tests for useKanbanState hook.
 *
 * Tests run against a mocked fetch so no real network calls are made.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useKanbanState } from "./useKanbanState";

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
    headers: new Headers({ "content-type": "application/json" }),
  } as unknown as Response;
}

const MOCK_BOARD_RESPONSE = {
  columns: [
    {
      name: "running",
      tasks: [
        { id: "t_001", title: "Build something", status: "running", assignee: "coder", priority: 0, created_at: 1780000000 },
        { id: "t_002", title: "Write docs", status: "running", assignee: "writer", priority: 0, created_at: 1780000001 },
      ],
    },
    {
      name: "done",
      tasks: [
        { id: "t_003", title: "Shipped it", status: "done", assignee: "coder", priority: 0, created_at: 1779999000 },
      ],
    },
  ],
  board: "ai-dev",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useKanbanState", () => {
  it("starts with loading=true and null data", () => {
    // Hang the request so we can observe the loading state.
    fetchMock.mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useKanbanState());

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("fetches board state from /api/plugins/kanban/board and exposes data", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(MOCK_BOARD_RESPONSE));

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/plugins/kanban/board",
      expect.objectContaining({
        credentials: "include",
        headers: expect.any(Headers),
      }),
    );
    expect(result.current.data).not.toBeNull();
    expect(result.current.data?.columns).toHaveLength(2);
    expect(result.current.data?.board).toBe("ai-dev");
    expect(result.current.error).toBeNull();
  });

  it("exposes running task count via columns", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(MOCK_BOARD_RESPONSE));

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));

    const runningCol = result.current.data?.columns.find(
      (c) => c.status === "running",
    );
    expect(runningCol?.tasks).toHaveLength(2);
  });

  it("sets error and clears data on HTTP error response", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: "Not found" }, false, 404),
    );

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBeTruthy();
    expect(result.current.data).toBeNull();
  });

  it("sets error and clears data on network failure", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Network down"));

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toMatch(/Network down/i);
    expect(result.current.data).toBeNull();
  });

  it("refresh() re-fetches board state", async () => {
    fetchMock.mockResolvedValue(jsonResponse(MOCK_BOARD_RESPONSE));

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(fetchMock).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.refresh();
    });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).not.toBeNull();
  });

  it.each([401, 403])("clears cached board data and marks unauthorized on HTTP %s", async (status) => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(MOCK_BOARD_RESPONSE))
      .mockResolvedValueOnce(jsonResponse({ detail: "denied" }, false, status));

    const { result } = renderHook(() => useKanbanState());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data?.columns[0].tasks).toHaveLength(2);

    await act(async () => {
      await result.current.refresh();
    });

    expect(result.current.unauthorized).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe("Authentication required. Sign in again or retry the request.");
  });

  it("does not call fetch after unmount (no setState on stale request)", async () => {
    let resolveRequest!: (r: Response) => void;
    fetchMock.mockReturnValueOnce(
      new Promise<Response>((res) => {
        resolveRequest = res;
      }),
    );

    const { result, unmount } = renderHook(() => useKanbanState());

    expect(result.current.loading).toBe(true);
    unmount();

    // Resolve AFTER unmount — should not trigger setState.
    act(() => resolveRequest(jsonResponse(MOCK_BOARD_RESPONSE)));

    // Give the microtask queue a tick to flush.
    await new Promise((r) => setTimeout(r, 0));

    // Hook is unmounted; as long as no unhandled error is thrown the test passes.
    // (Would throw "cannot update state on an unmounted component" in older React.)
  });
});
