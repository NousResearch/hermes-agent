/**
 * Unit tests for useMemoryData hook.
 *
 * Tests run against a mocked MemoryDataClient so no real network calls
 * are made. The hook exposes { data, loading, error, refetch, save }.
 */
import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useMemoryData } from "@/hooks/useMemoryData";
import type { MemoryDataClient, MemorySnapshot, MemoryTarget } from "@/hooks/useMemoryData";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockSnapshot: MemorySnapshot = {
  content: "entry one\n§\nentry two",
  entries: [{ text: "entry one" }, { text: "entry two" }],
  char_count: 21,
  char_limit: 2200,
  target: "memory",
};

const mockUpdateResult = {
  success: true,
  char_count: 21,
  char_limit: 2200,
};

function makeClient(overrides?: {
  fetchMemoryState?: (target: MemoryTarget) => Promise<MemorySnapshot>;
  updateMemoryState?: (target: MemoryTarget, content: string) => Promise<typeof mockUpdateResult>;
}): MemoryDataClient {
  return {
    fetchMemoryState:
      overrides?.fetchMemoryState ??
      vi
        .fn<(target: MemoryTarget) => Promise<MemorySnapshot>>()
        .mockResolvedValue(mockSnapshot),
    updateMemoryState:
      overrides?.updateMemoryState ??
      vi
        .fn<(target: MemoryTarget, content: string) => Promise<typeof mockUpdateResult>>()
        .mockResolvedValue(mockUpdateResult),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useMemoryData", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts with loading=true and data=null", () => {
    const neverResolves = (): Promise<MemorySnapshot> => new Promise(() => {});
    const client = makeClient({ fetchMemoryState: neverResolves });
    const { result } = renderHook(() => useMemoryData("memory", client));

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("populates data after a successful fetch", async () => {
    const client = makeClient();
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).not.toBeNull();
    expect(result.current.data?.entries).toHaveLength(2);
    expect(result.current.data?.entries[0].text).toBe("entry one");
    expect(result.current.data?.char_limit).toBe(2200);
    expect(result.current.error).toBeNull();
  });

  it("surfaces a typed error string when fetch fails", async () => {
    const failing = (): Promise<MemorySnapshot> =>
      Promise.reject(new Error("Network failure"));
    const client = makeClient({ fetchMemoryState: failing });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe("Network failure");
    expect(result.current.data).toBeNull();
  });

  it("handles non-Error thrown values gracefully", async () => {
    const client = makeClient({
      fetchMemoryState: () => Promise.reject("string error"),
    });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe("string error");
    expect(result.current.data).toBeNull();
  });

  it("clears error and re-populates on a successful refetch call", async () => {
    let call = 0;
    const impl = (): Promise<MemorySnapshot> => {
      call++;
      if (call === 1) return Promise.reject(new Error("First failure"));
      return Promise.resolve(mockSnapshot);
    };
    const client = makeClient({ fetchMemoryState: impl });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.error).toBe("First failure"));

    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.data?.entries).toHaveLength(2);
  });

  it("keepExisting=true preserves existing data during re-fetch", async () => {
    const snapshotV2: MemorySnapshot = {
      ...mockSnapshot,
      content: "updated",
      entries: [{ text: "updated" }],
    };
    let call = 0;
    const impl = (): Promise<MemorySnapshot> => {
      call++;
      return call === 1
        ? Promise.resolve(mockSnapshot)
        : Promise.resolve(snapshotV2);
    };
    const client = makeClient({ fetchMemoryState: impl });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data?.entries[0].text).toBe("entry one");

    await act(async () => {
      await result.current.refetch({ keepExisting: true });
    });

    expect(result.current.data?.entries[0].text).toBe("updated");
    expect(result.current.error).toBeNull();
  });

  it("refetch is callable on demand and triggers another API call", async () => {
    let callCount = 0;
    const impl = (): Promise<MemorySnapshot> => {
      callCount++;
      return Promise.resolve(mockSnapshot);
    };
    const client = makeClient({ fetchMemoryState: impl });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(callCount).toBe(1);

    await act(async () => {
      await result.current.refetch();
    });

    expect(callCount).toBe(2);
  });

  it("save calls updateMemoryState then refetches", async () => {
    const updateFn = vi
      .fn<(target: MemoryTarget, content: string) => Promise<typeof mockUpdateResult>>()
      .mockResolvedValue(mockUpdateResult);
    let fetchCallCount = 0;
    const fetchFn = vi
      .fn<(target: MemoryTarget) => Promise<MemorySnapshot>>()
      .mockImplementation(() => {
        fetchCallCount++;
        return Promise.resolve(mockSnapshot);
      });

    const client = makeClient({ fetchMemoryState: fetchFn, updateMemoryState: updateFn });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(fetchCallCount).toBe(1); // initial fetch

    await act(async () => {
      await result.current.save("new content");
    });

    expect(updateFn).toHaveBeenCalledWith("memory", "new content");
    expect(fetchCallCount).toBeGreaterThanOrEqual(2); // save triggered refetch
    expect(result.current.error).toBeNull();
  });

  it("save surfaces an error if updateMemoryState rejects", async () => {
    const client = makeClient({
      updateMemoryState: () => Promise.reject(new Error("Save failed")),
    });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.save("bad content");
    });

    expect(result.current.error).toBe("Save failed");
  });

  it("clears cached memory data and marks unauthorized when auth fails", async () => {
    const { AuthUnauthorizedError } = await import("@/lib/api");
    let call = 0;
    const client = makeClient({
      fetchMemoryState: () => {
        call++;
        return call === 1
          ? Promise.resolve(mockSnapshot)
          : Promise.reject(new AuthUnauthorizedError(401));
      },
    });
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data?.entries).toHaveLength(2);

    await act(async () => {
      await result.current.refetch({ keepExisting: true });
    });

    expect(result.current.unauthorized).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe("Authentication required. Sign in again or retry the request.");
  });

  it("works with the 'user' memory target", async () => {
    const userSnapshot: MemorySnapshot = {
      ...mockSnapshot,
      target: "user",
      char_limit: 1375,
    };
    const client = makeClient({
      fetchMemoryState: vi.fn().mockResolvedValue(userSnapshot),
    });
    const { result } = renderHook(() => useMemoryData("user", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data?.target).toBe("user");
    expect(result.current.data?.char_limit).toBe(1375);
  });

  it("accepts a custom adapter for WebSocket/SSE/polling without changing state shape", async () => {
    const wsSnapshot: MemorySnapshot = {
      ...mockSnapshot,
      content: "ws content",
      entries: [{ text: "ws content" }],
    };
    const wsAdapter: MemoryDataClient = {
      fetchMemoryState: () => Promise.resolve(wsSnapshot),
      updateMemoryState: () => Promise.resolve(mockUpdateResult),
    };

    const { result } = renderHook(() => useMemoryData("memory", wsAdapter));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data?.content).toBe("ws content");
    // State shape is stable regardless of adapter.
    expect(result.current).toMatchObject({
      data: expect.any(Object),
      loading: expect.any(Boolean),
      error: null,
      refetch: expect.any(Function),
      save: expect.any(Function),
    });
  });

  it("exposes both refetch and save functions (real-time integration point)", async () => {
    const client = makeClient();
    const { result } = renderHook(() => useMemoryData("memory", client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    // A future WS/SSE layer can call refetch() from a subscription callback.
    expect(typeof result.current.refetch).toBe("function");
    // The save path is separate so UI call sites are not coupled to transport.
    expect(typeof result.current.save).toBe("function");
  });
});
