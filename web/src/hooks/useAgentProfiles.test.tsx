/**
 * Unit tests for useAgentProfiles hook.
 *
 * Tests run against a mocked AgentProfilesClient so no real network calls
 * are made. The hook exposes { profiles, loading, error, refreshProfiles }.
 */
import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useAgentProfiles } from "@/hooks/useAgentProfiles";
import type { AgentProfilesClient, AgentProfilesSnapshot } from "@/hooks/useAgentProfiles";
import type { ProfileInfo } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockProfile: ProfileInfo = {
  name: "default",
  path: "/home/user/.hermes",
  is_default: true,
  model: "claude-sonnet-4-6",
  provider: "anthropic",
  has_env: true,
  skill_count: 3,
};

function makeClient(
  impl?: () => Promise<AgentProfilesSnapshot>,
): AgentProfilesClient {
  return {
    listProfiles:
      impl ??
      vi
        .fn<() => Promise<AgentProfilesSnapshot>>()
        .mockResolvedValue({ profiles: [mockProfile] }),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useAgentProfiles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with loading=true and no profiles", () => {
    // Hang indefinitely so we can observe the initial state before resolution.
    const neverResolves: () => Promise<AgentProfilesSnapshot> = () =>
      new Promise(() => {});
    const client = makeClient(neverResolves);
    const { result } = renderHook(() => useAgentProfiles(client));

    expect(result.current.loading).toBe(true);
    expect(result.current.profiles).toHaveLength(0);
    expect(result.current.error).toBeNull();
  });

  it("populates profiles after a successful fetch", async () => {
    const client = makeClient();
    const { result } = renderHook(() => useAgentProfiles(client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.profiles).toHaveLength(1);
    expect(result.current.profiles[0].name).toBe("default");
    expect(result.current.error).toBeNull();
  });

  it("surfaces a typed error string when fetch fails", async () => {
    const failing: () => Promise<AgentProfilesSnapshot> = () =>
      Promise.reject(new Error("Network failure"));
    const client = makeClient(failing);
    const { result } = renderHook(() => useAgentProfiles(client));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe("Network failure");
    expect(result.current.profiles).toHaveLength(0);
  });

  it("clears error and re-populates on a successful refreshProfiles call", async () => {
    let call = 0;
    const impl: () => Promise<AgentProfilesSnapshot> = () => {
      call++;
      if (call === 1) return Promise.reject(new Error("First failure"));
      return Promise.resolve({ profiles: [mockProfile] });
    };

    const client = makeClient(impl);
    const { result } = renderHook(() => useAgentProfiles(client));

    // Wait for the initial failing fetch to settle.
    await waitFor(() => expect(result.current.error).toBe("First failure"));

    // Manually refresh — should succeed now.
    await act(async () => {
      await result.current.refreshProfiles();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.profiles).toHaveLength(1);
  });

  it("keepExisting=true preserves existing profiles during re-fetch", async () => {
    const profilesV2: ProfileInfo[] = [{ ...mockProfile, name: "updated" }];
    let call = 0;
    const impl: () => Promise<AgentProfilesSnapshot> = () => {
      call++;
      return call === 1
        ? Promise.resolve({ profiles: [mockProfile] })
        : Promise.resolve({ profiles: profilesV2 });
    };

    const client = makeClient(impl);
    const { result } = renderHook(() => useAgentProfiles(client));

    // Wait for initial fetch.
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.profiles[0].name).toBe("default");

    // Trigger a background refresh that keeps existing profiles visible.
    await act(async () => {
      await result.current.refreshProfiles({ keepExisting: true });
    });

    expect(result.current.profiles[0].name).toBe("updated");
    expect(result.current.error).toBeNull();
  });

  it("clears cached profiles and marks unauthorized when auth fails", async () => {
    const { AuthUnauthorizedError } = await import("@/lib/api");
    let call = 0;
    const impl: () => Promise<AgentProfilesSnapshot> = () => {
      call++;
      return call === 1
        ? Promise.resolve({ profiles: [mockProfile] })
        : Promise.reject(new AuthUnauthorizedError(403));
    };

    const client = makeClient(impl);
    const { result } = renderHook(() => useAgentProfiles(client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.profiles).toHaveLength(1);

    await act(async () => {
      await result.current.refreshProfiles({ keepExisting: true });
    });

    expect(result.current.unauthorized).toBe(true);
    expect(result.current.profiles).toHaveLength(0);
    expect(result.current.error).toBe("Authentication required. Sign in again or retry the request.");
  });


  it("auto-recovers after a transient fetch failure without manual refresh", async () => {
    vi.useFakeTimers();
    let call = 0;
    const impl: () => Promise<AgentProfilesSnapshot> = () => {
      call++;
      if (call === 1) return Promise.reject(new Error("backend down"));
      return Promise.resolve({ profiles: [mockProfile] });
    };
    const client = makeClient(impl);
    const { result } = renderHook(() => useAgentProfiles(client));

    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.error).toBe("backend down");
    expect(result.current.connectionStatus).toBe("reconnecting");
    expect(result.current.nextRetryMs).toBe(2000);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
      await Promise.resolve();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.connectionStatus).toBe("connected");
    expect(result.current.profiles[0].name).toBe("default");
    expect(call).toBe(2);
  });

  it("exposes refreshProfiles for on-demand re-fetch (adapter-agnostic)", async () => {
    let callCount = 0;
    const impl: () => Promise<AgentProfilesSnapshot> = () => {
      callCount++;
      return Promise.resolve({ profiles: [mockProfile] });
    };

    const client = makeClient(impl);
    const { result } = renderHook(() => useAgentProfiles(client));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(callCount).toBe(1);

    await act(async () => {
      await result.current.refreshProfiles();
    });

    expect(callCount).toBe(2);
  });

  it("accepts a custom adapter for WebSocket/polling without changing internal state shape", async () => {
    // Simulate a WebSocket-backed adapter that resolves immediately.
    const wsProfiles: ProfileInfo[] = [{ ...mockProfile, name: "ws-profile" }];
    const wsAdapter: AgentProfilesClient = {
      listProfiles: () => Promise.resolve({ profiles: wsProfiles }),
    };

    const { result } = renderHook(() => useAgentProfiles(wsAdapter));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.profiles[0].name).toBe("ws-profile");
    // State shape is identical regardless of adapter.
    expect(result.current).toMatchObject({
      profiles: expect.any(Array),
      loading: expect.any(Boolean),
      error: null,
      refreshProfiles: expect.any(Function),
    });
  });
});
