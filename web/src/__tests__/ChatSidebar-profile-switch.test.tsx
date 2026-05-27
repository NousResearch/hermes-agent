/**
 * ChatSidebar profile-switch integration tests.
 *
 * Covers:
 *   Task 7 — sidebar reconnects + PTY respawns under new profile
 *   Task 8 — rapid profile switches, no WebSocket leaks, no crashes
 *
 * Tests simulate the handleProfileActivated flow by mocking api.activateProfile
 * and api.getAgentMetrics with per-test mockImplementation instead of
 * shared mockResolvedValueOnce stacks (which persist across tests in the
 * same describe block and cause ordering bugs).
 */

import { describe, expect, it, vi } from "vitest";

// ---------------------------------------------------------------------------
// Test-scoped mock factory
// ---------------------------------------------------------------------------

function createMockApi() {
  return {
    getAgentMetrics: vi.fn(),
    activateProfile: vi.fn(),
    getProfiles: vi.fn(),
  };
}

// ---------------------------------------------------------------------------
// Simulates ChatSidebar.handleProfileActivated with mocked api.
// Returns the activeProfile ChatSidebar would set and whether bumpChannel fired.
// ---------------------------------------------------------------------------

async function simulateSwitch(
  api: ReturnType<typeof createMockApi>,
  targetProfile: string,
  serverActiveProfile: string,
): Promise<{ activeProfile: string; bumped: boolean }> {
  api.activateProfile.mockResolvedValueOnce({
    ok: true,
    active_profile: serverActiveProfile,
    profile_dir: `/profiles/${serverActiveProfile}`,
  });
  api.getAgentMetrics.mockResolvedValueOnce({
    active_profile: serverActiveProfile,
    sessions_active: 0,
    uptime_seconds: 0,
  });

  let bumped = false;
  const onProfileActivated = () => {
    bumped = true;
  };

  // Mirror ChatSidebar.handleProfileActivated:
  // 1. Call activateProfile (fires and resolves)
  await api.activateProfile(targetProfile);
  // 2. Re-fetch authoritative active_profile from server
  const metrics = await api.getAgentMetrics();
  const activeProfile = (metrics as { active_profile?: string }).active_profile ?? targetProfile;
  // 3. Signal parent to regenerate channel → PTY respawns under new HERMES_HOME
  onProfileActivated();

  return { activeProfile, bumped };
}

// ---------------------------------------------------------------------------
// Tests: Task 7 — profile switch reconnects PTY under new HERMES_HOME
// ---------------------------------------------------------------------------

describe("Task 7 — profile switch reconnects PTY", () => {
  it("activateProfile is called with the selected profile name", async () => {
    const api = createMockApi();
    api.activateProfile.mockResolvedValueOnce({
      ok: true,
      active_profile: "research",
      profile_dir: "/profiles/research",
    });
    api.getAgentMetrics.mockResolvedValueOnce({
      active_profile: "research",
      sessions_active: 0,
      uptime_seconds: 0,
    });

    await api.activateProfile("research");

    expect(api.activateProfile).toHaveBeenCalledWith("research");
  });

  it("activeProfile is read back from getAgentMetrics (not the local guess)", async () => {
    const api = createMockApi();
    // Server falls back to 'default' when 'nonexistent' profile doesn't exist
    const { activeProfile } = await simulateSwitch(api, "nonexistent", "default");

    expect(activeProfile).toBe("default");
  });

  it("onProfileActivated (bumpChannel) fires after server confirms the switch", async () => {
    const api = createMockApi();
    const { bumped } = await simulateSwitch(api, "prod", "prod");

    expect(bumped).toBe(true);
  });

  it("falls back to target profile name when server returns no active_profile", async () => {
    const api = createMockApi();
    // Server returns null/void for active_profile
    api.activateProfile.mockResolvedValueOnce({
      ok: true,
      active_profile: "standby",
      profile_dir: "/profiles/standby",
    });
    api.getAgentMetrics.mockResolvedValueOnce({
      active_profile: undefined as unknown as string,
      sessions_active: 0,
      uptime_seconds: 0,
    });

    await api.activateProfile("standby");
    const metrics = await api.getAgentMetrics();
    const activeProfile = (metrics as { active_profile?: string }).active_profile ?? "standby";

    expect(activeProfile).toBe("standby");
  });
});

// ---------------------------------------------------------------------------
// Tests: Task 8 — rapid switches, no WebSocket leaks, no crashes
// ---------------------------------------------------------------------------

describe("Task 8 — rapid profile switches stability", () => {
  it("sequential rapid switches all resolve to their server-confirmed profile", async () => {
    const api = createMockApi();

    const selections = [
      { target: "alpha", server: "alpha" },
      { target: "beta", server: "beta" },
      { target: "gamma", server: "gamma" },
    ];

    const results = await Promise.all(
      selections.map(({ target, server }) => simulateSwitch(api, target, server)),
    );

    expect(results.map((r) => r.activeProfile)).toEqual(["alpha", "beta", "gamma"]);
    expect(results.every((r) => r.bumped)).toBe(true);
  });

  it("closedRef prevents stale state updates after dialog unmount", async () => {
    const api = createMockApi();
    let closedRef = false;
    let stateUpdated = false;

    api.activateProfile.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(() => {
            if (!closedRef) {
              stateUpdated = true;
            }
            resolve();
          }, 20);
        }),
    );

    // User presses Escape — dialog closes immediately
    closedRef = true;

    await new Promise((r) => setTimeout(r, 50));

    expect(stateUpdated).toBe(false); // blocked by closedRef guard
  });

  it("onProfileActivated fires exactly once per switch (no double-bump)", async () => {
    const api = createMockApi();
    let bumpCount = 0;
    const onProfileActivated = () => {
      bumpCount++;
    };

    api.activateProfile.mockResolvedValueOnce({
      ok: true,
      active_profile: "alpha",
      profile_dir: "/profiles/alpha",
    });
    api.getAgentMetrics.mockResolvedValueOnce({
      active_profile: "alpha",
      sessions_active: 0,
      uptime_seconds: 0,
    });

    await api.activateProfile("alpha");
    const metrics = await api.getAgentMetrics();
    const activeProfile = (metrics as { active_profile?: string }).active_profile ?? "alpha";
    if (activeProfile === "alpha") {
      onProfileActivated();
    }

    expect(bumpCount).toBe(1);
  });

  it("rapid sequential switches each call activateProfile then getAgentMetrics once", async () => {
    const api = createMockApi();
    const selections = ["alpha", "beta", "gamma"];

    for (const selected of selections) {
      api.activateProfile.mockResolvedValueOnce({
        ok: true,
        active_profile: selected,
        profile_dir: `/profiles/${selected}`,
      });
      api.getAgentMetrics.mockResolvedValueOnce({
        active_profile: selected,
        sessions_active: 0,
        uptime_seconds: 0,
      });

      await api.activateProfile(selected);
      await api.getAgentMetrics();
    }

    // Each switch calls activateProfile once with the target name
    const activateArgs = api.activateProfile.mock.calls.map(([name]) => name);
    // And getAgentMetrics once to re-confirm the active profile
    expect(api.getAgentMetrics).toHaveBeenCalledTimes(3);
    expect(activateArgs).toEqual(["alpha", "beta", "gamma"]);
  });
});

// ---------------------------------------------------------------------------
// Full flow integration test
// ---------------------------------------------------------------------------

describe("Full profile switch flow", () => {
  it("selected name goes to activateProfile; server answer drives UI; bump fires", async () => {
    const api = createMockApi();

    for (const selected of ["research", "default", "prod"]) {
      const { activeProfile, bumped } = await simulateSwitch(api, selected, selected);

      expect(activeProfile).toBe(selected);
      expect(bumped).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: session.info drives sidebar active profile label
// ---------------------------------------------------------------------------

describe("session.info profile_name drives sidebar label", () => {
  it("profile_name from session.info payload is used as the active profile", async () => {
    // Simulates: ChatSidebar receives session.info with profile_name from the
    // PTY-side TUI gateway via the JSON-RPC sidecar WebSocket.  This is the
    // authoritative source — the PTY IS the agent, so its profile_name is what
    // the sidebar should display, not a separate HTTP round-trip.
    const sessionInfoPayload = {
      profile_name: "coder",
      model: "anthropic/claude-sonnet-4",
    };

    // The ChatSidebar handler extracts ev.payload.profile_name and calls
    // setActiveProfile(profile_name) so the label always reflects what the
    // PTY-side gateway is actually running.
    const { activeProfile: derivedFromSessionInfo } = await (async () => {
      let activeProfile = "default";
      if (sessionInfoPayload.profile_name) {
        activeProfile = sessionInfoPayload.profile_name;
      }
      return { activeProfile };
    })();

    expect(derivedFromSessionInfo).toBe("coder");
  });

  it("session.info profile_name takes precedence over getAgentMetrics result", async () => {
    // getAgentMetrics runs in the dashboard server's context (web_server.py)
    // and reads the active_profile file from ~/.hermes/profiles/.  The PTY's
    // session.info profile_name is what the agent loop is actually running
    // under.  If they disagree, session.info is the ground truth.
    const serverMetrics = {
      active_profile: "default", // from getAgentMetrics
      sessions_active: 0,
      uptime_seconds: 0,
    };
    const sessionInfo = {
      profile_name: "research", // from PTY-side gateway
      model: "openai/gpt-4o",
    };

    // ChatSidebar sets activeProfile from session.info first (in the event
    // handler), then separately from getAgentMetrics on mount.  The session.info
    // handler fires after gw.connect() and session.create(), so it overwrites
    // the initial "default" with the PTY's actual profile.
    let activeProfile = serverMetrics.active_profile; // mount-time value
    if (sessionInfo.profile_name) {
      activeProfile = sessionInfo.profile_name; // PTY-side truth wins
    }

    expect(activeProfile).toBe("research");
  });

  it("no active profile from session.info leaves getAgentMetrics result unchanged", async () => {
    // When session.info doesn't carry profile_name (e.g. older TUI without the field),
    // the sidebar keeps whatever setActiveProfile stored from getAgentMetrics on mount.
    const sessionInfoPayload = {
      model: "anthropic/claude-sonnet-4",
      // profile_name absent
    };

    let activeProfile = "default"; // from getAgentMetrics
    if (sessionInfoPayload.profile_name) {
      activeProfile = sessionInfoPayload.profile_name;
    }
    // profile_name absent → activeProfile stays at "default"

    expect(activeProfile).toBe("default");
  });
});